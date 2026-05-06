import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = {};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(users, products) {
    /**
     * Aqui ele cria algumas informações bases, idades, preços, idade mínima, máxima, preço mínimo, máximo,
     * criação de listagem sem repetições de categorias e cores, e dicionário de categorias e cores com suas
     * posições.
     */
    const ages = users.map((u) => u.age);
    const prices = products.map((p) => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    const categories = [...new Set(products.map((p) => p.category))];
    const colors = [...new Set(products.map((p) => p.color))];
    
    const categoryIndex = Object.fromEntries(categories.map((category, index) => [category, index]));
    const colorIndex = Object.fromEntries(colors.map((color, index) => [color, index]));
    
    /**
     * Computar a média de idade dos compradores por produto (ajuda a personalizar as sugestões).
     */
    const midAge = (minAge + maxAge) / 2;
    const ageSumsByProduct = {};
    const ageCountByProduct = {};

    users.forEach((u) => {
        u.purchases.forEach((p) => {
            ageSumsByProduct[p.name] = (ageSumsByProduct[p.name] || 0) + u.age;
            ageCountByProduct[p.name] = (ageCountByProduct[p.name] || 0) + 1;
        });
    });

    const productAvgAgeNorm = Object.fromEntries(
        products.map((p) => {
            const avg = ageCountByProduct[p.name]
                ? ageSumsByProduct[p.name] / ageCountByProduct[p.name]
                : midAge;
            return [p.name, normalize(avg, minAge, maxAge)];
        })
    );

    /**
     * Retorno necessário para termos o contexto no sistema, utilizado nos encodes.
     */
    return {
        products,
        users,
        colorIndex,
        categoryIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // 1 (age) + 1 (price) + colors.length + categories.length
        dimensions: 1 + 1 + colors.length + categories.length,
        productAvgAgeNorm,
    };
}

const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, context) {
    /**
     * O encode do produto é baseado no preço, idade média dos compradores, categoria
     * e cor. Categoria e cor são textos, mas aqui foram normalizados para one hot.
     */

    const price = tf.tensor1d([normalize(
        product.price,
        context.minPrice,
        context.maxPrice
    ) * WEIGHTS.price]);

    const age = tf.tensor1d([normalize(
        context.productAvgAgeNorm[product.name],
        context.minAge,
        context.maxAge
    ) * WEIGHTS.age]);

    const category = oneHotWeighted(
        context.categoryIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );

    const color = oneHotWeighted(
        context.colorIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );

    /**
     * Aqui concatenamos as informações levantadas em um único tensor.
     */
    return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, context) {
    /**
     * O encode do user é baseado nas compras feitas, criando uma lista de tensores com base em suas
     * compras, tira uma média e, depois, recria o tensor, com as dimensões apresentadas no contexto
     */
    if (user.purchases) {
        return tf.stack(
            user.purchases.map((product) => encodeProduct(product, context))
        )
        .mean(0)
        .reshape([
            1,
            context.dimensions
        ]);
    }
}

function createTrainingData(context) {
    /**
     * Os dados do treinamento são criados com base nos inputs e labels, sendo os dados de entrada e os resultados
     * esperados para treinar o modelo.
     * Dito isso, como é um sistema de recomendação, as informações são com base nos usuários e suas compras.
     * Primeiramente é criado o vetor do usuário, com encodeUser, dataSync é uma função que transforma um tensor
     * em um vetor. Após isso, para cada produto é criado um vetor e verificado se o usuário possui aquele produto
     * dentro de suas compras, criando um label.
     * Feito isso, um input é inserido, associando o usuário ao produto, e o label é salvo, se houve a compra do
     * produto ou não, assim criando um registro para o modelo de compra ou não compra.
     * Ao final, são retornados os inputs, labels e inputDimension.
     */

    const inputs = [];
    const labels = [];
    context.users.filter((user) => user.purchases.length).forEach((user) => {
        const userVector = encodeUser(user, context).dataSync();
        context.products.forEach((product) => {
            const productVector = encodeProduct(product, context).dataSync();
            const label = user.purchases.some((purchase) => purchase.name === product.name ? 1 : 0);

            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        });
    });
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        /**
         * Esse inputDimension é 2x a dimensão do contexto pois temos as dimensões do usuário + dimensões do produto.
         */
        inputDimension: context.dimensions * 2,
    };
}

async function configureNeuralNetAndTrain(trainData) {
    /**
     * Aqui é criado o modelo e adicionado 4 camadas nele:
     * 1. Camada de entrada com 128 neurônios, o inputShape passando a quantidade de dimensões e activation "relu",
     * que mantém os sinais positivos, ajudando a aprender padrões não lineares;
     * 2. Camada oculta 1, com 64 neurônios para comprimir as informções e activation "relu";
     * 3. Camada oculta 2, com 32 neurônios para destilar as informações mais importantes, mantendo os padrões
     * mais fortes, com activation "relu";
     * 4. Camada de saída, onde temos apenas 1 neurônio para pontuação da recomendação, activation "sigmoid" que
     * irá comprimir o resultado entre 0 e 1, se for 0.1 é recomendação fraca, se for 0.9 é recomendação forte.
     * Após isso o modelo é compilado e treinado.
     */

    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimension],
            units: 128,
            activation: "relu",
        })
    );

    model.add(
        tf.layers.dense({
            units: 64,
            activation: "relu",
        })
    );

    model.add(
        tf.layers.dense({
            units: 32,
            activation: "relu",
        })
    );

    model.add(
        tf.layers.dense({
            units: 1,
            activation: "sigmoid",
        })
    );

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
    });

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc,
                });
            },
        },
    });
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const products = await (await fetch("/data/products.json")).json();
    
    const context = makeContext(users, products);

    context.productVectors = products.map((p) => ({
        product: p.name,
        meta: {...p},
        vector: encodeProduct(p, context).dataSync(),
    }));

    _globalCtx = context;

    const trainData = createTrainingData(context);

    const model = await configureNeuralNetAndTrain(trainData);

    _model = model;

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
