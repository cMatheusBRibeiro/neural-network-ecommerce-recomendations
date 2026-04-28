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

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(users, catalog) {
    const ages = users.map((u) => u.age);
    const prices = catalog.map((p) => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    const categories = [...new Set(catalog.map((p) => p.category))];
    const colors = [...new Set(catalog.map((p) => p.color))];
    
    const categoryIndex = Object.fromEntries(categories.map((category, index) => [category, index]));
    const colorIndex = Object.fromEntries(colors.map((color, index) => [color, index]));
    
    // Computar a média de idade dos compradores por produto
    // (ajuda a personalizar)
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
        catalog.map((p) => {
            const avg = ageCountByProduct[p.name]
                ? ageSumsByProduct[p.name] / ageCountByProduct[p.name]
                : midAge;
            return [p.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        catalog,
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
        context.numCategories,
        WEIGHTS.color
    );

    return tf.concat1d([price, age, category, color]);
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const catalog = await (await fetch("/data/products.json")).json();
    
    const context = makeContext(users, catalog);

    context.productVectors = catalog.map((p) => ({
        product: p.name,
        meta: {...p},
        vector: encodeProduct(p, context).dataSync(),
    }));

    _globalCtx = context;

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

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
