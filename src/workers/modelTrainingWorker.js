import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

async function makeContext(users, catalog) {
    const ages = users.map((u) => u.age);
    const prices = catalog.map((p) => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    const categories = [...new Set(catalog.map((p) => p.category))];
    const colors = [...new Set(catalog.map((p) => p.color))];
    
    const categoryIndex = Object.entries(categories.map((category, index) => [category, index]));
    const colorIndex = Object.entries(colors.map((color, index) => [color, index]));
    
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
        dimensions: 1 + 1 + colors.length + categories.length
    };
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const catalog = await (await fetch("/data/products.json")).json();
    
    const context = await makeContext(users, catalog);

    debugger

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
