import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-node";
import iris from "../data/iris.json";
import irisTesting from "../data/iris-testing.json";

// set up data
const trainingData = tf.tensor2d(
  iris.map(item => [
    parseInt(item.sepal_length),
    parseInt(item.sepal_width),
    parseInt(item.petal_length),
    parseInt(item.petal_width)
  ])
);
const outputData = tf.tensor2d(
  iris.map(item => [
    item.species === "setosa" ? 1 : 0,
    item.species === "virginica" ? 1 : 0,
    item.species === "versicolor" ? 1 : 0
  ])
);
const testingData = tf.tensor2d(
  irisTesting.map(item => [
    parseInt(item.sepal_length),
    parseInt(item.sepal_width),
    parseInt(item.petal_length),
    parseInt(item.petal_width)
  ])
);

// build neural network
const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5
  })
);
model.add(
  tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3
  })
);
model.add(
  tf.layers.dense({
    activation: "sigmoid",
    units: 3
  })
);

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(0.06)
});

// train network
const startTime = Date.now();
model
  .fit(trainingData, outputData, { epochs: 100 })
  .then(history => {
    // console.log("history :", history);
    console.log("🎉 Finished training model!", Date.now() - startTime, "ms");

    // test network
    console.log();
    console.log("Run prediction on testing data");
    model.predict(testingData).print();
  })
  .catch(err => {
    console.log("err :", err);
  });
