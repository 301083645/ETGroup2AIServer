//
//https://github.com/PacktPublishing/Hands-on-Machine-Learning-with-TensorFlow.js/tree/master/Section5_4
//
const tf = require('@tensorflow/tfjs');
    // require('@tensorflow/tfjs-node');
    //load iris training and testing data
    const iris = require('../../iris.json');
    const irisTesting = require('../../iris-testing.json');
    var lossValue;
    //
exports.trainAndPredict = function (req, res) {
    
    const dyspnoea = parseFloat(req.body.dyspnoea);
    const cough = parseFloat(req.body.cough);
    const chestpain = parseFloat(req.body.chestpain);
    const sputum = parseFloat(req.body.sputum);
    const epochs = 100;
    const learning_rate = 0.03;
    console.log("Testing Learning Rate: " + learning_rate);
    console.log("Testing Epochs: " + epochs);

    console.log(irisTesting)
    //
    // convert/setup our data for tensorflow.js
    //
    //tensor of features for training data
    // include only features, not the output
    const trainingData = tf.tensor2d(iris.map(item => [
        item.cough, item.dyspnoea, item.sputum,
        item.chestpain
    ]))
    //console.log(trainingData.dataSync())
    //
    //tensor of output for training data
    //the values for species will be:
    // setosa:       1,0,0
    // virginica:    0,1,0
    // versicolor:   0,0,1
    const outputData = tf.tensor2d(iris.map(item => [
        item.diagnosis === "cold" ? 1 : 0,
        item.diagnosis === "pneumonia" ? 1 : 0,
        item.diagnosis === "lungCancer" ? 1 : 0
    ]))
    //console.log(outputData.dataSync())
    //
    //tensor of features for testing data
    /*
    const testingData = tf.tensor2d(irisTesting.map(item => [
        item.sepal_length, item.sepal_width,
        item.petal_length, item.petal_width,
    ]))
    */
   const testingData = tf.tensor2d([[cough, dyspnoea, sputum, chestpain]])
    //console.log(testingData.dataSync())    
    //
    // build neural network using a sequential model
    const model = tf.sequential()
    //add the first layer
    model.add(tf.layers.dense({
        inputShape: [4], // four input neurons
        activation: "sigmoid",
        units: 5, //dimension of output space (first hidden layer)
    }))
    //add the hidden layer
    model.add(tf.layers.dense({
        inputShape: [5], //dimension of hidden layer
        activation: "sigmoid",
        units: 3, //dimension of final output (setosa, virginica, versicolor)
    }))
    //add output layer
    model.add(tf.layers.dense({
        activation: "sigmoid",
        units: 3, //dimension of final output (setosa, virginica, versicolor)
    }))
    //compile the model with an MSE loss function and Adam algorithm
    model.compile({
        loss: "meanSquaredError",
        optimizer: tf.train.adam(learning_rate),
    })
    console.log(model.summary())
    //
    //Train the model and predict the results for testing data
    //
    // train/fit the model for the fixed number of epochs
    async function run() {
        const startTime = Date.now()
        //train the model
        await model.fit(trainingData, outputData,         
            {
                epochs,
                callbacks: { //list of callbacks to be called during training
                    onEpochEnd: async (epoch, log) => {
                        lossValue = log.loss;
                        console.log(`Epoch ${epoch}: lossValue = ${log.loss}`);
                        elapsedTime = Date.now() - startTime;
                        console.log('elapsed time: ' + elapsedTime)
                    }
                }
            }
            
        )
            
        const results = model.predict(testingData);
        //console.log('prediction results: ', results.dataSync())
        //results.print()
        
        // get the values from the tf.Tensor
        //var tensorData = results.dataSync();
        results.array().then(resultArray => {
            console.log(resultArray)
            res.status(200).send(resultArray);
            //
            
            // res.render('results',
            //     {
            //         elapsedTime: elapsedTime / 1000,
            //         lossValue: lossValue,
            //         resultForData1: resultForData1[0],
            //         resultForData2: resultForData2,
            //         resultForData3: resultForData3
            //     }
            // )
                                
            //
        })
        //

    } //end of run function
    run()

};

