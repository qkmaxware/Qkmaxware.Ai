using static System.Console;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Qkmaxware.Ai.NeuralNetwork;
using Qkmaxware.Ai.NeuralNetwork.Activation;
using Qkmaxware.Ai.NeuralNetwork.Training;

namespace Qkmaxware.Ai.Test;

[TestClass]
public class XorTests {
    [TestMethod]
    public void TestManualXor() {
        // Create neural network
        // In1 \- Hidden1 \
        //      x          Out
        // In2 /- Hidden2 /
        var nn = new FeedForwardNeuralNetwork(activationFunction: new Tanh(), 2, 2, 1);

        nn.SetSynapseWeight(0, 0, 1, 0, 1);
        nn.SetSynapseWeight(0, 0, 1, 1, -1);
        nn.SetSynapseWeight(0, 1, 1, 0, 1);
        nn.SetSynapseWeight(0, 1, 1, 1, -1);

        nn.SetSynapseWeight(1, 0, 2, 0, 1);
        nn.SetSynapseWeight(1, 1, 2, 0, 1);

        // Run it
        const int TRUE = 1;
        const int FALSE = -1;
        var training = new ListTrainingSet();
        training.Add(new TaggedVector(new double[]{FALSE, FALSE}, "In1", "In2"), new TaggedVector(new double[]{ FALSE }, "Out"));
        training.Add(new TaggedVector(new double[]{FALSE, TRUE}, "In1", "In2"), new TaggedVector(new double[]{ TRUE }, "Out"));
        training.Add(new TaggedVector(new double[]{TRUE, FALSE}, "In1", "In2"), new TaggedVector(new double[]{ TRUE }, "Out"));
        training.Add(new TaggedVector(new double[]{TRUE, TRUE}, "In1", "In2"), new TaggedVector(new double[]{ FALSE }, "Out"));
        foreach (var set in training) {
            var result = nn.Evaluate(set.FeatureVector);
            for (var i = 0; i < result.Count; i++) {
                Assert.AreEqual(set.ResultVector[i], result[i], $"{set.ResultVector} != {result} for input {set.FeatureVector}");
            }
        }
    }

    [TestMethod]
    public void TestTrainedXor() {
        // Create training data
        const int TRUE = 1;
        const int FALSE = -1;
        var training = new ListTrainingSet();
        training.Add(new TaggedVector(new double[]{FALSE, FALSE}, "In1", "In2"), new TaggedVector(new double[]{ FALSE }, "Out"));
        training.Add(new TaggedVector(new double[]{FALSE, TRUE}, "In1", "In2"), new TaggedVector(new double[]{ TRUE }, "Out"));
        training.Add(new TaggedVector(new double[]{TRUE, FALSE}, "In1", "In2"), new TaggedVector(new double[]{ TRUE }, "Out"));
        training.Add(new TaggedVector(new double[]{TRUE, TRUE}, "In1", "In2"), new TaggedVector(new double[]{ FALSE }, "Out"));

        // Create neural network
        // In1 \- Hidden1 \
        //      x          Out
        // In2 /- Hidden2 /
        var nn = new FeedForwardNeuralNetwork(activationFunction: new Tanh(), 2, 2, 1);

        // Training the neural network (also validates the results)
        var trainer = new StochasticBackPropagationTrainer(
            epochs: 100, 
            iterationsPerEpoch: 500, 
            learningRate: 0.1, 
            desiredAccuracy: 0.1, 

            trainingData: training,
            validationData: training
        );
        Assert.AreEqual(true, trainer.TryTrain(nn));
    }
}