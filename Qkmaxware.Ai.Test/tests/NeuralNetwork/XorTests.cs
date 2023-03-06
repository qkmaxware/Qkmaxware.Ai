using static System.Console;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Qkmaxware.Ai.NeuralNetwork;
using Qkmaxware.Ai.NeuralNetwork.Activation;
using Qkmaxware.Ai.NeuralNetwork.Training;
using System.Collections.Generic;

namespace Qkmaxware.Ai.Test;

[TestClass]
public class XorTests {
    [TestMethod]
    public void ManualWeights() {
        // Create neural network
        // In1 \- Hidden1 \
        //      x          Out
        // In2 /- Hidden2 /
        var nn = new FeedForwardNeuralNetwork(activationFunction: new Signum(), 2, 2, 1);

        nn.SetNeuronBias((NeuronId)2, 0.5);
        nn.SetNeuronBias((NeuronId)3, -1.5);
        nn.SetSynapseWeight((NeuronId)0, (NeuronId)2,  1);
        nn.SetSynapseWeight((NeuronId)0, (NeuronId)3,  1);
        nn.SetSynapseWeight((NeuronId)1, (NeuronId)2,  1);
        nn.SetSynapseWeight((NeuronId)1, (NeuronId)3,  1);

        nn.SetNeuronBias((NeuronId)4, -1);
        nn.SetSynapseWeight((NeuronId)2, (NeuronId)4,  0.7);
        nn.SetSynapseWeight((NeuronId)3, (NeuronId)4,  -0.4);

        // Run it
        const int TRUE = 1;
        const int FALSE = -1;
        var training = new ListTrainingSet();
        training.Add(new TaggedVector<string>(("In1", FALSE), ("In2", FALSE)), new TaggedVector<string>(("Out", FALSE)));
        training.Add(new TaggedVector<string>(("In1", FALSE), ("In2", TRUE)) , new TaggedVector<string>(("Out", TRUE)));
        training.Add(new TaggedVector<string>(("In1", TRUE),  ("In2", FALSE)), new TaggedVector<string>(("Out", TRUE)));
        training.Add(new TaggedVector<string>(("In1", TRUE),  ("In2", TRUE)) , new TaggedVector<string>(("Out", FALSE)));
        foreach (var set in training) {
            var result = nn.Evaluate(set.FeatureVector).RemapTags(x => "Out");
            for (var i = 0; i < result.Count; i++) {
                Assert.AreEqual(set.ResultVector[i], System.Math.Round(result[i]), $"{set.ResultVector} != {result} for input {set.FeatureVector}");
            }
        }
    }

    [TestMethod]
    public void BackPropagationWeights() {
        // Create training data
        const int TRUE = 1;
        const int FALSE = -1;
        var training = new ListTrainingSet();
        training.Add(new TaggedVector<string>(("In1", FALSE), ("In2", FALSE)), new TaggedVector<string>(("Out", FALSE)));
        training.Add(new TaggedVector<string>(("In1", FALSE), ("In2", TRUE)) , new TaggedVector<string>(("Out", TRUE)));
        training.Add(new TaggedVector<string>(("In1", TRUE),  ("In2", FALSE)), new TaggedVector<string>(("Out", TRUE)));
        training.Add(new TaggedVector<string>(("In1", TRUE),  ("In2", TRUE)) , new TaggedVector<string>(("Out", FALSE)));

        // Create neural network
        // In1 \- Hidden1 \
        //      x          Out
        // In2 /- Hidden2 /
        IBackPropagableNeuralNetwork nn = new FeedForwardNeuralNetwork(activationFunction: new Exponential(), 2, 2, 1);
        nn.RandomizeSynapseWeights();

        // Training the neural network (also validates the results)
        var trainer = new StochasticBackPropagationTrainer(
            epochs: 100, 
            iterationsPerEpoch: 500, 
            learningRate: 0.1, 
            desiredAccuracy: 0.1, 

            trainingData: training,
            validationData: training
        );
        Assert.AreEqual(true, trainer.TryTrain(nn, out nn));
    }

    private class XorGenerator : IGenomeGenerator<FeedForwardNeuralNetworkGenome> {
        private BaseActivationFunction activationFunction = new Exponential();
        public IEnumerable<FeedForwardNeuralNetworkGenome> Generate(int max) {
            for (var i = 0; i < max; i++) {
                var nn = new FeedForwardNeuralNetwork(activationFunction: activationFunction, 2, 2, 1);
                nn.RandomizeSynapseWeights();
                yield return nn.EncodeToGenome();
            }
        }
    }

    [TestMethod]
    public void GeneticWeights() {
        // Create training data
        const int TRUE = 1;
        const int FALSE = -1;
        var training = new ListTrainingSet();
        training.Add(new TaggedVector<string>(("In1", FALSE), ("In2", FALSE)), new TaggedVector<string>(("Out", FALSE)));
        training.Add(new TaggedVector<string>(("In1", FALSE), ("In2", TRUE)) , new TaggedVector<string>(("Out", TRUE)));
        training.Add(new TaggedVector<string>(("In1", TRUE),  ("In2", FALSE)), new TaggedVector<string>(("Out", TRUE)));
        training.Add(new TaggedVector<string>(("In1", TRUE),  ("In2", TRUE)) , new TaggedVector<string>(("Out", FALSE)));

        // Training the neural network (also validates the results)
        var trainer = new GeneticEvolutionTrainer<FeedForwardNeuralNetworkGenome>(
            generations: 1000, 
            desiredAccuracy: 0.1,
            reproduction: new FeedForwardNeuralNetworkMeiosis(),
            fitness: new NeuralNetworkFitness<IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome>, FeedForwardNeuralNetworkGenome>(training),
            distribution: new PopulationSelection(elite: 2, crossover: 1, mutation: 4)
        );
        IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome> best;
        Assert.AreEqual(true, trainer.TryTrain(new XorGenerator().Generate(250), out best));  // Check that the training was a success

        // Verify that each element of the training set actually delivers the correct output
        foreach (var set in training) {
            var result = best.Evaluate(set.FeatureVector).RemapTags(x => "Out");
            for (var i = 0; i < result.Count; i++) {
                Assert.AreEqual(set.ResultVector[i], System.Math.Round(result[i]), $"{set.ResultVector} != {result} for input {set.FeatureVector}");
            }
        }
    }
}