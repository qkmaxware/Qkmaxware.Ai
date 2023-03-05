using Qkmaxware.Ai.NeuralNetwork.Activation;

namespace Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// Stochastic back-propagation trainer for feed-forward neural networks
/// </summary>
public class StochasticBackPropagationTrainer : ITrainer<IBackPropagableNeuralNetwork> {
    private int trainingEpochs;
    private int iterationsPerEpoch;
    private ITrainingSet trainingData;
    private ITrainingSet validationData;
    private double learningRate;
    private double desiredAccuracy;

    public StochasticBackPropagationTrainer(int epochs, int iterationsPerEpoch, double learningRate, double desiredAccuracy, ITrainingSet trainingData, ITrainingSet validationData) {
        this.trainingEpochs = Math.Max(1, epochs);
        this.iterationsPerEpoch = Math.Max(1, iterationsPerEpoch);
        this.learningRate = learningRate;
        this.trainingData = trainingData;
        this.validationData = validationData;
        this.desiredAccuracy = desiredAccuracy;
    }

    public bool TryTrain(IBackPropagableNeuralNetwork nn, out IBackPropagableNeuralNetwork trained) {
        // The trained network is the same object as the original network
        trained = nn;

        // Try training 'epoch' times
        for (var epoch = 0; epoch < this.trainingEpochs; epoch++) {
            // Start with a randomly weighted network
            nn.RandomizeSynapseWeights();

            // Train network through back-propagation
            for (var iteration = 0; iteration < this.iterationsPerEpoch; iteration++) {
                this.BackPropagate(nn, this.trainingData.Random(), this.learningRate);
            }

            // Test network for accuracy
            double accuracy = 0;
            foreach (var pair in this.validationData) {
                var results = nn.Evaluate(pair.FeatureVector);
                double a = 0;
                for (int j = 0; j < results.Count; j++) {
                    a += Math.Abs(results[j] - pair.ResultVector[j]);   // difference between computed and actual values is the accuracy of this specific value
                }
                accuracy += (a / results.Count);                        // average accuracy of each value
            }
            accuracy /= this.validationData.Size;                       // average accuracy over all validation data

            // If accurate enough return true;
            if (accuracy < this.desiredAccuracy) {
                return true;    // Successfully trained the network
            } else {
                continue;       // Didn't do well enough this time, try again next epoch
            }
        }

        // After all epochs, none of the networks were good enough
        return false;
    }

    struct NeuronData {
        public double output;
        public double delta;
        public double error;
    }

    protected void BackPropagate(IBackPropagableNeuralNetwork nn, ITrainingPair io, double eta) {
        var results = nn.Evaluate(io.FeatureVector);
        var expected = io.ResultVector;
        var activation = nn.ActivationFunction;

        // Calculate errors for output neurons
        var neuron_data = new NeuronData[nn.NeuronCount];
        var next_layer = new Queue<NeuronId>();
        for (var n = 0; n < results.Count; n++) {
            var neuron = results.GetTagOf(n);
            var output = results[n];
            var diff = output - expected[n];
            neuron_data[neuron] = new NeuronData {
                output = output,
                delta = diff,
                error = diff * activation.DerivativeAt(output)
            };
            foreach (var l in nn.GetInputsTo(neuron)) {
                if (!next_layer.Contains(l)) {
                    next_layer.Enqueue(l);
                }
            }
        }

        // Calculate errors for all other neurons
        while (next_layer.Count > 0) {
            var from = next_layer.Dequeue();
            var outputs = nn.GetOutputsFrom(from);

            var error = 0.0;
            foreach (var to in outputs) {
                error += nn.GetSynapseWeight(from, to) * neuron_data[to].error;
            }
            neuron_data[from].delta = error;
            neuron_data[from].error = error * nn.ActivationFunction.DerivativeAt(neuron_data[from].output);
        
            foreach (var n in nn.GetInputsTo(from)) {
                if (!next_layer.Contains(n))
                    next_layer.Enqueue(n);
            }
        }

        // Update weights
        for (var n = 0; n < nn.NeuronCount; n++) {
            var error = neuron_data[n].error;
            foreach (var inNeuron in nn.GetInputsTo((NeuronId)n)) {
                var input = neuron_data[inNeuron].output;
                var old_weight = nn.GetSynapseWeight(inNeuron, (NeuronId)n);
                var new_weight = old_weight - eta * error * input;
                nn.SetSynapseWeight(inNeuron, (NeuronId)n, new_weight);
            }
        }
    }

    // // https://github.com/qkmaxware/Rubik/blob/master/src/plus/machinelearning/MatrixNetwork.java#L106
    // // https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    // protected void BackPropagate(IBackPropagableNeuralNetwork nn, ITrainingPair io, double eta) {
    //     double[][] neuronOutputs;
    //     var output = nn.EvaluateWithTrace(io.FeatureVector, out neuronOutputs);
    //     var actual = io.ResultVector;

    //     // Create error matrix
    //     double[][] errors = new double[nn.LayerCount][];
    //     for (var layer = 0; layer < errors.Length; layer++) {
    //         errors[layer] = new double[nn.CountNeuronsInLayer(layer)];
    //     }

    //     // Calculate the error for each output neuron
    //     var last = errors.Length - 1;
    //     for (var neuron = 0; neuron < errors[last].Length; neuron++) {
    //         var error = outputError(output[neuron], actual[neuron], nn.ActivationFunction);
    //         errors[last][neuron] = error;
    //     }

    //     // Calculate the error in each hidden layer (reverse order)
    //     for (var invLayerIdx = 0; invLayerIdx < errors.Length - 1; invLayerIdx++) {
    //         var layerIdx = errors.Length - 2 - invLayerIdx;
    //         var nextLayerIdx = layerIdx + 1;
    //         double[] errorInNextLayer = errors[nextLayerIdx];

    //         // For each neuron in the current layer
    //         for (var neuronIdx = 0; neuronIdx < errors[layerIdx].Length; neuronIdx++) {
    //             var error = 0.0;
    //             for (var nextLayerNeuronIdx = 0; nextLayerNeuronIdx < errors[nextLayerIdx].Length; nextLayerNeuronIdx++) {
    //                 error += nn.GetSynapseWeight(layerIdx, neuronIdx, nextLayerIdx, nextLayerNeuronIdx) * errorInNextLayer[nextLayerNeuronIdx];
    //             }
    //             error *= nn.ActivationFunction.DerivativeAt(neuronOutputs[layerIdx][neuronIdx]);
    //             errors[layerIdx][neuronIdx] = error;
    //         }
    //     }

    //     // Update weights (ignore input layer)
    //     for (var layerIdx = 1; layerIdx < errors.Length; layerIdx++) {
    //         var prevLayerIdx = layerIdx - 1;

    //         // Foreach neuron in layer
    //         for (var neuronIdx = 0; neuronIdx < errors[layerIdx].Length; neuronIdx++) {
    //             var error = errors[layerIdx][neuronIdx];

    //             for (var prevNeuronIdx = 0; prevNeuronIdx < errors[prevLayerIdx].Length; prevNeuronIdx++) {
    //                 var input = neuronOutputs[prevLayerIdx][prevNeuronIdx];
    //                 var weight = nn.GetSynapseWeight(prevLayerIdx, prevNeuronIdx, layerIdx, neuronIdx) - eta * error * input;
    //                 nn.SetSynapseWeight(prevLayerIdx, prevNeuronIdx, layerIdx, neuronIdx, weight);
    //             }
    //         }
    //     }
    // }

    // private double outputError(double output, double expected, IActivationFunction fn) {
    //     return (output - expected) * fn.DerivativeAt(output);
    // }
}