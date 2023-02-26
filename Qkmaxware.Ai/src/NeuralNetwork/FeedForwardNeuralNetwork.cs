using Qkmaxware.Ai.NeuralNetwork.Activation;

namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Simple feed-forward neural network with a configurable number of hidden layers all of which are fully connected
/// /// </summary>
public class FeedForwardNeuralNetwork : INeuralNetwork, IBackPropagableNeuralNetwork {
    /// <summary>
    /// Number of neurons in this network
    /// </summary>
    /// <value>neuron count</value>
    public int NeuronCount => neurons.Length;
    private Neuron[] neurons;
    private int resultNeuronLayerOffset => this.neuronLayerOffsets[this.neuronLayerOffsets.Length - 1];
    private int[] neuronLayerOffsets;
    private Synapse[,] synapses;

    /// <summary>
    /// Number of layers in this network
    /// </summary>
    /// <value>layer count</value>
    public int LayerCount => layerSizes.Length;
    private int[] layerSizes;

    /// <summary>
    /// Get the count of neurons in the given layer
    /// </summary>
    /// <param name="n">layer index</param>
    /// <returns>number of neurons</returns>
    public int CountNeuronsInLayer(int n) => n >= 0 && n < layerSizes.Length ? layerSizes[n] : 0;

    /// <summary>
    /// Number of hidden layers in this network
    /// </summary>
    /// <value>layer count</value>
    public int HiddenLayerCount => LayerCount - 2;

    /// <summary>
    /// Size of input vectors compatible with this neural network
    /// </summary>
    /// <value>allowed size of input feature vectors</value>
    public int FeatureVectorSize => this.layerSizes[0];

    /// <summary>
    /// Size of output vectors created by this neural network
    /// </summary>
    /// <value>size of created output vectors</value>
    public int ResultVectorSize => this.layerSizes[this.layerSizes.Length - 1];

    public IActivationFunction ActivationFunction => activation;

    private Activation.IActivationFunction activation;

    public FeedForwardNeuralNetwork(Activation.IActivationFunction activationFunction, int inputSize, params int[] layerSizes) {
        // Create neurons
        this.neurons = new Neuron[layerSizes.Sum() + inputSize];
        
        // Create layers & compute offsets
        this.layerSizes = new int[1 + layerSizes.Length];
        this.layerSizes[0] = inputSize;
        this.neuronLayerOffsets = new int[this.layerSizes.Length];
        this.neuronLayerOffsets[0] = 0;
        var offset = inputSize;
        for (var i = 0; i < layerSizes.Length; i++) {
            this.layerSizes[1 + i] = layerSizes[i];
            this.neuronLayerOffsets[1 + i] = offset;
            offset += layerSizes[i]; 
        }

        // Create synapses
        synapses = new Synapse[this.NeuronCount, this.NeuronCount];

        // Store activation function
        this.activation = activationFunction;
    }

    private double getWeight(int neuronFrom, int neuronTo) {
        return this.synapses[neuronFrom, neuronTo].Weight;
    }

    /// <summary>
    /// Get the synapse weight between two neurons
    /// </summary>
    /// <param name="fromLayer">layer index containing the starting neuron</param>
    /// <param name="fromNeuron">starting neuron index in the layer</param>
    /// <param name="toLayer">layer index containing the ending neuron</param>
    /// <param name="toNeuron">ending neuron index in the layer</param>
    /// <returns>weight</returns>
    public double GetSynapseWeight(int fromLayer, int fromNeuron, int toLayer, int toNeuron) {
        return getWeight(
            this.neuronLayerOffsets[fromLayer] + fromNeuron,
            this.neuronLayerOffsets[toLayer] + toNeuron
        );
    }

    /// <summary>
    /// Set the synapse weight between two neurons
    /// </summary>
    /// <param name="fromLayer">layer index containing the starting neuron</param>
    /// <param name="fromNeuron">starting neuron index in the layer</param>
    /// <param name="toLayer">layer index containing the ending neuron</param>
    /// <param name="toNeuron">ending neuron index in the layer</param>
    /// <param name="weight">new weight valye</param>
    public void SetSynapseWeight(int fromLayer, int fromNeuron, int toLayer, int toNeuron, double weight) {
        var synapse = this.synapses[
            this.neuronLayerOffsets[fromLayer] + fromNeuron, 
            this.neuronLayerOffsets[toLayer] + toNeuron
        ];
        synapse.Weight = weight;
        this.synapses[
            this.neuronLayerOffsets[fromLayer] + fromNeuron, 
            this.neuronLayerOffsets[toLayer] + toNeuron
        ] = synapse;
    }

    private double getBias(int neuronId) {
        return this.neurons[neuronId].Bias;
    }

    public IVector EvaluateWithTrace(IVector features, out double[][] neuronOutputs) {
        // Safety check
        if (features.Count != this.FeatureVectorSize)
            throw new ArgumentException($"Feature vector is the wrong size. Expected {this.FeatureVectorSize} but got {features.Count}.");

        // Neuron inputs/outputs
        var outputs = new double[this.NeuronCount];

        // Set feature vector as output of input layer
        var inputLayerOffset = this.neuronLayerOffsets[0];
        for(var i = 0; i < this.FeatureVectorSize; i++) {
            outputs[inputLayerOffset + i] = activation.ValueAt(features[i] + this.neurons[i].Bias);
        }

        // Do feed-forward
        // Foreach layer (skipping input layer)
        for (var layerId = 1; layerId < LayerCount; layerId++) {
            var offset = this.neuronLayerOffsets[layerId];
            var prevLayerId = layerId - 1;
            var prevLayerOffset = this.neuronLayerOffsets[prevLayerId];
            var size = this.layerSizes[layerId];
            var prevLayerSize = this.layerSizes[prevLayerId];

            // Foreach neuron in layer
            for (var neuronId = 0; neuronId < size; neuronId++) {
                // Compute net sum of weighted inputs
                var sum = 0.0;
                var to = offset + neuronId;
                for (var prevLayerNeuronId = 0; prevLayerNeuronId < prevLayerSize; prevLayerNeuronId++) {
                    var from = prevLayerOffset + prevLayerNeuronId;
                    var weight = getWeight(from, to);
                    sum += weight * outputs[from];
                }

                // Apply activation function and save output
                outputs[to] = activation.ValueAt(sum + neurons[to].Bias);
            }
        }

        // Create results vector
        double[] outs = new double[this.ResultVectorSize];
        var resultOffset = this.resultNeuronLayerOffset;
        for (var i = 0; i < outs.Length; i++) {
            outs[i] = outputs[resultOffset + i];
        }

        // Create output matrix
        neuronOutputs = new double[this.LayerCount][];
        for (var layer = 0; layer < this.LayerCount; layer++) {
            var l = new double[this.CountNeuronsInLayer(layer)];
            for (var i = 0; i < l.Length; i++) {
                l[i] = outputs[this.neuronLayerOffsets[layer] + i];
            }
            neuronOutputs[layer] = l;
        }

        return new UntaggedVector(outs);
    }

    public IVector Evaluate(IVector features) {
        double[][] outputs;
        return EvaluateWithTrace(features, out outputs);
    }

    private static Random rng = new Random();
    public void RandomizeSynapseWeights() {
        for (var i = 0; i < this.synapses.GetLength(0); i++) {
            for (var j = 0; j < this.synapses.GetLength(1); j++) {
                this.synapses[i,j].Weight = rng.NextDouble();
            }
        }
    }
}