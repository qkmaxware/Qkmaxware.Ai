using Qkmaxware.Ai.NeuralNetwork.Activation;
using Qkmaxware.Ai.NeuralNetwork.Training;

namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Simple feed-forward neural network with a configurable number of hidden layers all of which are fully connected
/// /// </summary>
public class FeedForwardNeuralNetwork : INeuralNetwork, IBackPropagableNeuralNetwork, IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome> {
    /// <summary>
    /// Number of neurons in this network
    /// </summary>
    /// <value>neuron count</value>
    public int NeuronCount => neurons.Length;
    private Neuron[] neurons;
    private int[] neuron_layers;
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
        this.neuron_layers = new int[this.NeuronCount];
        for (var i = 0; i < this.NeuronCount; i++) {
            this.neurons[i] = new Neuron((NeuronId)i);
        }
        
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
        for (var l = 0; l < this.LayerCount; l++) {
            for (var n = 0; n < this.CountNeuronsInLayer(l); n++) {
                var o = this.neuronLayerOffsets[l];
                this.neuron_layers[o + n] = l;
            }
        }

        // Create synapses
        synapses = new Synapse[this.NeuronCount, this.NeuronCount];

        // Store activation function
        this.activation = activationFunction;
    }

    private double getWeight(int neuronFrom, int neuronTo) {
        return this.synapses[neuronFrom, neuronTo].Weight;
    }

    private double getBias(int neuronId) {
        return this.neurons[neuronId].Bias;
    }

    public ITaggedVector<NeuronId> Evaluate(IVector features) {
        // Safety check
        if (features.Count != this.FeatureVectorSize)
            throw new ArgumentException($"Feature vector is the wrong size. Expected {this.FeatureVectorSize} but got {features.Count}.");

        // Neuron inputs/outputs
        var outputs = new double[this.NeuronCount];

        // Set feature vector as output of input layer
        var inputLayerOffset = this.neuronLayerOffsets[0];
        for(var i = 0; i < this.FeatureVectorSize; i++) {
            outputs[inputLayerOffset + i] = features[i] + getBias(i);
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
                var net = 0.0;
                var to = offset + neuronId;
                for (var prevLayerNeuronId = 0; prevLayerNeuronId < prevLayerSize; prevLayerNeuronId++) {
                    var from = prevLayerOffset + prevLayerNeuronId;
                    var weight = getWeight(from, to);
                    net += outputs[from] * weight;
                }

                // Apply activation function and save output
                outputs[to] = activation.ValueAt(net + getBias(to));
            }
        }

        // Create results vector
        TaggedVector<NeuronId> outs = new TaggedVector<NeuronId>(this.ResultVectorSize);
        var resultOffset = this.resultNeuronLayerOffset;
        for (var i = 0; i < this.ResultVectorSize; i++) {
            outs.SetTagOf(i, this.neurons[resultOffset + i].Id);
            outs[i] = outputs[resultOffset + i];
        }
        return outs;
    }

    /// <summary>
    /// Get the synapse weight between two neurons
    /// </summary>
    /// <param name="from">id of the neuron the synapse starts at</param>
    /// <param name="to">id of the neuron the synapse ends at</param>
    /// <returns>weight</returns>
    public double GetSynapseWeight(NeuronId from, NeuronId to) {
        return getWeight(
            from,
            to
        );
    }

    /// <summary>
    /// Set the synapse weight between two neurons
    /// </summary>
    /// <param name="from">id of the neuron the synapse starts at</param>
    /// <param name="to">id of the neuron the synapse ends at</param>
    /// <param name="weight">new weight value</param>
    public void SetSynapseWeight(NeuronId from, NeuronId to, double weight) {
        var synapse = this.synapses[
            from, 
            to
        ];
        synapse.Weight = weight;
        this.synapses[
            from, 
            to
        ] = synapse;
    }

    // Get the layer of a particular neuron
    protected int LayerOf(NeuronId neuron) {
        return this.neuron_layers[neuron];
    }

    /// <summary>
    /// Get the ids of all neurons that act as input to the given neuron
    /// </summary>
    /// <param name="id">neuron id whose inputs we want to enumerate</param>
    /// <returns>enumeration of all neuron ids who are inputs to the given neuron id</returns>
    public IEnumerable<NeuronId> GetInputsTo(NeuronId id) {
        // All neurons in the previous layer
        var layer = LayerOf(id) - 1;
        if (layer < 0)
            yield break;
        
        var offset = this.neuronLayerOffsets[layer];
        for (var n = 0; n < this.CountNeuronsInLayer(layer); n++) {
            yield return this.neurons[offset + n].Id;
        }
    }

    /// <summary>
    /// Set the bias to a particular neuron
    /// </summary>
    /// <param name="neuron">neuron to change the bias for</param>
    /// <param name="bias">new bias value</param>
    public void SetNeuronBias(NeuronId neuron, double bias) {
        var n = this.neurons[neuron];
        n.Bias = bias;
        this.neurons[neuron] = n;
    }

    /// <summary>
    /// Set the bias to all non-input neurons to the same value
    /// </summary>
    /// <param name="bias">new bias value</param>
    public void SetGlobalBias(double bias) {
        for (var n = this.FeatureVectorSize; n < this.NeuronCount; n++) {
            SetNeuronBias((NeuronId)n, bias);
        }
    }

    /// <summary>
    /// Get the ids of all neurons that this neuron passes its output to 
    /// </summary>
    /// <param name="id">neuron id whose outputs we want to enumerate</param>
    /// <returns>enumeration of all neuron ids who receive input from the given neuron id</returns>
    public IEnumerable<NeuronId> GetOutputsFrom(NeuronId id) {
        // All neurons in the next layer
        var layer = LayerOf(id) + 1;
        if (layer >= this.LayerCount)
            yield break;
        
        var offset = this.neuronLayerOffsets[layer];
        for (var n = 0; n < this.CountNeuronsInLayer(layer); n++) {
            yield return this.neurons[offset + n].Id;
        }
    }

    private static Random rng = new Random();
    /// <summary>
    /// Utility method to randomize all the weights between neurons
    /// </summary>
    public void RandomizeSynapseWeights() {
        for (var i = 0; i < this.synapses.GetLength(0); i++) {
            for (var j = 0; j < this.synapses.GetLength(1); j++) {
                this.synapses[i,j].Weight = rng.NextDouble();
            }
        }
    }

    /// <summary>
    /// Encode this network as a genome for genetic training
    /// </summary>
    /// <returns>genome</returns>
    public FeedForwardNeuralNetworkGenome EncodeToGenome() {
        return new FeedForwardNeuralNetworkGenome(this);
    }
}