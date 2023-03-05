using Qkmaxware.Ai.NeuralNetwork;

namespace  Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// Interface for neural networks that have enough features that they can be back-propagated over for training
/// </summary>
public interface IBackPropagableNeuralNetwork : INeuralNetwork {
    /// <summary>
    /// The activation function used by the neural network
    /// </summary>
    /// <value>activation function</value>
    public Qkmaxware.Ai.NeuralNetwork.Activation.IActivationFunction ActivationFunction {get;}

    /// <summary>
    /// Number of layers in this network
    /// </summary>
    /// <value>layer count</value>
    public int LayerCount {get;}

    /// <summary>
    /// Get the synapse weight between two neurons
    /// </summary>
    /// <param name="from">id of the neuron the synapse starts at</param>
    /// <param name="to">id of the neuron the synapse ends at</param>
    /// <returns>weight</returns>
    public double GetSynapseWeight(NeuronId from, NeuronId to);

    /// <summary>
    /// Set the synapse weight between two neurons
    /// </summary>
    /// <param name="from">id of the neuron the synapse starts at</param>
    /// <param name="to">id of the neuron the synapse ends at</param>
    /// <param name="weight">new weight valye</param>
    public void SetSynapseWeight(NeuronId from, NeuronId to, double weight);

    /// <summary>
    /// Get the ids of all neurons that act as input to the given neuron
    /// </summary>
    /// <param name="id">neuron id whose inputs we want to enumerate</param>
    /// <returns>enumeration of all neuron ids who are inputs to the given neuron id</returns>
    public IEnumerable<NeuronId> GetInputsTo(NeuronId id);

    /// <summary>
    /// Get the ids of all neurons that this neuron passes its output to 
    /// </summary>
    /// <param name="id">neuron id whose outputs we want to enumerate</param>
    /// <returns>enumeration of all neuron ids who receive input from the given neuron id</returns>
    public IEnumerable<NeuronId> GetOutputsFrom(NeuronId id);

    /// <summary>
    /// Utility method to randomize all the weights between neurons
    /// </summary>
    public void RandomizeSynapseWeights();
}