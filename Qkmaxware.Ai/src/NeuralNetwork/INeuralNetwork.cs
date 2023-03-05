namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Interface for neural networks
/// </summary>
public interface INeuralNetwork {
    /// <summary>
    /// Number of neurons in this network
    /// </summary>
    /// <value>neuron count</value>
    public int NeuronCount {get;}

    /// <summary>
    /// Use the given feature vector to produce a result vector 
    /// </summary>
    /// <param name="features">feature vector</param>
    /// <returns>result vector</returns>
    public ITaggedVector<NeuronId> Evaluate(IVector features);
}