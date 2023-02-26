namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Interface for neural networks
/// </summary>
public interface INeuralNetwork {
    /// <summary>
    /// Use the given feature vector to produce a result vector 
    /// </summary>
    /// <param name="features">feature vector</param>
    /// <returns>result vector</returns>
    public IVector Evaluate(IVector features);

    /// <summary>
    /// Use the given feature vector to produce a result vector. Return additional trace information for training or debugging.
    /// </summary>
    /// <param name="features">feature vector</param>
    /// <param name="neuronOutputs">the outputs from each neuron</param>
    /// <returns>results vector</returns>
    public IVector EvaluateWithTrace(IVector features, out double[][] neuronOutputs);
}

/// <summary>
/// Interface for neural networks that have enough features that they can be back-propagated over for training
/// </summary>
public interface IBackPropagableNeuralNetwork : INeuralNetwork {
    /// <summary>
    /// The activation function used by the neural network
    /// </summary>
    /// <value>activation function</value>
    public Activation.IActivationFunction ActivationFunction {get;}

    /// <summary>
    /// Number of layers in this network
    /// </summary>
    /// <value>layer count</value>
    public int LayerCount {get;}

    /// <summary>
    /// Get the count of neurons in the given layer
    /// </summary>
    /// <param name="n">layer index</param>
    /// <returns>number of neurons</returns>
    public int CountNeuronsInLayer(int n);

    /// <summary>
    /// Get the synapse weight between two neurons
    /// </summary>
    /// <param name="fromLayer">layer index containing the starting neuron</param>
    /// <param name="fromNeuron">starting neuron index in the layer</param>
    /// <param name="toLayer">layer index containing the ending neuron</param>
    /// <param name="toNeuron">ending neuron index in the layer</param>
    /// <returns>weight</returns>
    public double GetSynapseWeight(int fromLayer, int fromNeuron, int toLayer, int toNeuron);

    /// <summary>
    /// Set the synapse weight between two neurons
    /// </summary>
    /// <param name="fromLayer">layer index containing the starting neuron</param>
    /// <param name="fromNeuron">starting neuron index in the layer</param>
    /// <param name="toLayer">layer index containing the ending neuron</param>
    /// <param name="toNeuron">ending neuron index in the layer</param>
    /// <param name="weight">new weight valye</param>
    public void SetSynapseWeight(int fromLayer, int fromNeuron, int toLayer, int toNeuron, double weight);

    /// <summary>
    /// Utility method to randomize all the weights between neurons
    /// </summary>
    public void RandomizeSynapseWeights();
}