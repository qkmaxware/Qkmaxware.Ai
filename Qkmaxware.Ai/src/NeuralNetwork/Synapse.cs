namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// A synapse between neurons within a neural network
/// </summary>
public struct Synapse {
    /// <summary>
    /// The id of the neuron this Synapse starts at
    /// </summary>
    /// <value>neuron id</value>
    public NeuronId From {get; private set;}
    /// <summary>
    /// The id of the neuron this Synapse ends at
    /// </summary>
    /// <value>neuron id</value>
    public NeuronId To {get; private set;}
    /// <summary>
    /// Weight of the synapse as values are passed between neurons
    /// </summary>
    /// <value>weight</value>
    public double Weight {get; set;} = 1;

    public Synapse(NeuronId from, NeuronId to) {
        this.From = from;
        this.To = to;
        this.Weight = 1;
    }
}