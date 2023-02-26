namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// A synapse between neurons within a neural network
/// </summary>
public struct Synapse {
    /// <summary>
    /// Weight of the synapse as values are passed between neurons
    /// </summary>
    /// <value>weight</value>
    public double Weight {get; set;} = 1;

    public Synapse() {
        this.Weight = 1;
    }
}