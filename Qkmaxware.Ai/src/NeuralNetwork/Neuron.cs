namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// 0 indexed sequential unique id representing a specific neuron in a neural network
/// </summary>
public struct NeuronId {
    private int value;

    public NeuronId(int val) {
        this.value = val;
    }

    public static explicit operator NeuronId(int val) {
        return new NeuronId(val);
    }
    public static implicit operator int(NeuronId id) {
        return id.value;
    }

    public override string ToString() {
        return $"neuron({value})";
    }
}

/// <summary>
/// A single neuron within a neural network
/// </summary>
public struct Neuron {
    /// <summary>
    /// 0 indexed sequential unique id representing this neuron
    /// </summary>
    /// <value>id</value>
    public NeuronId Id {get; init;}

    /// <summary>
    /// Neuron bias
    /// </summary>
    /// <value>bias</value>
    public double Bias {get; set;}

    /// <summary>
    /// Create a new neuron with the given id/index
    /// </summary>
    /// <param name="index">0 indexed sequential unique id representing this neuron</param>
    public Neuron(NeuronId index) {
        this.Id = index;
        this.Bias = default(double);
    }
}