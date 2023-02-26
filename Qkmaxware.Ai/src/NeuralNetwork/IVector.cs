namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Represents a vector of values
/// </summary>
public interface IVector {
    /// <summary>
    /// The number of elements in the vector
    /// </summary>
    /// <value>size</value>
    public int Count {get;}

    /// <summary>
    /// Get the nth element in this vector
    /// </summary>
    /// <value>value at the nth position</value>
    public double this[int n] {get;}
}