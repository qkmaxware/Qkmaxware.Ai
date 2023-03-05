namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Represents an ordered vector of values
/// </summary>
public interface IVector : IEnumerable<double> {
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

/// <summary>
/// Interface for a vector whose values are tagged with metadata
/// </summary>
/// <typeparam name="T">metadata tag</typeparam>
public interface ITaggedVector<T> : IVector {
    /// <summary>
    /// Get the tag name for the nth element in this vector
    /// </summary>
    /// <param name="n">element index</param>
    /// <returns>name of the nth element</returns>
    public T? GetTagOf(int n);
    /// <summary>
    /// Remap a vector from one set of tags to another
    /// </summary>
    /// <param name="tagMap">conversion function from one tag to another</param>
    /// <typeparam name="K">type of the new tag</typeparam>
    /// <returns>new vector with the same data but new tags</returns>
    public ITaggedVector<K> RemapTags<K>(Func<T?,K> tagMap);
}

/// <summary>
/// Represents an ordered vector of values whose values can be changes
/// </summary>
public interface IMutableVector : IVector {
    /// <summary>
    /// Get the nth element in this vector
    /// </summary>
    /// <value>value at the nth position</value>
    public new double this[int n] {get; set;}
}