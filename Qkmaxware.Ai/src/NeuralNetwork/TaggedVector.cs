using System.Text;

namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Vector with tagged names for each element
/// </summary>
public class TaggedVector : IVector {
    private double[] values;
    private string[] tags;

    /// <summary>
    /// Create a new tagged vector with the given element names
    /// </summary>
    /// <param name="tags">list of element names</param>
    public TaggedVector(params string[] tags) {
        this.values = new double[tags.Length];
        this.tags = tags;
    }

    /// <summary>
    /// Create a new tagged vector from a list of elements and assigning tag names to those elements
    /// </summary>
    /// <param name="values">list of elements in the vector</param>
    /// <param name="tags">tags to assign to each element</param>
    public TaggedVector(double[] values, params string[] tags) {
        this.values = new double[values.Length];
        this.tags = new string[values.Length];

        for (var i = 0 ; i < this.values.Length; i++) {
            this.values[i] = values[i];
            this.tags[i] = i >= 0 && i < tags.Length ? tags[i] : string.Empty;
        }
    }

    /// <summary>
    /// Create a new tagged vector by copying the items from another vector but assigning new tag names to those items
    /// </summary>
    /// <param name="vector">vector whose elements are to be copied</param>
    /// <param name="tags">tags to assign to each element</param>
    public TaggedVector(IVector vector, params string[] tags) {
        this.values = new double[vector.Count];
        this.tags = new string[vector.Count];

        for (var i = 0 ; i < this.values.Length; i++) {
            this.values[i] = vector[i];
            this.tags[i] = i >= 0 && i < tags.Length ? tags[i] : string.Empty;
        }
    }

    /// <summary>
    /// Get the nth element in this vector
    /// </summary>
    /// <value>value at the nth position</value>
    public double this[int n] => values[n];

    /// <summary>
    /// Get the tag name for the nth element in this vector
    /// </summary>
    /// <param name="n">element index</param>
    /// <returns>name of the nth element</returns>
    public string NameOf(int n) => tags[n];

    /// <summary>
    /// The number of elements in the vector
    /// </summary>
    /// <value>size</value>
    public int Count => values.Length;

    public override string ToString() {
        StringBuilder sb = new StringBuilder();
        sb.Append('[');
        for (var i = 0; i < this.Count; i++) {
            if (i != 0)
                sb.Append(", ");
            sb.Append(this.NameOf(i)); sb.Append(": "); sb.Append(this[i]);
        }
        sb.Append(']');
        return sb.ToString();
    }
}