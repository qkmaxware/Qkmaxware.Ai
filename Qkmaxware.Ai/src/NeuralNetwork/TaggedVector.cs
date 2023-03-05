using System.Collections;
using System.Text;

namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Vector with tagged names for each element
/// </summary>
public class TaggedVector<T> : ITaggedVector<T>, IMutableVector {
    private double[] values;
    private T?[] tags;

    /// <summary>
    /// Create a new tagged vector with the given element names
    /// </summary>
    /// <param name="tags">list of element names</param>
    public TaggedVector(params T[] tags) {
        this.values = new double[tags.Length];
        this.tags = tags;
    }

    /// <summary>
    /// Create a new tagged vector of the given size, tags are set to their default value
    /// </summary>
    /// <param name="size">number of elements</param>
    public TaggedVector(int size) {
        this.values = new double[size];
        this.tags = new T[size];
    }

    /// <summary>
    /// Create a new tagged vector from a list of elements and assigning tag names to those elements
    /// </summary>
    /// <param name="values">list of elements in the vector</param>
    /// <param name="tags">tags to assign to each element</param>
    public TaggedVector(double[] values, params T[] tags) {
        this.values = new double[values.Length];
        this.tags = new T[values.Length];

        for (var i = 0 ; i < this.values.Length; i++) {
            this.values[i] = values[i];
            this.tags[i] = i >= 0 && i < tags.Length ? tags[i] : default(T);
        }
    }

    /// <summary>
    /// Create a new tagged vector from a list of elements and their name
    /// </summary>
    /// <param name="values">list of tag, value tuples</param>
    public TaggedVector(params (T, double)[] values) {
        this.values = new double[values.Length];
        this.tags = new T[values.Length];

        for (var i = 0 ; i < this.values.Length; i++) {
            this.values[i] = values[i].Item2;
            this.tags[i] = values[i].Item1;
        }
    }

    /// <summary>
    /// Create a new tagged vector by copying the items from another vector but assigning new tag names to those items
    /// </summary>
    /// <param name="vector">vector whose elements are to be copied</param>
    /// <param name="tags">tags to assign to each element</param>
    public TaggedVector(IVector vector, params T[] tags) {
        this.values = new double[vector.Count];
        this.tags = new T[vector.Count];

        for (var i = 0 ; i < this.values.Length; i++) {
            this.values[i] = vector[i];
            this.tags[i] = i >= 0 && i < tags.Length ? tags[i] : default(T);
        }
    }

    /// <summary>
    /// Get the nth element in this vector
    /// </summary>
    /// <value>value at the nth position</value>
    public double this[int n] {
        get => values[n];
        set => values[n] = value;
    }

    /// <summary>
    /// Get the tag name for the nth element in this vector
    /// </summary>
    /// <param name="n">element index</param>
    /// <returns>name of the nth element</returns>
    public T? GetTagOf(int n) => tags[n];
    /// <summary>
    /// Set the tag of the nth element in this vector
    /// </summary>
    /// <param name="n">element index</param>
    /// <param name="tag">new tag</param>
    public void SetTagOf(int n, T? tag) {
        if (n >= 0 && n < this.tags.Length)
            this.tags[n] = tag;
    }

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
            sb.Append(this.GetTagOf(i)?.ToString() ?? string.Empty); sb.Append(": "); sb.Append(this[i]);
        }
        sb.Append(']');
        return sb.ToString();
    }

    /// <summary>
    /// Remap a vector from one set of tags to another
    /// </summary>
    /// <param name="tagMap">conversion function from one tag to another</param>
    /// <typeparam name="K">type of the new tag</typeparam>
    /// <returns>new vector with the same data but new tags</returns>
    ITaggedVector<K> ITaggedVector<T>.RemapTags<K>(Func<T?,K> tagMap) {
        return new TaggedVector<K>(
            this.values,
            this.tags.Select(tag => tagMap(tag)).ToArray()
        );
    }

    public IEnumerator<double> GetEnumerator() {
        return (IEnumerator<double>)this.values.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator() {
        return this.values.GetEnumerator();
    }
}