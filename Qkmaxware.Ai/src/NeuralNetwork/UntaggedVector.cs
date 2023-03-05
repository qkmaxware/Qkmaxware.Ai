using System.Collections;
using System.Text;

namespace Qkmaxware.Ai.NeuralNetwork;

/// <summary>
/// Vector with tagged names for each element
/// </summary>
public class UntaggedVector : IMutableVector {
    private double[] values;

    /// <summary>
    /// Create an un-tagged vector of the given size
    /// </summary>
    /// <param name="size">size of the vector</param>
    public UntaggedVector(int size) {
        this.values = new double[size];
    }

    /// <summary>
    /// Create an un-tagged vector with the given values
    /// </summary>
    /// <param name="values">vector values</param>
    public UntaggedVector(double[] values) {
        this.values = values;
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
            sb.Append(this[i]);
        }
        sb.Append(']');
        return sb.ToString();
    }

    public IEnumerator<double> GetEnumerator() {
        return (IEnumerator<double>)this.values.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator() {
        return this.values.GetEnumerator();
    }
}