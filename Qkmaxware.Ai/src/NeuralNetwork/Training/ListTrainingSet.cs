using System.Collections;

namespace Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// A minimal implementation of a pair of vectors for use as training data
/// </summary>
public record class SimpleTrainingPair : ITrainingPair {
    public SimpleTrainingPair(IVector FeatureVector, IVector ResultVector) {
        this.FeatureVector = FeatureVector;
        this.ResultVector = ResultVector;
    }

    public IVector FeatureVector {get; init;}

    public IVector ResultVector {get; init;}
}

/// <summary>
/// A training set backed by a List which can be appended to. Allows for the creation of training data via code.
/// </summary>
public class ListTrainingSet : ITrainingSet {
    private List<ITrainingPair> values = new List<ITrainingPair>();

    public int Size => values.Count;

    public void Add(IVector feature, IVector result) {
        this.Add(new SimpleTrainingPair(feature, result));
    }

    public void Add(ITrainingPair pair) {
        this.values.Add(pair);
    }

    public void AddRange(IEnumerable<ITrainingPair> pairs) {
        this.values.AddRange(pairs);
    }

    private static Random rng = new Random();

    public ITrainingPair Random() {
        return this.values[rng.Next(this.values.Count)];
    }

    public IEnumerator<ITrainingPair> GetEnumerator() {
        return values.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator() {
        return values.GetEnumerator();
    }
}