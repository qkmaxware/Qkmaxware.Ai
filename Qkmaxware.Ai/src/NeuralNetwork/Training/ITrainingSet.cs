namespace Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// A vector pair for training data
/// </summary>
public interface ITrainingPair {
    /// <summary>
    /// Feature vector
    /// </summary>
    /// <value>vector</value>
    public IVector FeatureVector {get;}
    /// <summary>
    /// Correct result vector when the feature vector is fed to a neural network
    /// </summary>
    /// <value>vector</value>
    public IVector ResultVector {get;}
}

/// <summary>
/// Interface for neural network training data
/// </summary>
public interface ITrainingSet : IEnumerable<ITrainingPair> {
    /// <summary>
    /// The number of elements in the set
    /// </summary>
    /// <value>size</value>
    public int Size {get;}

    /// <summary>
    /// Get a random element from the training data
    /// </summary>
    /// <returns>randomly chosen element</returns>
    public ITrainingPair Random();
}