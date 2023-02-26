namespace Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// Interface for neural network trainers
/// </summary>
/// <typeparam name="TNetwork">Type of neural network that can be trained by this trainer</typeparam>
public interface ITrainer<TNetwork> where TNetwork:INeuralNetwork {
    /// <summary>
    /// Try to train the given neural network
    /// </summary>
    /// <param name="nn">network to train</param>
    /// <returns>true if network has been trained successfully</returns>
    public bool TryTrain(TNetwork nn);
}