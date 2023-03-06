using Qkmaxware.Ai.NeuralNetwork;

namespace  Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// Base interface for genetic training genomes
/// </summary>
public interface IGenome {
    /// <summary>
    /// Clone the existing genome into a new genome
    /// </summary>
    /// <returns>new genome identical to the current one</returns>
    IGenome Clone();
}

/// <summary>
/// Interface for a system that controls genome creation during reproduction
/// </summary>
/// <typeparam name="TGenome">type of genome</typeparam>
public interface IReproductiveRules<TGenome> where TGenome:IGenome {
    public TGenome Mutate(TGenome genome);
    public (TGenome, TGenome) Crossover(TGenome a, TGenome b);
    public double DifferenceBetween(TGenome a, TGenome b);
}

/// <summary>
/// Base interface for genomes that can be decoded back into a neural network
/// </summary>
/// <typeparam name="TNN">decoded object type</typeparam>
public interface IDecodableGenome<TNN> : IGenome {
    public TNN Decode();
}

/// <summary>
/// A fitness test for a genome to undergo
/// </summary>
public interface IFitnessTest<TGenome> where TGenome:IGenome {
    /// <summary>
    /// Compute the error of this genome. A higher value indicates a greater error
    /// </summary>
    /// <param name="genome">genome to test</param>
    /// <returns>error value</returns>
    public double TestError(TGenome genome);
}

/// <summary>
/// Interface for neural networks that have enough features that they can be used in genetic training
/// </summary>
public interface IGenomicNeuralNetwork<TGenome> : INeuralNetwork where TGenome:IGenome {
    /// <summary>
    /// Encode this network to a genome for genetic training
    /// </summary>
    /// <returns>network encoded as a genome</returns>
    public TGenome EncodeToGenome();
}