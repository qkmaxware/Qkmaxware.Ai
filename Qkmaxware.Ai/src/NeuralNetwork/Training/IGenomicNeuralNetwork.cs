using Qkmaxware.Ai.NeuralNetwork;

namespace  Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// Base interface for genetic training genomes
/// </summary>
public interface IGenome {}

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
    public double Test(TGenome genome);
}

/// <summary>
/// Interface for neural networks that have enough features that they can be used in genetic training
/// </summary>
public interface IGenomicNeuralNetwork<TGenome> : INeuralNetwork where TGenome:IGenome {
    public TGenome EncodeToGenome();
}