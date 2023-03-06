using Qkmaxware.Ai.NeuralNetwork.Training;

namespace Qkmaxware.Ai.NeuralNetwork.Training {

/// <summary>
/// Fitness trainer for FeedForwardNeuralNetwork genomes
/// </summary>
/// <typeparam name="TGenome"></typeparam>
public class NeuralNetworkFitness<TNetwork, TGenome> : IFitnessTest<TGenome> where TGenome:IDecodableGenome<TNetwork> where TNetwork:INeuralNetwork {

    private Training.ITrainingSet data;

    public NeuralNetworkFitness(Training.ITrainingSet data) {
        this.data = data;
    }

    public double TestError(TGenome genome) {
        var nn = genome.Decode();

        double error = 0;
        foreach (var pair in data) {
            var results = nn.Evaluate(pair.FeatureVector);
            double a = 0;
            for (int j = 0; j < results.Count; j++) {
                a += Math.Abs(results[j] - pair.ResultVector[j]);   // difference between computed and actual values is the error of this specific value
            }
            error += a / results.Count;
        }
        error /= data.Size;                       // average error over all validation data

        return error; // Higher value is greater error
    }
}

}

namespace Qkmaxware.Ai.NeuralNetwork {

/// <summary>
/// Neural network genome encoded from feed forward neural networks
/// </summary>
public class FeedForwardNeuralNetworkGenome : IGenome, IDecodableGenome<IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome>> {

    public Activation.IActivationFunction ActivationFunction {get; init;}
    public int LayerCount {get; init;}
    private int[] NeuronCounts {get; init;}

    private double[] weights;

    public FeedForwardNeuralNetworkGenome(FeedForwardNeuralNetwork network) {
        this.ActivationFunction = network.ActivationFunction;
        this.LayerCount = network.LayerCount;
        this.NeuronCounts = new int[LayerCount];
        for (var i = 0; i < this.LayerCount; i++)
            this.NeuronCounts[i] = network.CountNeuronsInLayer(i);

        this.weights = network.EnumerateSynapses().Select(x => x.Weight).ToArray();
    }

    public FeedForwardNeuralNetworkGenome(FeedForwardNeuralNetworkGenome genome) {
        this.ActivationFunction = genome.ActivationFunction;
        this.LayerCount = genome.LayerCount;
        this.NeuronCounts = new int[LayerCount];
        for (var i = 0; i < this.LayerCount; i++)
            this.NeuronCounts[i] = genome.NeuronCounts[i];

        this.weights = new double[genome.weights.Length];
        for (var i = 0; i < this.weights.Length; i++)
            this.weights[i] = genome.weights[i];
    }

    /// <summary>
    /// Clone the existing genome into a new genome
    /// </summary>
    /// <returns>new genome identical to the current one</returns>
    public IGenome Clone() {
        return new FeedForwardNeuralNetworkGenome(this);
    }

    public IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome> Decode() {
        var nn = new FeedForwardNeuralNetwork(this.ActivationFunction, this.NeuronCounts[0], this.NeuronCounts.Skip(1).ToArray());
        nn.BatchSetSynapseWeights(this.weights);
        return nn;
    }

    public int CountWeights => this.weights.Length;

    public double GetWeight(int n) {
        if (n >= 0 && n < this.weights.Length)
            return this.weights[n];
        return 0;
    }

    public void SetWeight(int n, double weight) {
        if (n >= 0 && n < this.weights.Length)
            this.weights[n] = weight;
    }
}

/// <summary>
/// Crossover and mutation rules for FeedForwardNeuralNetworkGenome objects
/// </summary>
public class FeedForwardNeuralNetworkMeiosis : Training.IReproductiveRules<FeedForwardNeuralNetworkGenome> {
    public (FeedForwardNeuralNetworkGenome, FeedForwardNeuralNetworkGenome) Crossover(FeedForwardNeuralNetworkGenome a, FeedForwardNeuralNetworkGenome b) {
        if (a.ActivationFunction != b.ActivationFunction)
            throw new ArgumentException("Genomes use different activation functions and cannot be crossed-over");
        if (a.LayerCount != b.LayerCount)
            throw new ArgumentException("Genomes have different layer counts and cannot be crossed over");
        if (a.CountWeights != b.CountWeights)
            throw new ArgumentException("Genomes have a different number of weights and cannot be crossed over");

        var childA = new FeedForwardNeuralNetworkGenome(a);
        var childB = new FeedForwardNeuralNetworkGenome(a);

        for (var i = 0; i < a.CountWeights; i++) {
            int mask = rng.Next(2);
            if (mask == 0) {
                childA.SetWeight(i, a.GetWeight(i));
                childB.SetWeight(i, b.GetWeight(i));
            } else {
                childA.SetWeight(i, b.GetWeight(i));
                childB.SetWeight(i, a.GetWeight(i));
            }
        }

        return (childA, childB);
    }

    public double DifferenceBetween(FeedForwardNeuralNetworkGenome a, FeedForwardNeuralNetworkGenome b) {
        var diff = 0.0;
        var count = Math.Max(a.CountWeights, b.CountWeights);

        for(int g = 0; g < count; g++){
            var da = g < a.CountWeights ? a.GetWeight(g) : 0;
            var db = g < b.CountWeights ? b.GetWeight(g) : 0;
            diff += Math.Abs(da - db);
        }
        return diff;
    }

    private static Random rng = new Random();
    public FeedForwardNeuralNetworkGenome Mutate(FeedForwardNeuralNetworkGenome genome) {
        var next = new FeedForwardNeuralNetworkGenome(genome);
        for (var i = 0; i < next.CountWeights; i++) {
            next.SetWeight(i, genome.GetWeight(i) + (rng.NextDouble() - 0.5));
        }
        return next;
    }
}

}