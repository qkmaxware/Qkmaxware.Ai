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

    public double Test(TGenome genome) {
        var nn = genome.Decode();

        double error = 0;
        foreach (var pair in data) {
            var results = nn.Evaluate(pair.FeatureVector);
            double a = 0;
            for (int j = 0; j < results.Count; j++) {
                a += Math.Abs(results[j] - pair.ResultVector[j]);   // difference between computed and actual values is the error of this specific value
            }
        }
        error /= data.Size;                       // average error over all validation data

        return -error; // by inverting it we guarantee that "stronger" networks are towards 0 and "worse" networks are < than 0
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

        // TODO weights
    }

    public double GetGene(int i) {
        if (weights == null)
            return 0;
        if (i >= 0 && i < weights.Length)
            return weights[i];
        return 0;
    }

    public IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome> Decode() {
        var nn = new FeedForwardNeuralNetwork(this.ActivationFunction, this.NeuronCounts[0], this.NeuronCounts.Skip(1).ToArray());
        
        // TODO weights
        
        return nn;
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

        throw new NotImplementedException();
    }

    public double DifferenceBetween(FeedForwardNeuralNetworkGenome a, FeedForwardNeuralNetworkGenome b) {
        throw new NotImplementedException();
    }

    public FeedForwardNeuralNetworkGenome Mutate(FeedForwardNeuralNetworkGenome genome) {
        throw new NotImplementedException();
    }
}

}