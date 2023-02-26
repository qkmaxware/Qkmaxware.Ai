namespace Qkmaxware.Ai.NeuralNetwork.Activation;

/// <summary>
/// Exponential activation function
/// </summary>
public class Exponential : BaseActivationFunction {
    public override double ValueAt(double x) {
        return 1.0 / (1 + Math.Exp(-x));
    }

    public override double DerivativeAt(double x) {
        return Math.Exp(x) / Math.Pow(Math.Exp(x) + 1, 2);
    }
}