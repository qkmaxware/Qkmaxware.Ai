namespace Qkmaxware.Ai.NeuralNetwork.Activation;

/// <summary>
/// Signum activation function
/// </summary>
public class Signum : BaseActivationFunction {

    public override double ValueAt(double x) {
        return x >= 0 ? 1 : -1;
    }

    public override double DerivativeAt(double x) {
        return 0;
    }
}