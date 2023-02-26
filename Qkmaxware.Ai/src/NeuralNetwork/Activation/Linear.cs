namespace Qkmaxware.Ai.NeuralNetwork.Activation;

/// <summary>
/// Linear activation function
/// </summary>
public class Linear : BaseActivationFunction {

    public double Slope {get; private set;}

    public Linear() : this(1.0) {}

    public Linear(double slope) { 
        this.Slope = slope;
    }

    public override double ValueAt(double x) {
        return Slope * x;
    }

    public override double DerivativeAt(double x) {
        return Slope;
    }
}