namespace Qkmaxware.Ai.NeuralNetwork.Activation;

/// <summary>
/// Hyperbolic tangent activation function
/// </summary>
public class Tanh : BaseActivationFunction {

    public double Scale {get; private set;}
    public double Stretch {get; private set;}

    public Tanh() : this(1.7, 0.6) {}

    public Tanh(double scale, double stretch) {
        this.Scale = scale;
        this.Stretch = stretch;
    }

    public override double ValueAt(double x) {
        return Scale * Math.Tanh(Stretch * x);
    }

    private static double Sech2(double x) {
        var sech = 1.0/Math.Cosh(x);
        return sech * sech;
    }

    public override double DerivativeAt(double x) {
        return Stretch * Scale * Sech2(Stretch * x);
        //return 4.08 * (Math.Cosh(0.6*x) * Math.Cosh(0.6*x)) / ((Math.Cosh(1.2 * x) + 1) * (Math.Cosh(1.2 * x) + 1));
    }
}