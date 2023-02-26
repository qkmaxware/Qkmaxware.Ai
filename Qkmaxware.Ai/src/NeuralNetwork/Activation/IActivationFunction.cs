namespace Qkmaxware.Ai.NeuralNetwork.Activation;

/// <summary>
/// Interface for activation functions
/// </summary>
public interface IActivationFunction {
    /// <summary>
    /// Evaluate the value of the function at the position 'x'
    /// </summary>
    /// <param name="x">position to evaluate at</param>
    /// <returns>value of the function at position 'x'</returns>
    public double ValueAt(double x);
    /// <summary>
    /// Evaluate the value of the derivative of the function at the position 'x'
    /// </summary>
    /// <param name="x">position to evaluate at</param>
    /// <returns>value of the derivative of the function at position 'x'</returns>
    public double DerivativeAt(double x);
}

/// <summary>
/// Base class for build-in activation functions
/// </summary>
public abstract class BaseActivationFunction : IActivationFunction
{
    /// <summary>
    /// Evaluate the value of the function at the position 'x'
    /// </summary>
    /// <param name="x">position to evaluate at</param>
    /// <returns>value of the function at position 'x'</returns>
    public abstract double ValueAt(double x);
    /// <summary>
    /// Evaluate the value of the derivative of the function at the position 'x'
    /// </summary>
    /// <param name="x">position to evaluate at</param>
    /// <returns>value of the derivative of the function at position 'x'</returns>
    public abstract double DerivativeAt(double x);
}