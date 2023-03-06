using static System.Console;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Qkmaxware.Ai.NeuralNetwork;
using Qkmaxware.Ai.NeuralNetwork.Activation;
using Qkmaxware.Ai.NeuralNetwork.Training;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Drawing;

namespace Qkmaxware.Ai.Test;

[TestClass]
public class LetterDecodeTests {

    private class LetterDecoderGenerator : IGenomeGenerator<FeedForwardNeuralNetworkGenome> {
        private FeedForwardNeuralNetwork template = new FeedForwardNeuralNetwork(
            activationFunction: new Exponential(),
            64 * 64, 2 * 64 * 64, 26
        );

        public IEnumerable<FeedForwardNeuralNetworkGenome> Generate(int max) {
            for (var i = 0; i < max; i++) {
                template.RandomizeSynapseWeights();
                yield return template.EncodeToGenome();
            }
        }
    }

    private static ListTrainingSet ReadImages() {
        ListTrainingSet data = new ListTrainingSet();

        var assembly = typeof(LetterDecodeTests).Assembly;
        var defaultFontFolder = assembly.GetName().Name + ".tests.letters.";
        foreach (var resource in assembly.GetManifestResourceNames().Where(name => name.StartsWith(defaultFontFolder) && name.EndsWith(".png"))) {
            var filename = Path.GetFileNameWithoutExtension(resource.Substring(defaultFontFolder.Length));
            
            using(var stream = assembly.GetManifestResourceStream(resource)) {
                if (stream == null)
                    continue;
                try {
                    // Load image pixels into a row major vector
                    var bmp = new Bitmap(stream);
                    var size = bmp.Width * bmp.Height;
                    UntaggedVector pixels = new UntaggedVector(size);
                    int i = 0;
                    for (var row = 0; row < bmp.Height; row++) {
                        for (var col = 0; col < bmp.Width; col++) {
                            var pixel = bmp.GetPixel(col, row);
                            pixels[i++] = (pixel.R + pixel.G + pixel.B) / (255.0 * 3.0);
                        }
                    }

                    // Use filename to map this image to a particular letter
                    string[] tags = new string[]{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
                    var match = new TaggedVector<string>(tags);
                    var index = System.Array.IndexOf(tags, Path.GetFileNameWithoutExtension(filename.ToUpper()));
                    if (index == -1)
                        continue;
                    match[index] = 1;

                    // Create the training pair
                    var pair = new SimpleTrainingPair(
                        pixels,
                        match
                    );
                    data.Add(pair);
                } catch { 
                    // Eat bad characters just in case. We don't want failure here
                }
            }
        }
        return data;
    }

    [TestMethod]
    public void GeneticWeights() {
        // Create training data
        var training = ReadImages();

        // Training the neural network (also validates the results)
        var trainer = new GeneticEvolutionTrainer<FeedForwardNeuralNetworkGenome>(
            generations: 1000, 
            desiredAccuracy: 0.1,
            reproduction: new FeedForwardNeuralNetworkMeiosis(),
            fitness: new NeuralNetworkFitness<IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome>, FeedForwardNeuralNetworkGenome>(training),
            distribution: new PopulationSelection(elite: 2, crossover: 1, mutation: 4)
        );
        IGenomicNeuralNetwork<FeedForwardNeuralNetworkGenome> best;
        Assert.AreEqual(true, trainer.TryTrain(new LetterDecoderGenerator().Generate(500), out best));  // Check that the training was a success
    }

}