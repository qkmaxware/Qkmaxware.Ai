namespace Qkmaxware.Ai.NeuralNetwork.Training;

/// <summary>
/// Class for storing data about genetic training population selection
/// </summary>
public class PopulationSelection {
    public double PercentageSelectedFromElite {get; private set;}
    public double PercentageSelectedFromMutation {get; private set;}
    public double PercentageSelectedFromCrossover {get; private set;}

    public PopulationSelection(double elite, double crossover, double mutation) {
        var total = Math.Min(elite + mutation + crossover, 1);

        this.PercentageSelectedFromElite = elite / total;
        this.PercentageSelectedFromCrossover = crossover / total;
        this.PercentageSelectedFromMutation = mutation / total;
    }
}

/// <summary>
/// Interface to generate a particular population for genetic training
/// </summary>
public interface IGenomeGenerator<TGenome> where TGenome:IGenome {
    /// <summary>
    /// Generate a sequence of genomes for a population up to the max size
    /// </summary>
    /// <param name="max">max number of genomes to generate</param>
    /// <returns>sequence of genomes</returns>
    public IEnumerable<TGenome> Generate(int max);
}

/// <summary>
/// Class for performing a genetic algorithm against a population set to get most suited for survival
/// </summary>
/// <typeparam name="TGenome">genome type to evolve</typeparam>
public class GeneticEvolution<TGenome> where TGenome:IGenome {
    public int MaxGenerations {get; private set;}
    public double DesiredAccuracy {get; private set;}
    public IFitnessTest<TGenome> FitnessTest {get; private set;}
    private IReproductiveRules<TGenome> reproductiveRules;
    
    public PopulationSelection PopulationSelection {get; private set;}

    public GeneticEvolution(int generations, double desiredAccuracy, IReproductiveRules<TGenome> reproduction, IFitnessTest<TGenome> fitness, PopulationSelection distribution) {
        this.MaxGenerations = Math.Max(generations, 1);
        this.DesiredAccuracy = desiredAccuracy;
        this.reproductiveRules = reproduction;
        this.FitnessTest = fitness;
        this.PopulationSelection = distribution;
    }

    private static readonly DiversityComparator byDiversity = new DiversityComparator();
    private static readonly FitnessComparator byFitness = new FitnessComparator();
    private static readonly CombinedRankComparator byCombinedRank = new CombinedRankComparator();

    public bool Evolve(IEnumerable<TGenome> initial_population, out TGenome best) {
        // Form population 0
        List<TableRow> population = initial_population.Select(x => new TableRow(x)).ToList();

        // Simulate each generation
        for (int generation = 0; generation < this.MaxGenerations; generation++) {
            // Ensure members to test
            if (population.Count < 0) {
                throw new ArgumentException("Empty population cannot be trained");
            }
            Console.WriteLine($"generation: {generation}");

            // Test fitness
            TestAll(population);

            // Sort by fitness
            population.Sort(byFitness);
            Rank(
                population, 
                (member, rank) => member.fitness_rank = rank, 
                (member, probability) => member.probability_of_selection_from_fitness = probability
            );
            
            // If best genome good enough?
            Console.WriteLine($"    best has fitness: {population[0].fitness}");
            Console.WriteLine($"    worst has fitness: {population[population.Count - 1].fitness}");
            if (Math.Abs(population[0].fitness) < this.DesiredAccuracy) {
                best = population[0].genome;
                return true;
            }

            // Create elite population
            population = Evolve(population);
        }

        // If best genome good enough?
        best = population[0].genome;
        return false;
    }

    private void TestAll(IEnumerable<TableRow> population) {
        foreach (var sample in population) {
            test(sample);
        }
    }
    private void test(TableRow row) {
        row.fitness = this.FitnessTest.TestError(row.genome);
    }

    private void Rank(List<TableRow> population, Action<TableRow, int> rank, Action<TableRow, double> selectionProbability) {
        double last_prob = 0;
        for (var i = 0; i < population.Count; i++) {
            var selection = population[i];
            rank(selection, i);
            last_prob = (0.667)*(1-last_prob);
            selectionProbability(selection, last_prob);
        }
    }

    private List<TableRow> Evolve(List<TableRow> current) {
        List<TableRow> next = new List<TableRow>(current.Count);

        // Select 
        EliteSelection(next, current);
        CrossoverSelection(next, current);
        MutationSelection(next, current);
        return next;
    }

    private void EliteSelection(List<TableRow> elite, List<TableRow> population) {
        // Select elites
        {
            // Add the population "strongest" member
            int elite_count = Math.Max((int)(population.Count * this.PopulationSelection.PercentageSelectedFromElite), 1);
            elite.Add(population[0]); 
            // Maybe remove it?

            // Add the remaining elites based on diversity score from the population selected elites
            for (var i = 1; i < elite_count; i++) {
                // First test for diversity from the elites
                TestDiversity(population, elite);
                population.Sort(byDiversity);
                Rank(
                    population, 
                    (member, rank) => member.diversity_rank = rank, 
                    (member, probability) => member.probability_of_selection_from_diversity = probability
                );

                // Combine fitness + diversity and use rank the options based on that 
                population.Sort(byCombinedRank);
                Rank(
                    population, 
                    (member, rank) => { /* Rank is a calculation, is not set here */ }, 
                    (member, probability) => member.probability_of_selection_from_combined_rank = probability
                );

                // Pick elite elite based on the combined rank
                var choice = rng.NextDouble();
                var sum = 0.0;
                foreach (var member in population) {
                    sum += member.probability_of_selection_from_combined_rank;
                    if (choice <= sum) {
                        elite.Add(member);
                        // Maybe remove it? Or does diversity make this less likely to happen
                        break;
                    }
                }
            }
        }
    }

    private static System.Random rng = new System.Random();
    private void CrossoverSelection(List<TableRow> elite, List<TableRow> population) {
        int crossover_count = Math.Max((int)(population.Count * this.PopulationSelection.PercentageSelectedFromCrossover), 0) / 2;
        
        for (var i = 0; i < crossover_count; i++) {
            var A = elite[rng.Next(elite.Count)];
            var B = elite[rng.Next(elite.Count)];

            var offspring = this.reproductiveRules.Crossover(A.genome, B.genome);
            elite.Add(new TableRow(offspring.Item1));
            elite.Add(new TableRow(offspring.Item2));
        }
    }

    private void MutationSelection(List<TableRow> next, List<TableRow> population) {
        int crossover_count = Math.Min((int)(population.Count * this.PopulationSelection.PercentageSelectedFromMutation), 0) / 2;
        int remaining_space = population.Count - next.Count;
        var amount_to_create = Math.Max(crossover_count, remaining_space);

        var rng = new System.Random();
        for (var i = 0; i < amount_to_create; i++) {
            var A = population[rng.Next(population.Count)]; // Random from the population, not the elite

            next.Add(new TableRow(this.reproductiveRules.Mutate(A.genome)));
        }
    }

    private void TestDiversity(List<TableRow> population, IEnumerable<TableRow> elite) {
        // Calculate diversity score
        foreach (var A in population) {
            var diversity = 0.0;
            foreach (var B in elite) {
                diversity += this.reproductiveRules.DifferenceBetween(A.genome, B.genome);
            }
            A.diversity = diversity;
        }
    }

    #region HelperClasses
    /// <summary>
    /// Utility class for genetic training
    /// </summary>
    private class TableRow{
        public TGenome genome;
        public double fitness;
        public int fitness_rank;
        public double probability_of_selection_from_fitness;
        public double diversity;
        public int diversity_rank;
        public double probability_of_selection_from_diversity;
        public double probability_of_selection_from_combined_rank;
        
        public TableRow(TGenome genome) {
            this.genome = genome;
        }

        public int CombinedRank(){
            return fitness_rank + diversity_rank;
        }
        
    }

    private class FitnessComparator : IComparer<TableRow> {
        public int Compare(TableRow? x, TableRow? y) {
            #nullable disable
            return (x.fitness).CompareTo(y.fitness); // put low values first (low error = high fitness)
            #nullable restore
        }
    }
    private class DiversityComparator : IComparer<TableRow> {
        public int Compare(TableRow? x, TableRow? y) {
            #nullable disable
            return -(x.diversity).CompareTo(y.diversity); // high diversity first
            #nullable restore
        }
    }
    private class CombinedRankComparator : IComparer<TableRow> {
        public int Compare(TableRow? x, TableRow? y) {
            #nullable disable
            return (x.CombinedRank()).CompareTo(y.CombinedRank()); // low rank first 
            #nullable restore
        }
    }
    #endregion
}

/// <summary>
/// Class for using genetic training to evolve a neural network into a more accurate one
/// </summary>
/// <typeparam name="TGenome">genome type to evolve, genome is decodable as a neural network</typeparam>
public class GeneticEvolutionTrainer<TGenome> : GeneticEvolution<TGenome>, ITrainer<IGenomicNeuralNetwork<TGenome>> where TGenome : IDecodableGenome<IGenomicNeuralNetwork<TGenome>> {

    private int _defPop = 500;
    public int DefaultPopulation {
        get => _defPop;
        set => DefaultPopulation = Math.Max(value, 1);
    }

    public GeneticEvolutionTrainer(int generations, double desiredAccuracy, IReproductiveRules<TGenome> reproduction, IFitnessTest<TGenome> fitness, PopulationSelection distribution) 
    : base(generations, desiredAccuracy, reproduction, fitness, distribution) {}

    public bool TryTrain(IEnumerable<TGenome> initial, out IGenomicNeuralNetwork<TGenome> trained) {
        TGenome best;
        bool success = this.Evolve(initial, out best);
        trained = best.Decode();

        return success;
    }

    public bool TryTrain(IGenomicNeuralNetwork<TGenome> initial, out IGenomicNeuralNetwork<TGenome> trained) {
        var genome = initial.EncodeToGenome();
        var population = new TGenome[DefaultPopulation];
        for (var i = 0; i < population.Length; i++) {
            population[i] = genome; // Same genome 'x' times in initial population
        }

        TGenome best;
        bool success = this.Evolve(population, out best);
        trained = best.Decode();

        return success;
    }
}