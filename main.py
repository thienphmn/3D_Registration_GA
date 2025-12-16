from modules import image_utils as iu
import SimpleITK as sitk
import numpy as np
from joblib import Parallel, delayed
import time, scipy, os
import matplotlib.pyplot as plt


# === Population Initialization ===
def initialize_population(size: int, tx_bounds: tuple, ty_bounds: tuple, tz_bounds: tuple,
                          rx_bounds: tuple, ry_bounds: tuple, rz_bounds: tuple) -> np.ndarray:
    """Initialize population with individuals containing transformation parameters."""
    tx = np.random.uniform(*tx_bounds, size=size)
    ty = np.random.uniform(*ty_bounds, size=size)
    tz = np.random.uniform(*tz_bounds, size=size)
    rx = np.random.uniform(*rx_bounds, size=size)
    ry = np.random.uniform(*ry_bounds, size=size)
    rz = np.random.uniform(*rz_bounds, size=size)

    population = np.stack((tx, ty, tz, rx, ry, rz), axis=1)
    return population


# === Fitness Evaluation ===
def evaluate_fitness(fixed: sitk.Image, moving: sitk.Image, population: np.ndarray) -> np.ndarray:
    """Evaluate fitness scores across a population."""
    def evaluate_individual(individual):
        resampled_image = iu.transform_image(moving=moving, fixed=fixed,
                                             tx = individual[0],
                                             ty = individual[1],
                                             tz = individual[2],
                                             rx = individual[3],
                                             ry = individual[4],
                                             rz = individual[5])
        return iu.calculate_mutual_information(fixed=fixed, resampled=resampled_image)

    fitness_scores = (Parallel(n_jobs=-1)
                      (delayed(evaluate_individual)(individual) for individual in population))

    return np.array(fitness_scores)


# === Selection Methods ===
def tournament_selection(population: np.ndarray,
                         fitness: np.ndarray,
                         num_selected: int,
                         tournament_size: int = 3) -> tuple:
    """Tournament Selection, where random individuals enter tournament where the fittest survives."""
    n_individuals = population.shape[0]
    selected_indices = np.zeros(num_selected, dtype=int)

    for i in range(num_selected):
        # Select random individuals and get their similarity scores
        tournament_indices = np.random.choice(a=n_individuals, size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]

        # Select individual with the highest fitness scores
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices[i] = winner_index

    selected_population = population[selected_indices]
    return selected_population, selected_indices


# === Crossover Methods ===
def create_mates(selected_pop: np.ndarray, target_size: int) -> tuple:
    """Randomly select parent indices indicating which parents will produce offspring"""
    if target_size % 2 != 0:
        target_size += 1

    # randomly select indices of individuals who will mate
    n_selected = selected_pop.shape[0]
    parent_pairs = np.random.choice(n_selected,
                                    size=(target_size // 2, 2),
                                    replace=True)
    parent_indices_1 = parent_pairs[:, 0]
    parent_indices_2 = parent_pairs[:, 1]

    # Retrieve parent pairs
    parents1 = selected_pop[parent_indices_1]  # shape: (n_pairs, n_params)
    parents2 = selected_pop[parent_indices_2]  # shape: (n_pairs, n_params)
    return parents1, parents2

def arithmetic_crossover(selected_pop: np.ndarray,
                         target_size: int) -> np.ndarray:
    """Arithmetic Crossover, where the child is the weighted average of the parents."""
    # Get parent indices
    parents1, parents2 = create_mates(selected_pop, target_size)

    # Randomize alphas and creating children
    alphas1 = np.random.uniform(0.1, 0.9, size=(target_size // 2, 1))
    alphas2 = np.random.uniform(0.1, 0.9, size=(target_size // 2, 1))

    child1 = alphas1 * parents1 + (1 - alphas1) * parents2
    child2 = alphas2 * parents2 + (1 - alphas2) * parents1

    # Stack all offspring and trim to exact target size
    offspring = np.vstack((child1, child2))[:target_size]
    return offspring


def blend_crossover(selected_pop: np.ndarray,
                    target_size: int,
                    mins: np.ndarray,
                    maxs: np.ndarray,
                    alpha: float = 0.25) -> np.ndarray:
    """Blend Crossover, where children are sampled from an extended range around the parents' values."""
    # Get parent indices
    parents1, parents2 = create_mates(selected_pop=selected_pop, target_size=target_size)

    # Calculate the range between parents for each parameter
    d = np.abs(parents1 - parents2)  # shape: (n_pairs, n_params)

    # Define the extended interval [interval_min, interval_max] for each parameter
    interval_min = np.minimum(parents1, parents2) - alpha * d
    interval_max = np.maximum(parents1, parents2) + alpha * d

    # Sample children uniformly from the extended intervals
    child1 = np.clip(np.random.uniform(interval_min, interval_max), mins, maxs)
    child2 = np.clip(np.random.uniform(interval_min, interval_max), mins, maxs)

    # Stack all offspring and trim to exact target size
    offspring = np.vstack((child1, child2))[:target_size]
    return offspring


# === Mutation Methods ===
def gaussian_mutation(population: np.ndarray,
                      mutation_rate: np.ndarray,
                      mins: np.ndarray,
                      maxs: np.ndarray) -> np.ndarray:
    """Gaussian Mutation where gaussian noise is added onto existing individuals."""
    n_individuals, n_params = population.shape
    mutated_pop = population.copy()

    # Generate Gaussian noise for all parameters (vectorized)
    noise = np.random.normal(0, mutation_rate, (n_individuals, n_params))
    # Add gaussian noise to population
    mutated_pop = mutated_pop + noise
    # Clip to bounds
    mutated_pop = np.clip(mutated_pop, mins, maxs)

    return mutated_pop


def linear_decrease_mutation(initial_rate: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
    """Linearly decrease mutation rate over generations."""
    return initial_rate * (1 - generation/ max_generations)


def exponential_decrease_mutation(initial_rate: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
    """Exponentially decrease mutation rate over generations."""
    k = 1.0 / max_generations
    return initial_rate * np.exp(-k * generation)


# === Metric Tracking ===
def compute_normalized_diversity(population: np.ndarray, min_bounds: np.ndarray, max_bounds:np.ndarray) -> float:
    """Average pairwise Euclidean distance in population normalized between 0 and 1."""
    # Compute average pairwise distance
    avg_distance = scipy.spatial.distance.pdist(population).mean()

    # Maximum possible distance in this 3D space
    translation_range = max_bounds[0] - min_bounds[0]
    rotation_range = max_bounds[2] - min_bounds[2]
    max_distance = np.sqrt(translation_range ** 2 + translation_range ** 2 + rotation_range ** 2)

    # Normalize to [0,1]
    return avg_distance / max_distance


def convergence_metrics(population: np.ndarray, fitness: np.ndarray,
                        min_bounds: np.ndarray, max_bounds: np.ndarray) -> tuple:
    """Calculation of Convergence Metrics."""
    parameter_diversity = compute_normalized_diversity(population, min_bounds, max_bounds)
    fitness_std = np.std(fitness)
    fitness_mean = np.mean(fitness)
    fitness_best = np.max(fitness)

    return parameter_diversity, fitness_std, fitness_mean, fitness_best


# === Genetic Algorithm Design (µ+λ) ===
def run_genetic_algorithm(fixed_image: sitk.Image, moving_image: sitk.Image, modality: str):
    # Configuration Parameters
    INITIAL_POP_SIZE = 100
    MAX_GENERATION = 50
    MIN_BOUNDS = np.array([-50.0, -50.0, -50.0, -5.0, -5.0, -10.0])  # order: [tx,ty,tz,rx,ry,rz]
    MAX_BOUNDS = MIN_BOUNDS * -1
    PARAMETER_RANGE_FRACTION = 0.15
    INITIAL_MUTATION_RATE = np.array([PARAMETER_RANGE_FRACTION * np.abs(MIN_BOUNDS[0]-MAX_BOUNDS[0]),
                                      PARAMETER_RANGE_FRACTION * np.abs(MIN_BOUNDS[1]-MAX_BOUNDS[1]),
                                      PARAMETER_RANGE_FRACTION * np.abs(MIN_BOUNDS[2]-MAX_BOUNDS[2]),
                                      PARAMETER_RANGE_FRACTION * np.abs(MIN_BOUNDS[3]-MAX_BOUNDS[3]),
                                      PARAMETER_RANGE_FRACTION * np.abs(MIN_BOUNDS[4]-MAX_BOUNDS[4]),
                                      PARAMETER_RANGE_FRACTION * np.abs(MIN_BOUNDS[5]-MAX_BOUNDS[5])])

    # Load CT (fixed) and MRI (moving) and their downscaled versions for coarse to fine registration
    fixed_full = fixed_image
    moving_full = moving_image
    fixed_eighth = iu.smooth_and_resample(fixed_full, 8)
    fixed_fourth = iu.smooth_and_resample(fixed_full, 4)
    fixed_half = iu.smooth_and_resample(fixed_full, 2)
    moving_half = iu.smooth_and_resample(moving_full, 2)

    # Initialize first generation and evaluate fitness
    population = initialize_population(size=INITIAL_POP_SIZE,
                                       tx_bounds=(MIN_BOUNDS[0], MAX_BOUNDS[0]),
                                       ty_bounds=(MIN_BOUNDS[1], MAX_BOUNDS[1]),
                                       tz_bounds=(MIN_BOUNDS[2], MAX_BOUNDS[2]),
                                       rx_bounds=(MIN_BOUNDS[3], MAX_BOUNDS[3]),
                                       ry_bounds=(MIN_BOUNDS[4], MAX_BOUNDS[4]),
                                       rz_bounds=(MIN_BOUNDS[5], MAX_BOUNDS[5]))
    fitness = evaluate_fitness(fixed=fixed_eighth, moving=moving_half, population=population)

    # Initialize empty arrays to save convergence metrics
    parameter_diversity = np.empty(shape=(MAX_GENERATION+1,))
    fitness_std = np.empty(shape=(MAX_GENERATION+1,))
    fitness_mean = np.empty(shape=(MAX_GENERATION+1,))
    fitness_best = np.empty(shape=(MAX_GENERATION+1,))

    # Core Genetic Algorithm Loop
    start = time.perf_counter()

    # Pyramid registration setup and adaptive population size
    for generation in range(MAX_GENERATION):
        if generation <= int(MAX_GENERATION * 0.5):
            fixed = fixed_eighth
            moving = moving_half
            pop_size = int(INITIAL_POP_SIZE)
        elif generation <= int(MAX_GENERATION * 0.8):
            fixed = fixed_fourth
            moving = moving_full
            pop_size = int(INITIAL_POP_SIZE * (3/4))
        elif generation <= int(MAX_GENERATION * 0.95):
            fixed = fixed_half
            moving = moving_full
            pop_size = int(INITIAL_POP_SIZE * (2/4))
        else:
            fixed = fixed_full
            moving = moving_full
            pop_size = int(INITIAL_POP_SIZE * (1/4))


        # Recording Metrics
        (p_diversity, fit_std, fit_mean, fit_best) = convergence_metrics(population=population,
                                                                         fitness=fitness,
                                                                         min_bounds=MIN_BOUNDS,
                                                                         max_bounds=MAX_BOUNDS)
        parameter_diversity[generation] = p_diversity
        fitness_std[generation] = fit_std
        fitness_mean[generation] = fit_mean
        fitness_best[generation] = fit_best

        # Parent Selection
        parents = tournament_selection(population=population, fitness=fitness,
                                       num_selected=pop_size, tournament_size=2)[0]

        # Offspring creation
        offspring = blend_crossover(selected_pop=parents, target_size=3 * pop_size, alpha=0.3,
                                    mins=MIN_BOUNDS, maxs=MAX_BOUNDS)

        # Offspring mutation
        mutation_rate = linear_decrease_mutation(initial_rate=INITIAL_MUTATION_RATE,
                                                 generation=generation+1,
                                                 max_generations=MAX_GENERATION+1)

        offspring = gaussian_mutation(population = offspring,
                                      mutation_rate=mutation_rate,
                                      mins=MIN_BOUNDS,
                                      maxs=MAX_BOUNDS)

        # Offspring fitness calculation
        offspring_fitness = evaluate_fitness(fixed=fixed, moving=moving, population=offspring)

        # Combine populations
        combined_pop = np.vstack([population, offspring])
        combined_fitness = np.concatenate([fitness, offspring_fitness])

        # Select best mu individuals
        best_indices = np.argsort(combined_fitness)[-pop_size:]
        population = combined_pop[best_indices]
        fitness = combined_fitness[best_indices]
        print(f"Generation {generation+1} done.")

    # Get optimal transformation parameters for transformation of moving image
    best_index = np.argmax(fitness)
    best_individual = population[best_index]
    best_score = fitness[best_index]
    end = time.perf_counter()
    print(f"Time to optimize registration: {end-start} seconds.")

    # Recording Metrics for the last generation
    (p_diversity, fit_std, fit_mean, fit_best) = convergence_metrics(population=population,
                                                                     fitness=fitness,
                                                                     min_bounds=MIN_BOUNDS,
                                                                     max_bounds=MAX_BOUNDS)
    parameter_diversity[MAX_GENERATION] = p_diversity
    fitness_std[MAX_GENERATION] = fit_std
    fitness_mean[MAX_GENERATION] = fit_mean
    fitness_best[MAX_GENERATION] = fit_best

    # Plotting convergence metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    axs[0].plot(parameter_diversity, label="Parameter Diversity", color="purple")
    axs[0].set_title("Parameter Diversity")
    axs[0].set_ylabel("Diversity")
    axs[0].set_xlabel("Generation")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(fitness_std, label="Fitness Std", color="orange")
    axs[1].set_title("Fitness Standard Deviation")
    axs[1].set_ylabel("Std")
    axs[1].set_xlabel("Generation")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(fitness_mean, label="Fitness Mean", color="blue")
    axs[2].set_title("Fitness Mean")
    axs[2].set_ylabel("Mean Fitness")
    axs[2].set_xlabel("Generation")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(fitness_best, label="Fitness Best", color="green")
    axs[3].set_title("Best Fitness")
    axs[3].set_ylabel("Best Fitness")
    axs[3].set_xlabel("Generation")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "graphs", f"{modality}"))
    plt.show()
    plt.close()

    return best_individual, best_score


def main(patient_number: str, modality: str):
    ct_path = os.path.abspath(os.path.join("dataset", "RIRE", f"patient_{patient_number}", "ct",
                                           f"patient_00{patient_number}_ct.mhd"))
    mri_path = os.path.abspath(os.path.join("dataset", "RIRE", f"patient_{patient_number}", f"mr_{modality}_rectified",
                                            f"patient_00{patient_number}_mr_{modality}_rectified.mhd"))

    ct = iu.load_image(ct_path, "CT")
    mri = iu.load_image(mri_path, "MRI")

    best_individual, best_score = run_genetic_algorithm(fixed_image=ct, moving_image=mri, modality=modality)
    print(best_individual, best_score)

    resampled_image = iu.transform_image(moving=mri, fixed=ct,
                                         tx=float(best_individual[0]),
                                         ty=float(best_individual[1]),
                                         tz=float(best_individual[2]),
                                         rx=float(best_individual[3]),
                                         ry=float(best_individual[4]),
                                         rz=float(best_individual[5]))

    volume_folder = os.path.abspath(os.path.join("results", "volumes", f"patient_{patient_number}_{modality}.nii.gz"))
    sitk.WriteImage(resampled_image, volume_folder)



if __name__ == "__main__":
    patient_number = "3"
    modality = "T2"

    # main(patient_number, modality)

    # === For visualising results ===
    ct_path = os.path.abspath(os.path.join("dataset", "RIRE", f"patient_{patient_number}", "ct",
                                           f"patient_00{patient_number}_ct.mhd"))
    mri_path = os.path.abspath(os.path.join("dataset", "RIRE", f"patient_{patient_number}", f"mr_{modality}_rectified",
                                            f"patient_00{patient_number}_mr_{modality}_rectified.mhd"))
    mri_registered_path = os.path.abspath(os.path.join("results", "volumes", f"patient_{patient_number}_{modality}.nii.gz"))

    ct = iu.load_image(ct_path, "CT")
    ct_padded = sitk.ConstantPad(ct,
                          padLowerBound = [10,10,10],
                          padUpperBound = [10,10,10],
                          constant = 0.0)
    mri = iu.load_image(mri_path, "MRI")
    mri_registered = sitk.ReadImage(mri_registered_path)

    iu.show_3d(ct_padded, mri)
    iu.show_3d(ct_padded, mri_registered)
    iu.show_registered_images(ct, mri)
    iu.show_registered_images(ct, mri_registered, True)
