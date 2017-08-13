"""
This module contains the logic for finding the best performing system at equilibrium
"""

import itertools
import numpy as np
import math
from scipy.stats import t
import time

import gtconfig
import syseval
import simdata

if gtconfig.is_windows:
    import winsound

logger = gtconfig.get_logger("process_comparison", "process_comparison.txt")


def get_difference_sample_variance(system_1_samples, system_2_samples):
    """
    Obtains the sample variance of the difference

    :param system_1_samples: Samples from the first system.
    :param system_2_samples: Samples from the second system.
    :return: Sample variance of the difference.
    """
    system_1_mean = np.mean(system_1_samples)
    system_2_mean = np.mean(system_2_samples)

    accumulate = 0
    united_samples = zip(system_1_samples, system_2_samples)

    for system_1, system_2 in united_samples:
        accumulate += (system_1 - system_2 - (system_1_mean - system_2_mean)) ** 2

    return 1.0 / (len(united_samples) - 1.0) * accumulate


def get_new_sample_size(samples, initial_sample_size, confidence, difference):
    difference_variances = []

    for first_configuration, second_configuration in list(itertools.combinations(samples.keys(), 2)):
        logger.info("Calculating difference variance: " + first_configuration + " - " + second_configuration)
        difference_variances.append(
            get_difference_sample_variance(samples[first_configuration], samples[second_configuration]))

    largest_sample_variance = max(difference_variances)
    logger.info("Largest sample variance: " + str(largest_sample_variance))

    alpha = 1 - confidence
    design_number = len(samples.keys())
    degrees_of_freedom = initial_sample_size - 1
    t_parameter = alpha / (design_number - 1)
    t_score = t.ppf(t_parameter, degrees_of_freedom)

    logger.info("Difference: " + str(difference) + " Alpha: " + str(alpha) + " T-Score: " + str(float(t_score)))
    potential_sample_size = math.ceil(t_score ** 2 * largest_sample_variance / difference ** 2)

    logger.info("Potential sample size: " + str(float(potential_sample_size)))
    return max(initial_sample_size, potential_sample_size)


def main():
    initial_sample_size = 12
    confidence = 0.95
    difference = 0.1

    logger.info("Initial sample size: " + str(initial_sample_size))

    simulation_configuration, simfunction, input_params, empirical_profile = syseval.gather_experiment_inputs()
    simulation_configuration["REPLICATIONS_PER_PROFILE"] = initial_sample_size

    uo_equilibria = syseval.get_unsupervised_prioritization_equilibria(simulation_configuration, input_params)
    throttling_equilibria = syseval.get_throttling_equilibria(simulation_configuration, input_params)

    samples = {}
    for equilibrium_info in (uo_equilibria + throttling_equilibria):

        profiles = equilibrium_info["equilibrium_profiles"]
        configuration = equilibrium_info["desc"]

        logger.info("Configuration " + configuration + " has " + str(len(profiles)) + " equilibrium profiles")

        for index, profile in enumerate(profiles):
            sample_key = configuration + "_TSNE" + str(index)

            logger.info("Producing samples for " + sample_key)

            syseval.apply_strategy_profile(input_params.player_configuration, profile)
            simulation_output = syseval.run_scenario(simfunction, input_params,
                                                     equilibrium_info["simulation_configuration"])
            samples[sample_key] = simulation_output.get_fixed_ratio_per_priority(simdata.SEVERE_PRIORITY)

    new_sample_size = get_new_sample_size(samples=samples, initial_sample_size=initial_sample_size,
                                          confidence=confidence, difference=difference)
    logger.info("New sample size: " + str(new_sample_size))

    if new_sample_size != initial_sample_size:
        # TODO(cgavidia) Work this later
        pass

    means = {}
    for scenario, samples in samples.iteritems():
        overall_sample_mean = np.mean(samples)
        means[scenario] = overall_sample_mean
        logger.info("Overall Sample Mean for " + scenario + ": " + str(overall_sample_mean))

    best_performer_key = max(means, key=means.get)
    best_performer_value = means[best_performer_key]

    logger.info("Best performer " + best_performer_key + ". Value: " + str(best_performer_value))

    for scenario, mean in means.iteritems():
        left_parameter = mean - best_performer_value - difference
        right_parameter = mean - best_performer_value + difference

        left_boundary = min(0, left_parameter)
        right_boundary = max(0, right_parameter)

        logger.info(
            "Confidence interval for " + scenario + ": ( " + str(left_boundary) + ", " + str(right_boundary) + ")")

        if right_parameter <= 0:
            logger.info(scenario + " is inferior to the best")
        else:
            logger.info(scenario + " is statistically indistinguihable from the best")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    logger.info("Execution time in seconds: " + str(time.time() - start_time))
