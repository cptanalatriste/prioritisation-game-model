"""
This module contains the logic for finding the best performing system at equilibrium, using the
two-stage Bonferroni procedure described by Banks in Discrete-Event System Simulation (page 399)
"""

import itertools
import numpy as np
import math
from scipy.stats import t
import time
from matplotlib import pyplot as plt
import pandas as pd

import eqcatalog
import gtconfig
import syseval
import simdata

if gtconfig.is_windows:
    import winsound

logger = gtconfig.get_logger("process_comparison", "bestperformer.txt")


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


def plot_results(means, desc=""):
    # label_map = {"GATEKEEPER_SUCC100": "Gatekeeper (0% Error)",
    #              "GATEKEEPER_SUCC090": "Gatekeeper (10% Error)",
    #              "THROTTLING_INF005": "Throttling (5% Penalty)",
    #              "THROTTLING_INF001": "Throttling (1% Penalty)",
    #              "THROTTLING_INF003": "Throttling (3% Penalty)",
    #              "UNSUPERVISED": "Unsupervised Prioritization"}
    #
    # color_map = {"GATEKEEPER_SUCC100": "blue",
    #              "GATEKEEPER_SUCC090": "darkblue",
    #              "THROTTLING_INF005": "red",
    #              "THROTTLING_INF001": "firebrick",
    #              "THROTTLING_INF003": "tomato",
    #              "UNSUPERVISED": "green"}

    sorted_keys = sorted(means.keys())
    data = [means[key] for key in sorted_keys]
    # colors = []
    # labels = []

    # for key in sorted_keys:
    #     for prefix in color_map.keys():
    #         if key.startswith(prefix):
    #             colors.append(color_map[prefix])
    #             labels.append(label_map[prefix])
    #             break

    plt.clf()
    fig, ax = plt.subplots()
    width = 0.5
    tick_locations = np.arange(len(sorted_keys))
    rect_locations = tick_locations - (width / 2.0)

    plt.xticks(rotation="vertical")
    # ax.bar(rect_locations, data, width, color=colors)

    ax.bar(rect_locations, data, width)

    ax.set_xticks(ticks=tick_locations)
    # ax.set_xticklabels(labels)
    ax.set_xticklabels(sorted_keys)

    ax.set_xlim(min(tick_locations) - 0.6, max(tick_locations) + 0.6)
    ax.set_ylim((0, 1.0))

    ax.yaxis.grid(True)
    ax.set_xlabel('Bug Reporting Process Equilibrium')
    ax.set_ylabel('% Severe Reports Fixed')

    fig.suptitle("Performance Comparison")
    fig.tight_layout(pad=2)

    file_name = 'img/' + desc + "_performance_comparison.png"
    fig.savefig(file_name, dpi=125)

    logger.info("Comparison plot saved at " + file_name)


def main():
    initial_sample_size = 120

    confidence = 0.95
    difference = 0.05

    dev_team_factors = [0.5, 1.0]
    priority_disciplines = [False, True]

    simulation_configuration, simfunction, input_params, empirical_profile = syseval.gather_experiment_inputs()
    simulation_configuration["REPLICATIONS_PER_PROFILE"] = initial_sample_size
    original_team_size = input_params.dev_team_size

    logger.info("Initial sample size: " + str(initial_sample_size))
    logger.info("Original team size: " + str(original_team_size))

    for priority_discipline in priority_disciplines:
        simulation_configuration["PRIORITY_QUEUE"] = priority_discipline

        logger.info("Using Priority Queue? " + str(priority_discipline))

        for dev_team_factor in dev_team_factors:

            logger.info("Using dev team factor " + str(dev_team_factor))

            input_params.dev_team_size = int(original_team_size * dev_team_factor)

            uo_equilibria = eqcatalog.get_unsupervised_prioritization_equilibria(simulation_configuration, input_params,
                                                                                 priority_queue=priority_discipline,
                                                                                 dev_team_factor=dev_team_factor)
            gatekeeper_equilibria = eqcatalog.get_gatekeeper_equilibria(simulation_configuration, input_params,
                                                                        priority_queue=priority_discipline,
                                                                        dev_team_factor=dev_team_factor)
            throttling_equilibria = eqcatalog.get_throttling_equilibria(simulation_configuration, input_params,
                                                                        priority_queue=priority_discipline,
                                                                        dev_team_factor=dev_team_factor)

            samples = {}
            for equilibrium_info in (uo_equilibria + throttling_equilibria + gatekeeper_equilibria):

                profiles = equilibrium_info["equilibrium_profiles"]
                configuration = equilibrium_info["desc"]

                logger.info("Configuration " + configuration + " has " + str(len(profiles)) + " equilibrium profiles")

                success_rate = 1.0
                if 'SUCCESS_RATE' in equilibrium_info["simulation_configuration"]:
                    success_rate = equilibrium_info["simulation_configuration"]["SUCCESS_RATE"]

                input_params.catcher_generator.configure(values=[True, False],
                                                         probabilities=[success_rate, (1 - success_rate)])

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
                raise Exception("New sample collection is needed!")

            means = {}

            for scenario, samples in samples.iteritems():
                overall_sample_mean = np.mean(samples)
                means[scenario] = overall_sample_mean
                logger.info("Overall Sample Mean for " + scenario + ": " + str(overall_sample_mean))

            best_performer_key = max(means, key=means.get)
            best_performer_value = means[best_performer_key]

            logger.info("Best performer " + best_performer_key + ". Value: " + str(best_performer_value))

            experiment_desc = "priority_queue_" + str(priority_discipline) + "_dev_team_factor_" + str(dev_team_factor)
            plot_results(means, desc=experiment_desc)

            report_rows = []

            for scenario, mean in means.iteritems():
                left_parameter = mean - best_performer_value - difference
                right_parameter = mean - best_performer_value + difference

                left_boundary = min(0, left_parameter)
                right_boundary = max(0, right_parameter)

                logger.info(
                    "Confidence interval for " + scenario + ": ( " + str(left_boundary) + ", " + str(
                        right_boundary) + ")")

                if right_parameter <= 0:
                    logger.info(scenario + " is inferior to the best")
                    inferior_to_best = True
                else:
                    logger.info(scenario + " is statistically indistinguishable from the best")
                    inferior_to_best = False

                report_rows.append({'scenario': scenario,
                                    'mean': mean,
                                    'inferior_to_best': inferior_to_best})

            experiment_results = pd.DataFrame(report_rows)
            file_name = "csv/performance_exp_" + experiment_desc + ".csv"
            experiment_results.to_csv(file_name, index=False)
            
            logger.info("Experiment results written to: " + file_name)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    logger.info("Execution time in seconds: " + str(time.time() - start_time))
