"""
This module contains too for system comparison anc evaluation with simulation models.

The approach is the one proposed by Jeremy Banks in Discrete-Event System Simulation (Chapter 12)
"""
import numpy as np
import statsmodels.stats.api as sms
import pandas as pd
import sys

import time

import gtconfig
import payoffgetter
import penaltyexp
import simdata
import simdriver
import simmodel
import simutils

if gtconfig.is_windows:
    import winsound

logger = gtconfig.get_logger("process_comparison", "process_comparison.txt")

# Empirical strategy catalog
empirical_honest = {"name": "empirical_honest",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.05,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.08}

persistent_deflator = {"name": "persistent_deflator",
                       simmodel.NON_SEVERE_INFLATED_COLUMN: 0.08,
                       simmodel.SEVERE_DEFLATED_COLUMN: 1.00}

regular_deflator = {"name": "regular_deflator",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.04,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.58}

empirical_inflator = {"name": "empirical_inflator",
                      simmodel.NON_SEVERE_INFLATED_COLUMN: 0.19,
                      simmodel.SEVERE_DEFLATED_COLUMN: 0.02}

occasional_deflator = {"name": "occasional_deflator",
                       simmodel.NON_SEVERE_INFLATED_COLUMN: 0.06,
                       simmodel.SEVERE_DEFLATED_COLUMN: 0.26}


def compare_with_independent_sampling(first_system_replications, second_system_replications,
                                      first_system_desc="System 1",
                                      second_system_desc="System 2", alpha=0.05):
    """
    Different and independent random number streams will be used to simulate the two systems. We are not assuming that
    the variances are equal.

    :param first_system_replications: Observations for simulated system 1
    :param second_system_replications: Observations for simulated system 2
    :return:
    """

    logger.info("Comparing systems performance: " + first_system_desc + " vs " + second_system_desc)

    first_system_mean = np.mean(first_system_replications)
    first_system_variance = np.var(first_system_replications, ddof=1)
    logger.info(first_system_desc + ": Sample mean " + str(first_system_mean) + " Sample variance: " + str(
        first_system_variance))

    second_system_mean = np.mean(second_system_replications)
    second_system_variance = np.var(second_system_replications, ddof=1)

    logger.info(second_system_desc + ": Sample mean " + str(second_system_mean) + " Sample variance: " + str(
        second_system_variance))

    point_estimate = first_system_mean - second_system_mean
    logger.info("Point estimate: " + str(point_estimate))

    compare_means = sms.CompareMeans(sms.DescrStatsW(data=first_system_replications),
                                     sms.DescrStatsW(data=second_system_replications))
    conf_interval = compare_means.tconfint_diff(usevar="unequal", alpha=alpha)
    logger.info("Confidence Interval with alpha " + str(alpha) + " : " + str(conf_interval))


def test():
    # This data is taken from the book. We use it for testing only
    first_system_replications = [29.59, 23.49, 25.68, 41.09, 33.84, 39.57, 37.04, 40.20, 61.82, 44.00]
    second_system_replications = [51.62, 51.91, 45.27, 30.85, 56.15, 28.82, 41.30, 73.06, 23.00, 28.44]

    compare_with_independent_sampling(first_system_replications, second_system_replications)


def run_scenario(simfunction, input_params, simulation_configuration):
    """
    Convenient method, to avoid copy-pasting.
    :param simfunction:
    :param input_params:
    :param simulation_configuration:
    :param inflation_factor:
    :param gatekeeper_config:
    :return: Samples for the variable of interest.
    """
    simulation_output = simfunction(
        team_capacity=input_params.dev_team_size,
        ignored_gen=input_params.ignored_gen,
        reporter_gen=input_params.reporter_gen,
        target_fixes=input_params.target_fixes,
        batch_size_gen=input_params.batch_size_gen,
        interarrival_time_gen=input_params.interarrival_time_gen,
        priority_generator=input_params.priority_generator,
        reporters_config=input_params.player_configuration,
        resolution_time_gen=input_params.resolution_time_gen,
        max_time=sys.maxint,
        catcher_generator=input_params.catcher_generator,
        max_iterations=simulation_configuration["REPLICATIONS_PER_PROFILE"],
        inflation_factor=simulation_configuration["INFLATION_FACTOR"],
        quota_system=simulation_configuration["THROTTLING_ENABLED"],
        gatekeeper_config=simulation_configuration["GATEKEEPER_CONFIG"])

    return simulation_output


def evaluate_actual_vs_equilibrium(simfunction, input_params, simulation_configuration, empirical_profile=None,
                                   equilibrium_profiles=[], desc="", empirical_output=None):
    if empirical_output is None:
        logger.info("Simulating Empirical Profile for: " + desc)
        apply_strategy_profile(input_params.player_configuration, empirical_profile)
        empirical_output = run_scenario(simfunction, input_params, simulation_configuration)
    else:
        logger.info("The empirical output was already provided. No simulation for empirical profile needed.")

    for index, equilibrium_profile in enumerate(equilibrium_profiles):
        prefix = "TSNE" + str(index) + "-"

        apply_strategy_profile(input_params.player_configuration, equilibrium_profile)
        equilibrium_output = run_scenario(simfunction, input_params, simulation_configuration)

        compare_with_independent_sampling(empirical_output.get_time_ratio_per_priority(simdata.SEVERE_PRIORITY),
                                          equilibrium_output.get_time_ratio_per_priority(simdata.SEVERE_PRIORITY),
                                          first_system_desc=desc + "_TIME_RATIO_EMPIRICAL",
                                          second_system_desc=prefix + desc + "_TIME_RATIO_EQUILIBRIUM")

        compare_with_independent_sampling(empirical_output.get_completed_per_real_priority(simdata.SEVERE_PRIORITY),
                                          equilibrium_output.get_completed_per_real_priority(simdata.SEVERE_PRIORITY),
                                          first_system_desc=desc + "_FIXED_EMPIRICAL",
                                          second_system_desc=prefix + desc + "_FIXED_EQUILIBRIUM")

        compare_with_independent_sampling(empirical_output.get_fixed_ratio_per_priority(simdata.SEVERE_PRIORITY),
                                          equilibrium_output.get_fixed_ratio_per_priority(simdata.SEVERE_PRIORITY),
                                          first_system_desc=desc + "_FIXED_RATIO_EMPIRICAL",
                                          second_system_desc=prefix + desc + "_FIXED_RATIO_EQUILIBRIUM")


def extract_empirical_profile(player_configuration):
    """
    Given a list of players, extracts its strategy configuration.
    :param player_configuration: List of players.
    :return: A dict with strategy configs
    """

    if gtconfig.use_empirical_strategies:
        return {reporter['name']: reporter[simmodel.STRATEGY_KEY].strategy_config for reporter in player_configuration}
    else:
        raise Exception(
            "Cannot extract empirical profile if the flag gtconfig.use_empirical_strategies is not active!!")


def generate_single_strategy_profile(player_configuration, strategy_config):
    """
    Returns a strategy profile with a single strategy
    :return: None
    """

    return {reporter['name']: strategy_config for reporter in player_configuration}


def apply_strategy_profile(player_configuration, strategy_profile):
    """
    Applies a strategy profile to a list of players
    :param player_configuration: List of players.
    :param strategy_profile: Profile to apply
    :return: None
    """

    for reporter in player_configuration:

        strategy_config = strategy_profile[reporter['name']]
        mixed_profile = 'strategy_configs' in strategy_config.keys()

        if not mixed_profile:
            reporter[simmodel.STRATEGY_KEY] = simutils.EmpiricalInflationStrategy(
                strategy_config=strategy_config)
        else:
            reporter[simmodel.STRATEGY_KEY] = simutils.MixedEmpiricalInflationStrategy(
                mixed_strategy_config=strategy_config)


def get_heuristic_strategy_catalog():
    """
    The collection of strategies for our game-theoretic model of bug reporting: It includes two heuristic ones and 5
    found in our dataset
    :return: List of strategy configurations.
    """

    return [empirical_honest, persistent_deflator, regular_deflator, empirical_inflator, occasional_deflator,
            simmodel.HONEST_CONFIG, simmodel.SIMPLE_INFLATE_CONFIG]


def do_unsupervised_prioritization(simulation_configuration, simfunction, input_params, empirical_profile):
    desc = "UNSUPERVISED"
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration["GATEKEEPER_CONFIG"] = None
    equilibrium_profile = generate_single_strategy_profile(input_params.player_configuration,
                                                           simmodel.SIMPLE_INFLATE_CONFIG)
    evaluate_actual_vs_equilibrium(simfunction, input_params, simulation_configuration, empirical_profile,
                                   [equilibrium_profile],
                                   desc)


def do_gatekeeper(simulation_configuration, simfunction, input_params, empirical_profile):
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration['GATEKEEPER_CONFIG'] = penaltyexp.DEFAULT_GATEKEEPER_CONFIG

    desc = "GATEKEEPER_SUCC090"
    success_rate = 0.90
    simulation_configuration["SUCCESS_RATE"] = success_rate
    input_params.catcher_generator.configure(values=[True, False], probabilities=[success_rate, (1 - success_rate)])

    equilibrium_profile = generate_single_strategy_profile(input_params.player_configuration,
                                                           simmodel.SIMPLE_INFLATE_CONFIG)
    evaluate_actual_vs_equilibrium(simfunction, input_params, simulation_configuration, empirical_profile,
                                   [equilibrium_profile],
                                   desc)

    desc = "GATEKEEPER_SUCC100"
    success_rate = 1.00
    simulation_configuration["SUCCESS_RATE"] = success_rate
    input_params.catcher_generator.configure(values=[True, False], probabilities=[success_rate, (1 - success_rate)])

    tsne1_profile = generate_single_strategy_profile(input_params.player_configuration, persistent_deflator)
    tsne2_profile = generate_single_strategy_profile(input_params.player_configuration,
                                                     {'name': desc + "_TSNE2",
                                                      'strategy_configs': get_heuristic_strategy_catalog(),
                                                      'probabilities': [0.22, 0.22, 0.0, 0.0, 0.56, 0.00, 0.0]})
    tsne3_profile = generate_single_strategy_profile(input_params.player_configuration,
                                                     {'name': desc + "_TSNE3",
                                                      'strategy_configs': get_heuristic_strategy_catalog(),
                                                      'probabilities': [0.00, 0.67, 0.0, 0.0, 0.0, 0.33, 0.0]})
    tsne4_profile = generate_single_strategy_profile(input_params.player_configuration, occasional_deflator)
    tsne5_profile = generate_single_strategy_profile(input_params.player_configuration, simmodel.HONEST_CONFIG)

    evaluate_actual_vs_equilibrium(simfunction=simfunction, input_params=input_params,
                                   simulation_configuration=simulation_configuration,
                                   equilibrium_profiles=[tsne1_profile,
                                                         tsne2_profile,
                                                         tsne3_profile,
                                                         tsne4_profile,
                                                         tsne5_profile],
                                   desc=desc, empirical_profile=empirical_profile)


def do_throttling(simulation_configuration, simfunction, input_params, empirical_profile):
    desc = "THROTTLING_INF001"
    simulation_configuration["THROTTLING_ENABLED"] = True
    simulation_configuration["INFLATION_FACTOR"] = 0.01

    tsne1_profile = generate_single_strategy_profile(input_params.player_configuration, empirical_honest)
    tsne2_profile = generate_single_strategy_profile(input_params.player_configuration,
                                                     {'name': desc + "_TSNE2",
                                                      'strategy_configs': get_heuristic_strategy_catalog(),
                                                      'probabilities': [0.62, 0.0, 0.0, 0.0, 0.0, 0.38, 0.0]})
    tsne3_profile = generate_single_strategy_profile(input_params.player_configuration, simmodel.HONEST_CONFIG)

    evaluate_actual_vs_equilibrium(simfunction=simfunction, input_params=input_params,
                                   simulation_configuration=simulation_configuration,
                                   equilibrium_profiles=[tsne1_profile,
                                                         tsne2_profile,
                                                         tsne3_profile],
                                   desc=desc, empirical_profile=empirical_profile)

    desc = "THROTTLING_INF003"
    simulation_configuration["INFLATION_FACTOR"] = 0.03
    equilibrium_profile = generate_single_strategy_profile(input_params.player_configuration, simmodel.HONEST_CONFIG)

    evaluate_actual_vs_equilibrium(simfunction=simfunction, input_params=input_params,
                                   simulation_configuration=simulation_configuration,
                                   equilibrium_profiles=[equilibrium_profile],
                                   desc=desc, empirical_profile=empirical_profile)

    desc = "THROTTLING_INF005"
    simulation_configuration["INFLATION_FACTOR"] = 0.05
    equilibrium_profile = generate_single_strategy_profile(input_params.player_configuration, simmodel.HONEST_CONFIG)

    evaluate_actual_vs_equilibrium(simfunction=simfunction, input_params=input_params,
                                   simulation_configuration=simulation_configuration,
                                   equilibrium_profiles=[equilibrium_profile],
                                   desc=desc, empirical_profile=empirical_profile)


def main():
    logger.info("Loading information from " + simdata.ALL_ISSUES_CSV)
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    logger.info("Adding calculated fields...")
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    all_valid_projects = simdriver.get_valid_projects(enhanced_dataframe)

    simulation_configuration = dict(payoffgetter.DEFAULT_CONFIGURATION)
    simulation_configuration['REPLICATIONS_PER_PROFILE'] = gtconfig.replications_per_profile
    simulation_configuration['EMPIRICAL_STRATEGIES'] = gtconfig.use_empirical_strategies
    simulation_configuration['N_CLUSTERS'] = 5
    simulation_configuration['PROJECT_FILTER'] = None
    simulation_configuration['SYMMETRIC'] = False
    simulation_configuration['TWINS_REDUCTION'] = False

    input_params = payoffgetter.prepare_simulation_inputs(enhanced_dataframe, all_valid_projects,
                                                          simulation_configuration)

    simfunction = simutils.launch_simulation_parallel
    if not gtconfig.parallel:
        logger.info("PARALLEL EXECUTION: Has been disabled.")
        simfunction = simutils.launch_simulation

    empirical_profile = extract_empirical_profile(input_params.player_configuration)

    do_unsupervised_prioritization(simulation_configuration, simfunction, input_params, empirical_profile)
    do_throttling(simulation_configuration, simfunction, input_params, empirical_profile)
    do_gatekeeper(simulation_configuration, simfunction, input_params, empirical_profile)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    logger.info("Execution time in seconds: " + str(time.time() - start_time))
