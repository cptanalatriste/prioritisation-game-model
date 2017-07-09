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
import simdata
import simdriver
import simutils

if gtconfig.is_windows:
    import winsound


def compare_with_independent_sampling(first_system_replications, second_system_replications, alpha=0.05):
    """
    Different and independent random number streams will be used to simulate the two systems. We are not assuming that
    the variances are equal.

    :param first_system_replications: Observations for simulated system 1
    :param second_system_replications: Observations for simulated system 2
    :return:
    """

    first_system_mean = np.mean(first_system_replications)
    first_system_variance = np.var(first_system_replications, ddof=1)
    print "System 1: Sample mean ", first_system_mean, " Sample variance: ", first_system_variance

    second_system_mean = np.mean(second_system_replications)
    second_system_variance = np.var(second_system_replications, ddof=1)

    print "System 2: Sample mean ", second_system_mean, " Sample variance: ", second_system_variance

    point_estimate = first_system_mean - second_system_mean
    print "Point estimate: ", point_estimate

    compare_means = sms.CompareMeans(sms.DescrStatsW(data=first_system_replications),
                                     sms.DescrStatsW(data=second_system_replications))
    conf_interval = compare_means.tconfint_diff(usevar="unequal", alpha=alpha)
    print "Confidence Interval with alpha ", alpha, " : ", conf_interval


def test():
    # This data is taken from the book. We use it for testing only
    first_system_replications = [29.59, 23.49, 25.68, 41.09, 33.84, 39.57, 37.04, 40.20, 61.82, 44.00]
    second_system_replications = [51.62, 51.91, 45.27, 30.85, 56.15, 28.82, 41.30, 73.06, 23.00, 28.44]

    compare_with_independent_sampling(first_system_replications, second_system_replications)


def run_scenario(simfunction, input_params, simulation_configuration, quota_system,
                 gatekeeper_config):
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
        quota_system=quota_system,
        gatekeeper_config=gatekeeper_config)

    return simulation_output.get_time_ratio_per_priority(simdata.NON_SEVERE_PRIORITY)


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
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
        print "PARALLEL EXECUTION: Has been disabled."
        simfunction = simutils.launch_simulation

    print "Simulating unsupervised prioritization ..."

    gatekeeper_config = None
    quota_system = False

    print "Simulating unsupervised prioritization ..."
    unsupervised_ratio_samples = run_scenario(simfunction, input_params, simulation_configuration, quota_system,
                                              gatekeeper_config)

    simulation_configuration['INFLATION_FACTOR'] = 0.5
    quota_system = True

    print "Simulating Throttling with an inflatio factor of ", simulation_configuration['INFLATION_FACTOR']
    throttling_ratio_samples = run_scenario(simfunction, input_params, simulation_configuration, quota_system,
                                            gatekeeper_config)

    compare_with_independent_sampling(unsupervised_ratio_samples, throttling_ratio_samples)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
