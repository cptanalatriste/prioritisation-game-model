"""
This module contains the procedures for the validation of the simulation.
"""

import simutils
import simdata

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import power

PLOT = None


def analyse_input_output(metrics_on_test, simulation_result):
    """
    Validate the results applying hypothesis testing, according to Discrete-Event Simulation by Jerry Banks.

    :param metrics_on_test: Testing batches.
    :param simulation_result: Simulation results.
    :return: None
    """

    resolved_bugs = [data['true_resolved'] for data in metrics_on_test]
    resolved_bugs_mean = np.mean(resolved_bugs)
    resolved_bugs_std = np.std(resolved_bugs)

    print "Resolved Bugs data information: len ", len(
        resolved_bugs), " mean ", resolved_bugs_mean, " std ", resolved_bugs_std

    resolved_samples = simulation_result['resolved_samples']
    sample_mean = np.mean(resolved_samples)
    sample_size = len(resolved_samples)
    sample_std = np.std(resolved_samples)
    print "Resolved Bugs samples from simulation: len ", sample_size, " mean ", sample_mean, " std ", sample_std

    alpha = 0.05
    difference = 1.0

    apply_t_test(resolved_samples, resolved_bugs_mean, alpha=alpha, difference=difference)
    analyze_confidence_interval(resolved_samples, resolved_bugs_mean, alpha=alpha, difference=difference)


def apply_t_test(samples, population_mean, alpha=0.05, difference=1.0):
    """
    Applies a t-test, considering test power also.
    :param samples: Samples from simulation.
    :param population_mean: Mean on the test dataset.
    :param alpha: Type 1 risk: Rejecting null when null is true.
    :param difference: Expected difference.
    :return: None
    """
    t_stat, two_tail_prob = stats.ttest_1samp(samples, population_mean)
    print "Two-Sample T-test: t-statistic ", t_stat, " p-value ", two_tail_prob

    sample_mean = np.mean(samples)
    sample_size = len(samples)
    sample_std = np.std(samples)
    df = sample_size - 1

    # FYI: The examples on Discrete-Event Simulation by Jerry Banks were replicated using the following python code.
    threshold = stats.t.ppf(1 - alpha / 2, df)
    print "Critical value of t: ", threshold, " for a level of significance (alpha) ", alpha, " and degrees of freedom ", df

    null_hypothesis = "The mean of the sample (" + str(sample_mean) + ") is equal to the population mean ( " + str(
        population_mean) + ")"

    if abs(t_stat) > threshold:
        print "We REJECT the null hypothesis: ", null_hypothesis
    else:
        print "We CANNOT REJECT the null hypothesis ", null_hypothesis

    effect_size = difference / sample_std
    print "Efect size for a difference of ", difference, ": ", effect_size

    test_power = power.tt_solve_power(effect_size=effect_size, alpha=alpha, nobs=sample_size)
    print "Test power: ", test_power


def analyze_confidence_interval(samples, population_mean, alpha=0.05, difference=1.0):
    """
    Performs an analysis based on confidence intervals

    :param samples: Samples from simulation.
    :param population_mean: Mean on the test dataset.
    :param alpha: Type 1 risk: Rejecting null when null is true.
    :param difference: Expected difference.
    :return: None
    """
    conf_alpha = 1 - alpha

    sample_mean = np.mean(samples)
    sample_size = len(samples)
    sample_sem = stats.sem(samples)

    df = sample_size - 1
    lower_bound, upper_bound = stats.t.interval(alpha=conf_alpha, df=df, loc=sample_mean, scale=sample_sem)
    print "confidence_interval: ( ", lower_bound, ", ", upper_bound, ")"

    one_error = abs(population_mean - lower_bound)
    other_error = abs(population_mean - upper_bound)

    best_case_error = min(one_error, other_error)
    worst_case_error = max(one_error, other_error)

    accept_msg = "Accept simulation. Close enough to be considered valid."
    more_simulation_msg = "Additional simulation replications are necessary until a conclussion can be reached."
    refine_msg = "We need to refine the simulation model :("

    print "Difference: ", difference, " best-case error ", best_case_error, " worst-case error ", worst_case_error
    if lower_bound <= population_mean <= upper_bound:
        if best_case_error > difference or worst_case_error > difference:
            print more_simulation_msg

        if worst_case_error <= difference:
            print accept_msg
    else:
        if best_case_error > difference:
            print refine_msg

        if worst_case_error <= difference:
            print accept_msg

        if best_case_error <= difference < worst_case_error:
            print more_simulation_msg


def analyse_results_regression(name="", reporters_config=None, simulation_results=None, project_key=None, debug=False,
                               plot=PLOT):
    """
    Per each tester, it anaysis how close is simulation to real data.

    We are performing a regression-based validation of the simulation model, as suggested by J. Sokolowski in Modeling and
    Simulation Fundamentals.

    :param reporters_config: Tester configuration.
    :param simulation_results: Result from simulation.
    :return: None
    """

    # TODO: This reporter/priority logic can be refactored.
    if reporters_config:
        for reporter_config in reporters_config:
            reporter_name = reporter_config['name']
            completed_true = []
            completed_predicted = []

            for simulation_result in simulation_results:
                reporter_true = [result['true_resolved'] for result in simulation_result["results_per_reporter"] if
                                 result["reporter_name"] == reporter_name][0]
                completed_true.append(reporter_true)

                reporter_predicted = \
                    [result['predicted_resolved'] for result in simulation_result["results_per_reporter"] if
                     result["reporter_name"] == reporter_name][0]
                completed_predicted.append(reporter_predicted)

                if debug:
                    print "period: ", simulation_result[
                        "period"], " reporter ", reporter_name, " predicted ", reporter_predicted, " true ", reporter_true

            simutils.collect_and_print(project_key, "Tester " + reporter_name, completed_true, completed_predicted)

    total_completed = [result['true_resolved'] for result in simulation_results]
    total_predicted = [result['predicted_resolved'] for result in simulation_results]

    if debug:
        print "total_completed ", total_completed
        print "total_predicted ", total_predicted

    mmre, mdmre = simutils.collect_and_print(project_key, "Total_bugs_resolved-" + name, total_completed,
                                             total_predicted)

    if plot is not None:
        simutils.plot_correlation(total_predicted, total_completed, "_".join(project_key) + "-Total Resolved-" + name,
                                  "Points:{} MMRE:{} MdMRE:{}".format(len(total_predicted), int(mmre), int(mdmre)),
                                  plot)

    for priority in simdata.SUPPORTED_PRIORITIES:
        completed_true = []
        completed_predicted = []

        reported_true = []
        reported_predicted = []

        periods = []

        for simulation_result in simulation_results:
            periods.append(simulation_result['period'])

            priority_resolved_true = [result['true_resolved'] for result in simulation_result['results_per_priority'] if
                                      result['priority'] == priority][0]
            completed_true.append(priority_resolved_true)

            priority_resolved_predicted = \
                [result['predicted_resolved'] for result in simulation_result['results_per_priority']
                 if
                 result['priority'] == priority][0]
            completed_predicted.append(priority_resolved_predicted)

            priority_reported_true = [result['true_reported'] for result in simulation_result['results_per_priority'] if
                                      result['priority'] == priority][0]
            reported_true.append(priority_reported_true)

            priority_reported_predicted = \
                [result['predicted_reported'] for result in simulation_result['results_per_priority']
                 if
                 result['priority'] == priority][0]
            reported_predicted.append(priority_reported_predicted)

        mmre, mdmre = simutils.collect_and_print(project_key, "Priority_" + str(priority) + "-" + name, completed_true,
                                                 completed_predicted)

        priority_dataframe = pd.DataFrame({
            "completed_true": completed_true,
            "completed_predicted": completed_predicted,
            "reported_true": reported_true,
            "reported_predicted": reported_predicted,
            "periods": periods
        })

        priority_dataframe.to_csv("csv/pred_results_" + "_".join(project_key) + "_Priority_" + str(priority) + ".csv",
                                  index=False)

        if debug:
            print " completed_true ", completed_true
            print " completed_predicted ", completed_predicted
            print " reported_true ", reported_true
            print " reported_predicted ", reported_predicted

        if plot is not None:
            simutils.plot_correlation(completed_predicted, completed_true,
                                      "-".join(project_key) + "-Priority " + str(priority) + "-" + name,
                                      "Points:{} MMRE:{} MdMRE:{}".format(len(completed_predicted), int(mmre),
                                                                          int(mdmre)),
                                      plot)
