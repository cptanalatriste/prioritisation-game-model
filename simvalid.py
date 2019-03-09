"""
This module contains the procedures for the validation of the simulation.
"""

import simutils
import simdata

import logging
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import power

import gtconfig

PLOT = None

logger = gtconfig.get_logger("exp_equilibrium_results", "exp_equilibrium_results.txt", level=logging.INFO)


def get_data_priority_value(metrics_on_test, data_item, target_priority):
    """
    Collect a priority metric from the dataset.
    :param metrics_on_test: Metrics on dataset
    :param data_item: Metric name
    :param target_priority: Priority
    :return: List of values from dataset.
    """
    results_per_priority = [data['results_per_priority'] for data in metrics_on_test]

    metric_values = []
    for priority_list in results_per_priority:
        metric_values.append(
            [data[data_item] for data in priority_list if data['priority'] == target_priority][0])

    return metric_values


def get_simulation_priority_value(simulation_result, data_item, target_priority):
    """
    Collect a priority metric from the simulation.
    :param simulation_result: Metrics on simulation
    :param data_item: Metric name
    :param target_priority: Priority
    :return: List of values from simulation.
    """

    resolved_samples = [data[data_item] for data in simulation_result['results_per_priority'] if
                        data['priority'] == target_priority][0]

    return resolved_samples


def analyse_input_output(metrics_on_test, simulation_result, difference=2.0, prefix=""):
    """
    Validate the results applying hypothesis testing, according to Discrete-Event Simulation by Jerry Banks.

    :param metrics_on_test: Testing batches.
    :param simulation_result: Simulation results.
    :return: None
    """

    validation_results = []

    resolved_bugs = [data['true_resolved'] for data in metrics_on_test]
    resolved_samples = simulation_result['resolved_samples']
    desc = prefix + "_" + "RESOLVED_BUGS"
    logger.info("Response variable: " + desc)
    validation_results.append(statistical_validation(resolved_bugs, resolved_samples, desc=desc, difference=difference))

    # TODO: Now this screams refactoring
    reporting_times = [data['reporting_time'] for data in metrics_on_test]
    reporting_times_samples = simulation_result['reporting_times_samples']
    desc = prefix + "_" + "REPORTING_TIME"
    logger.info("Response variable: " + desc)
    validation_results.append(
        statistical_validation(reporting_times, reporting_times_samples, desc=desc, difference=difference))

    resolved_in_data = {}

    for target_priority in simdata.SUPPORTED_PRIORITIES:
        resolved_bugs = get_data_priority_value(metrics_on_test, 'true_resolved', target_priority)
        resolved_in_data['Priority_' + str(target_priority)] = resolved_bugs
        resolved_samples = get_simulation_priority_value(simulation_result, 'resolved_samples', target_priority)
        desc = prefix + "_" + "RESOLVED_BUGS_FROM_PRIORITY_" + str(target_priority)
        logger.info("Response variable: " + desc)

        result = statistical_validation(resolved_bugs, resolved_samples, desc=desc, difference=difference)
        validation_results.append(result)

        time_ratios = get_data_priority_value(metrics_on_test, 'true_time_ratio', target_priority)
        time_ratio_samples = get_simulation_priority_value(simulation_result, 'time_ratio_samples', target_priority)
        desc = prefix + "_" + "TIME_RATIO_FROM_PRIORITY_" + str(target_priority)
        logger.info("Response variable: " + desc)
        ratio_difference = 0.2
        result = statistical_validation(time_ratios, time_ratio_samples, desc=desc, difference=ratio_difference)
        validation_results.append(result)

        fix_ratios = get_data_priority_value(metrics_on_test, 'true_fixed_ratio', target_priority)
        fix_ratio_samples = get_simulation_priority_value(simulation_result, 'fixed_ratio_samples', target_priority)
        desc = prefix + "_" + "FIX_RATIO_FROM_PRIORITY_" + str(target_priority)
        logger.info("Response variable: " + desc)
        result = statistical_validation(fix_ratios, fix_ratio_samples, desc=desc, difference=ratio_difference)
        validation_results.append(result)

    file_name = "csv/" + prefix + "_resolved_in_population.csv"
    pd.DataFrame(resolved_in_data).to_csv(file_name)
    logger.info("Resolution report stored in " + file_name)

    return validation_results


def statistical_validation(population_data, sample_data, alpha=0.05, difference=1.0, desc="", plot=False):
    """
    Triggers the statistical validation procedures: t-test and confidence interval.
    :param population_data: Data points gathered from the system.
    :param sample_data: Data points gathered from the simulation.
    :param alpha: Significance Level.
    :param difference: Difference for obtaining the power of the test.
    :return:
    """

    if plot:
        config = {'title': 'Population: ' + desc,
                  'xlabel': desc,
                  'ylabel': 'counts',
                  'file_name': 'population_' + desc + '.png'}
        simdata.launch_histogram(population_data, config=config)

        config = {'title': 'Sample: ' + desc,
                  'xlabel': desc,
                  'ylabel': 'counts',
                  'file_name': 'sample_' + desc + '.png'}

        simdata.launch_histogram(sample_data, config=config)

    population_mean = np.mean(population_data)
    population_std = np.std(population_data)

    logger.info(desc + ": Population data information: len " + str(len(
        population_data)) + " mean " + str(population_mean) + " std " + str(population_std))

    sample_mean = np.mean(sample_data)
    sample_size = len(sample_data)
    sample_std = np.std(sample_data)
    logger.info(
        desc + ": Samples from simulation: len " + str(sample_size) + " mean " + str(sample_mean) + " std " + str(
            sample_std))

    reject_null, test_power = apply_t_test(sample_data, population_mean, alpha=alpha, difference=difference, desc=desc)
    accept_simulation, more_replications, lower_bound, upper_bound = analyze_confidence_interval(sample_data,
                                                                                                 population_mean,
                                                                                                 alpha=alpha,
                                                                                                 difference=difference,
                                                                                                 desc=desc)

    return {'desc': desc,
            'population_mean': population_mean,
            'sample_mean': sample_mean,
            't_test_reject_null': reject_null,
            't_test_test_power': test_power,
            'ci_accept_simulation': accept_simulation,
            'ci_more_replications': more_replications,
            'ci_lower_bound': lower_bound,
            'ci_upper_bound': upper_bound}


def apply_t_test(samples, population_mean, alpha=0.05, difference=1.0, desc=""):
    """
    Applies a t-test, considering test power also.
    :param samples: Samples from simulation.
    :param population_mean: Mean on the test dataset.
    :param alpha: Type 1 risk: Rejecting null when null is true.
    :param difference: Expected difference.
    :return: None
    """
    t_stat, two_tail_prob = stats.ttest_1samp(samples, population_mean)
    logger.info(desc + ": Two-Sample T-test: t-statistic " + str(t_stat) + " p-value " + str(two_tail_prob))

    sample_mean = np.mean(samples)
    sample_size = len(samples)
    sample_std = np.std(samples)
    df = sample_size - 1

    # FYI: The examples on Discrete-Event Simulation by Jerry Banks were replicated using the following python code.
    threshold = stats.t.ppf(1 - alpha / 2, df)
    logger.info(desc + ": Critical value of t: " + str(threshold) + " for a level of significance (alpha) " + str(
        alpha) + " and degrees of freedom " + str(df))

    null_hypothesis = "The mean of the sample (" + str(sample_mean) + ") is equal to the population mean ( " + str(
        population_mean) + ")"

    reject_null = None
    if abs(t_stat) > threshold:
        reject_null = True
        logger.info(desc + ": We REJECT the null hypothesis: " + null_hypothesis)
    else:
        reject_null = False
        logger.info(desc + ": We CANNOT REJECT the null hypothesis " + null_hypothesis)

    effect_size = difference / sample_std
    logger.info(desc + ": Effect size for a difference of " + str(difference) + ": " + str(effect_size))

    test_power = power.tt_solve_power(effect_size=effect_size, alpha=alpha, nobs=sample_size)
    logger.info(desc + ": Test power: " + str(test_power))

    return reject_null, test_power


def analyze_confidence_interval(samples, population_mean, alpha=0.05, difference=1.0, desc=""):
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
    logger.info(desc + " : confidence_interval: ( " + str(lower_bound) + ", " + str(upper_bound) + ")")

    one_error = abs(population_mean - lower_bound)
    other_error = abs(population_mean - upper_bound)

    best_case_error = min(one_error, other_error)
    worst_case_error = max(one_error, other_error)

    accept_msg = desc + ": Accept simulation. Close enough to be considered valid."
    more_simulation_msg = desc + ": Additional simulation replications are necessary until a conclussion can be reached"
    refine_msg = desc + ": We need to refine the simulation model :("

    logger.info(desc + ": Difference: " + str(difference) + " best-case error " + str(
        best_case_error) + " worst-case error " + str(worst_case_error))

    accept_simulation = False
    more_replications = None

    if lower_bound <= population_mean <= upper_bound:
        if best_case_error > difference or worst_case_error > difference:
            logger.info(more_simulation_msg)
            more_replications = True

        if worst_case_error <= difference:
            logger.info(accept_msg)
            accept_simulation = True
    else:
        if best_case_error > difference:
            logger.info(refine_msg)
            accept_simulation = False

        if worst_case_error <= difference:
            logger.info(accept_msg)
            accept_simulation = True

        if best_case_error <= difference < worst_case_error:
            logger.info(more_simulation_msg)
            more_replications = True

    return accept_simulation, more_replications, lower_bound, upper_bound


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
