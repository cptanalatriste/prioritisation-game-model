"""
This modules triggers the bug report simulation. Launch this module to trigger the simulation validation per
project dataset.
"""
import time
import traceback

from scipy import stats
import numpy as np

import pandas as pd

import analytics
import defaultabuse
import simdata
import simvalid
import simutils
import siminput

import gtconfig

if gtconfig.is_windows:
    import winsound

DEBUG = False

TARGET_FIXES = 10
DIFFERENCE = 2

# According to  Modelling and Simulation Fundamentals by J. Sokolowski (Chapter 2)
# Also, it was found in Discrete Event Simulation by George Fishman (Chapter 100
MINIMUM_P_VALUE = 0.05

from collections import defaultdict


def get_reporter_groups(bug_dataset):
    """
    Given a bug dataset, it returns a list of diferent bug reporters in order of activity.
    :param bug_dataset: Bug report dataset.
    :return: List of reporters, sorted by activity.
    """
    issues_by_tester = bug_dataset[simdata.REPORTER_COLUMN].value_counts()
    testers_in_order = [index for index, _ in issues_by_tester.iteritems()]
    tester_groups = [[tester] for tester in testers_in_order]

    return tester_groups


def get_reporter_configuration(training_dataset, tester_groups=None, drive_by_filter=True,
                               debug=False):
    """
    Returns the reporting information required for the simulation to run.

    Includes drive-in tester removal.

    :param training_dataset: Bug report data frame
    :return: List containing reporter information.
    """

    reporters_config = []
    period_start = training_dataset[simdata.CREATED_DATE_COLUMN].min()

    if tester_groups is None or len(tester_groups) == 0:
        tester_groups = get_reporter_groups(training_dataset)

    batching = gtconfig.report_stream_batching

    if batching:
        print "Report stream: The bug report arrival will be batched using a window."
    else:
        print "Report Stream: No batching is made for the bug arrival."

    for index, reporter_list in enumerate(tester_groups):

        bug_reports = simdata.filter_by_reporter(training_dataset, reporter_list)
        reports = len(bug_reports.index)

        if batching:
            batches = simdata.get_report_batches(bug_reports)
            arrival_times = [batch["batch_head"] for batch in batches]

            batch_sizes_sample = [batch["batch_count"] for batch in batches]
            sample_as_observations = pd.Series(data=batch_sizes_sample)

            batch_size_gen = simutils.DiscreteEmpiricalDistribution(name="batch_dist",
                                                                    observations=sample_as_observations,
                                                                    inverse_cdf=True)
        else:
            report_dates = bug_reports[simdata.CREATED_DATE_COLUMN]
            arrival_times = report_dates.sort_values().values
            sample_as_observations = 1

            batch_size_gen = simutils.ConstantGenerator(name="batch_dist", value=sample_as_observations)

        inter_arrival_sample = simdata.get_interarrival_times(arrival_times, period_start)

        try:

            inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(observations=inter_arrival_sample)
            reporter_name = "Consolidated Testers (" + str(len(reporter_list)) + ")"
            if len(reporter_list) == 1:
                reporter_name = reporter_list[0]

            priority_distribution = simutils.DiscreteEmpiricalDistribution(
                observations=bug_reports[simdata.SIMPLE_PRIORITY_COLUMN])
            priority_map = priority_distribution.get_probabilities()

            reports_per_priority = {index: value for index, value in
                                    bug_reports[simdata.SIMPLE_PRIORITY_COLUMN].value_counts().iteritems()}
            reports_per_priority = defaultdict(int, reports_per_priority)

            modified_priority = simdata.get_modified_priority_bugs(bug_reports)
            with_modified_priority = len(modified_priority.index)

            inflation_records = {}
            for priority in simdata.SUPPORTED_PRIORITIES:
                bugs = modified_priority[modified_priority[simdata.SIMPLE_PRIORITY_COLUMN] == priority]
                true_reports = bugs[
                    bugs[simdata.SIMPLE_PRIORITY_COLUMN] == bugs[simdata.ORIGINAL_SIMPLE_PRIORITY_COLUMN]]
                inflated_reports = bugs[
                    bugs[simdata.SIMPLE_PRIORITY_COLUMN] != bugs[simdata.ORIGINAL_SIMPLE_PRIORITY_COLUMN]]

                inflation_records["priority_" + str(priority) + "_true"] = len(true_reports.index)
                inflation_records["priority_" + str(priority) + "_false"] = len(inflated_reports.index)

            if debug:
                print "Reports made ", reports, "Interrival-time for tester ", reporter_name, " mean: ", np.mean(
                    inter_arrival_sample), " std: ", np.std(
                    inter_arrival_sample), "Batch-size", reporter_name, " mean: ", np.mean(
                    batch_sizes_sample), " std: ", np.std(
                    batch_sizes_sample), " priority_map ", priority_map, " with_modified_priority ", with_modified_priority

            config = {'name': reporter_name,
                      'interarrival_time_gen': inter_arrival_time_gen,
                      'inter_arrival_sample': inter_arrival_sample,
                      'batch_size_gen': batch_size_gen,
                      'batch_size_sample': sample_as_observations,
                      'reporter_list': reporter_list,
                      'reports_per_priority': reports_per_priority,
                      'with_modified_priority': with_modified_priority,
                      'modified_details': inflation_records,
                      'reports': reports}

            reporters_config.append(config)

        except ValueError as _:
            if debug:
                print "Reporters ", reporter_list, " could not be added. Possible because insufficient samples."

    if drive_by_filter:
        original_reporters = len(reporters_config)
        reporters_config = simutils.remove_drive_in_testers(reporters_config, min_reports=10)
        print "Original reporters: ", original_reporters, "Number of reporters after drive-by filtering: ", len(
            reporters_config)
    else:
        print "No drive-by filtering was performed."

    return reporters_config


def fit_reporter_distributions(reporters_config):
    """
    Fits theoretical probability distributions  used for modelling reporter behaviour.
    :param reporters_config: List of basic reporter configurations.
    :return: None.
    """
    for config in reporters_config:
        inter_arrival_sample = config['inter_arrival_sample']
        print "Fitting distribution according to the current sample: ", inter_arrival_sample.describe()

        reporter_list = config['name']

        description = "INTERRIVAL_TIME_" + str(reporter_list)
        file_name = "csv/" + description + ".csv"
        inter_arrival_sample.to_csv(file_name)
        print "Inter-arrival samples stored in ", file_name

        best_fit = siminput.launch_input_analysis(inter_arrival_sample, description,
                                                  show_data_plot=False, save_plot=False)
        inter_arrival_time_gen = None

        if best_fit["ks_p_value"] >= MINIMUM_P_VALUE:
            print "Using ", best_fit["dist_name"], " for Tester ", str(
                reporter_list), " Interarrival time with parameters ", best_fit["parameters"], " with p-value ", \
                best_fit["ks_p_value"]
            inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(distribution=best_fit["distribution"],
                                                                              parameters=best_fit["parameters"],
                                                                              observations=inter_arrival_sample)
        elif len(inter_arrival_sample.index) >= simutils.MINIMUM_OBSERVATIONS:
            print "Using an Empirical Distribution for Tester ", str(reporter_list), " Interarrival time"
            inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(observations=inter_arrival_sample)

        config['interarrival_time_gen'] = inter_arrival_time_gen


def get_reporting_metrics(reported_dataset, resolved_dataset, reporters_config):
    """
    Gathers data metrics from a list of resolved issues.
    :param resolved_dataset: Dataframe with resolved issues.
    :return: Map with the resolved metrics.
    """

    first_report = reported_dataset[simdata.CREATED_DATE_COLUMN].min()
    last_report = reported_dataset[simdata.CREATED_DATE_COLUMN].max()

    reporting_time = ((last_report - first_report).total_seconds()) / simdata.TIME_FACTOR

    resolution_metrics = {"results_per_priority": [],
                          "results_per_reporter": [],
                          'true_resolved': len(resolved_dataset.index),
                          'reporting_time': reporting_time}

    for priority in simdata.SUPPORTED_PRIORITIES:
        resolved_per_priority = resolved_dataset[resolved_dataset[simdata.SIMPLE_PRIORITY_COLUMN] == priority]
        reported_per_priority = reported_dataset[reported_dataset[simdata.SIMPLE_PRIORITY_COLUMN] == priority]

        resolution_metrics['results_per_priority'].append({'priority': priority,
                                                           'true_resolved': len(resolved_per_priority.index),
                                                           'true_reported': len(reported_per_priority.index)})

    for reporter_config in reporters_config:
        reporter_name = reporter_config['name']
        true_resolved = simdata.filter_by_reporter(resolved_dataset, reporter_config['reporter_list'])
        true_reported = simdata.filter_by_reporter(reported_dataset, reporter_config['reporter_list'])

        resolution_metrics["results_per_reporter"].append({"reporter_name": reporter_name,
                                                           "true_resolved": len(true_resolved.index),
                                                           'true_reported': len(true_reported.index)})

    return resolution_metrics


def consolidate_results(year_month, issues_for_period, resolved_in_month, reporters_config, completed_per_reporter,
                        completed_per_priority, reports_per_priority, reporting_times, project_keys,
                        debug=False):
    """
    It consolidates the results from the simulation with the information contained in the data.

    :param debug: Detailed output messages
    :param year_month: Period identifier.
    :param issues_for_period: Issues reported on the same period of report.
    :param resolved_in_month: Issues resolved on the same period of report.
    :param reporters_config:   Reporter configuration.
    :param completed_per_reporter: Simulation results per reporter.
    :param completed_per_priority: Simulation results per priority.
    :return:
    """

    resolution_metrics = None
    true_resolved = None
    if issues_for_period is not None and resolved_in_month is not None:
        resolution_metrics = get_reporting_metrics(issues_for_period, resolved_in_month, reporters_config)
        true_resolved = resolution_metrics['true_resolved']

    simulation_result = {"period": year_month,
                         "results_per_reporter": [],
                         "results_per_priority": [],
                         "true_resolved": true_resolved}
    results = []
    for report in completed_per_reporter:
        total_resolved = 0
        for reporter_config in reporters_config:
            total_resolved += report[reporter_config['name']]
        results.append(total_resolved)

    simulation_result["resolved_samples"] = results
    simulation_result["reporting_times_samples"] = reporting_times
    simulation_result["predicted_resolved"] = np.mean(results)

    # TODO: This reporter/priority logic can be refactored.

    simulation_details = {}

    for priority in simdata.SUPPORTED_PRIORITIES:

        true_resolved = None
        true_reported = None
        if resolution_metrics is not None:
            true_results = [result for result in resolution_metrics['results_per_priority'] if
                            result['priority'] == priority][0]

            true_resolved = true_results['true_resolved']
            true_reported = true_results['true_reported']

        resolved_on_simulation = [report[priority] for report in completed_per_priority]
        predicted_resolved = np.mean(resolved_on_simulation)

        reported_on_simulation = [report[priority] for report in reports_per_priority]
        predicted_reported = np.mean(reported_on_simulation)

        simulation_result['results_per_priority'].append({'priority': priority,
                                                          'true_resolved': true_resolved,
                                                          'true_reported': true_reported,
                                                          'predicted_resolved': predicted_resolved,
                                                          'resolved_samples': resolved_on_simulation,
                                                          'predicted_reported': predicted_reported})

        simulation_details["Resolved_Pri_" + str(priority)] = resolved_on_simulation
        simulation_details["Reported_Pri_" + str(priority)] = reported_on_simulation

    details_dataframe = pd.DataFrame(data=simulation_details)
    filename = "csv/" + "_".join(project_keys) + "_sim_details.csv"
    details_dataframe.to_csv(filename)
    print "Simulation results by priority are stored in ", filename

    for reporter_config in reporters_config:
        reporter_name = reporter_config['name']

        true_resolved = None
        true_reported = None

        if resolution_metrics is not None:
            true_results = [result for result in resolution_metrics['results_per_reporter'] if
                            result['reporter_name'] == reporter_name][0]

            true_resolved = true_results['true_resolved']
            true_reported = true_results['true_reported']

        resolved_on_simulation = [report[reporter_name] for report in completed_per_reporter]
        predicted_resolved = np.mean(resolved_on_simulation)

        sample_median, sample_std, sample_size = predicted_resolved, np.std(resolved_on_simulation), len(
            resolved_on_simulation)
        alpha = 0.95
        confidence_interval = stats.norm.interval(alpha, loc=sample_median, scale=sample_std / np.sqrt(sample_size))

        if debug:
            print "Reporter ", reporter_name, "sample_median ", sample_median, " sample_std ", sample_std, " confidence interval: ", \
                confidence_interval, " true_resolved ", len(true_resolved.index), " true_reported ", len(
                true_reported.index)

        simulation_result["results_per_reporter"].append({"reporter_name": reporter_name,
                                                          "true_resolved": true_resolved,
                                                          "true_reported": true_reported,
                                                          "predicted_resolved": predicted_resolved})

    if debug:
        print "simulation_result ", simulation_result

    return simulation_result


def get_resolution_time_gen(resolved_issues, desc=""):
    """
    Generates a sample generator for resolution time.
    :param resolution_time_sample:
    :param desc: Description of the sample
    :return: Resolution time generator.
    """

    resolution_time_sample = resolved_issues[simdata.RESOLUTION_TIME_COLUMN].dropna()

    print "Resolution times in Training Range for ", desc, ": \n", resolution_time_sample.describe()

    description = "RESOL_TIME_" + desc

    file_name = "csv/" + description + ".csv"
    resolution_time_sample.to_csv(file_name)
    print "Resolution time samples stored in ", file_name

    best_fit = siminput.launch_input_analysis(resolution_time_sample, description,
                                              show_data_plot=False, save_plot=False)
    resolution_time_gen = None

    # According to  Modelling and Simulation Fundamentals by J. Sokolowski (Chapter 2 - Page 46)
    if best_fit["ks_p_value"] >= MINIMUM_P_VALUE:
        print "Using ", best_fit["dist_name"], " for ", desc, " Resolution Time with parameters ", \
            best_fit["parameters"], " with p-value ", best_fit["ks_p_value"]
        resolution_time_gen = simutils.ContinuousEmpiricalDistribution(distribution=best_fit["distribution"],
                                                                       parameters=best_fit["parameters"],
                                                                       observations=resolution_time_sample)
    elif len(resolution_time_sample.index) >= simutils.MINIMUM_OBSERVATIONS:
        print "Using an Empirical Distribution for ", desc, " Resolution Time"
        resolution_time_gen = simutils.ContinuousEmpiricalDistribution(observations=resolution_time_sample)

    return resolution_time_gen


def get_priority_change_gen(training_issues):
    """
    Gets a variate generator for the time for a priority to get corrected. In hours.
    :param training_issues: Dataframe with bug reports.
    :return: An random variate generator.
    """
    with_changed_priority = simdata.get_modified_priority_bugs(training_issues)
    change_time_sample = with_changed_priority[simdata.PRIORITY_CHANGE_TIME_COLUMN].dropna()

    print "Priority changes per project:  \n", with_changed_priority[simdata.PROJECT_KEY_COUMN].value_counts()
    print "Priority change times in Training Range : \n", change_time_sample.describe()

    description = "PRIORITY_CHANGE"
    file_name = "csv/" + description + ".csv"
    change_time_sample.to_csv(file_name)
    print "Priority change samples stored in ", file_name

    best_fit = siminput.launch_input_analysis(change_time_sample, description,
                                              show_data_plot=False, save_plot=False)

    change_time_gen = None
    # According to  Modelling and Simulation Fundamentals by J. Sokolowski (Chapter 2 - Page 46)
    if best_fit["ks_p_value"] >= MINIMUM_P_VALUE:
        print "Using ", best_fit["dist_name"], " for Priority Change Time with parameters", best_fit[
            "parameters"], " with p-value ", best_fit["ks_p_value"]
        change_time_gen = simutils.ContinuousEmpiricalDistribution(distribution=best_fit["distribution"],
                                                                   parameters=best_fit["parameters"],
                                                                   observations=change_time_sample)
    elif len(change_time_sample.index) >= simutils.MINIMUM_OBSERVATIONS:
        print "Using an Empirical Distribution for Priority Change Time"
        change_time_gen = simutils.ContinuousEmpiricalDistribution(observations=change_time_sample)

    return change_time_gen


def get_simulation_input(training_issues=None):
    """
    Extract the simulation paramaters from the training dataset.
    :param training_issues: Training data set.
    :return: Variate generator for resolution times, priorities and reporter inter-arrival time.
    """

    priority_sample = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    counts_per_priority = priority_sample.value_counts()
    print "Simplified Priorities in Training Range: \n ", counts_per_priority

    resolution_per_priority = defaultdict(lambda: None)
    all_resolved_issues = simdata.filter_resolved(training_issues, only_with_commits=True,
                                                  only_valid_resolution=True)

    all_ignored_issues = training_issues[training_issues[simdata.STATUS_COLUMN].isin(['Open'])]
    ignored_per_priority = defaultdict(lambda: None)

    total_ignored = float(len(all_ignored_issues.index))
    print "Total number of ignored reports: ", total_ignored

    most_relevant_priority = None
    most_relevant_probability = None

    for priority in priority_sample.unique():
        if not np.isnan(priority):
            priority_resolved = all_resolved_issues[all_resolved_issues[simdata.SIMPLE_PRIORITY_COLUMN] == priority]
            resolution_time_gen = get_resolution_time_gen(priority_resolved, desc="Priority_" + str(priority))
            resolution_per_priority[priority] = resolution_time_gen

            priority_ignored = all_ignored_issues[all_ignored_issues[simdata.SIMPLE_PRIORITY_COLUMN] == priority]
            priority_reported = training_issues[training_issues[simdata.SIMPLE_PRIORITY_COLUMN] == priority]

            ignored_probability = 0.0

            total_reported = float(len(priority_reported.index))
            if total_reported > 0:
                ignored_probability = len(priority_ignored.index) / total_reported

            print "Ignored probability for Priority ", priority, " is ", ignored_probability, ". Ignored reports: ", len(
                priority_ignored.index)

            if most_relevant_priority is None or ignored_probability < most_relevant_probability:
                most_relevant_priority = priority
                most_relevant_probability = ignored_probability

            ignored_per_priority[priority] = simutils.DiscreteEmpiricalDistribution(name="Ignored_" + str(priority),
                                                                                    values=[True, False],
                                                                                    probabilities=[ignored_probability,
                                                                                                   (
                                                                                                       1 - ignored_probability)])

    print "MOST RELEVANT PRIORITY: ", most_relevant_priority
    priorities_in_training = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    priority_generator = simutils.DiscreteEmpiricalDistribution(observations=priorities_in_training)
    print "Training Priority Map : ", priority_generator.get_probabilities()

    return resolution_per_priority, ignored_per_priority, priority_generator


def get_valid_reports(project_keys, enhanced_dataframe, exclude_priority=None):
    """
    Returns the issues valid for simulation analysis. It includes:

    - Filtered by project
    - Excluding self-fixes

    :param project_keys: Project identifiers.
    :param enhanced_dataframe: Bug report dataframe.
    :return: Filtered dataframe
    """
    print "Starting analysis for projects ", project_keys, " ..."

    project_bugs = simdata.filter_by_project(enhanced_dataframe, project_keys)
    print "Total issues for projects ", project_keys, ": ", len(project_bugs.index)

    project_bugs = simdata.exclude_self_fixes(project_bugs)
    print "After self-fix exclusion: ", project_keys, ": ", len(project_bugs.index)

    project_reporters = project_bugs[simdata.REPORTER_COLUMN].value_counts()
    print "Total Reporters: ", len(project_reporters.index)

    if exclude_priority is not None:
        project_bugs = project_bugs[project_bugs[simdata.SIMPLE_PRIORITY_COLUMN] != exclude_priority]
        print "After Priority exclusion: ", exclude_priority, project_keys, ": ", len(project_bugs.index)

    return project_bugs


def is_valid_period(issues_for_period, batch=-1):
    """
    Determines the rule for launching the simulation for that period.
    :param issues_for_period: Total issues in the period.
    :param resolved_in_period: Resolved issues in the period.
    :return: True if valid for simulation. False otherwise.
    """

    tolerance = 0.0
    resolved_issues = issues_for_period[issues_for_period[simdata.RESOLVED_IN_BATCH_COLUMN]]
    result = abs(len(resolved_issues.index) - TARGET_FIXES) <= tolerance

    if not result:
        print "The invalid period only has ", len(resolved_issues.index), " fixes in a batch of ", len(
            issues_for_period.index), " Identifier: ", batch

    return result


def get_dev_team_production(issues_for_period, debug=False):
    """
    Returns the production of the development team for a specific period.
    :return: Developer Team Size and Developer Team Production.
    """

    if simdata.RESOLVED_IN_BATCH_COLUMN in issues_for_period.columns:
        resolved_in_period = issues_for_period[issues_for_period[simdata.RESOLVED_IN_BATCH_COLUMN]]
    else:
        print "No resolution in batch information found. Considering all resolved"
        resolved_in_period = simdata.filter_resolved(issues_for_period, only_with_commits=True,
                                                     only_valid_resolution=True)

    if debug:
        print "Developer productivity: ", len(resolved_in_period.index), " issues resolved from ", len(
            issues_for_period.index), " reports"

    bug_resolvers = resolved_in_period['JIRA Resolved By']
    dev_team_size = bug_resolvers.nunique()
    issues_resolved = len(resolved_in_period.index)
    dev_team_bandwith = resolved_in_period[simdata.RESOLUTION_TIME_COLUMN]

    dev_team_bandwith = dev_team_bandwith.sum()
    return dev_team_size, issues_resolved, resolved_in_period, dev_team_bandwith


def get_team_training_data(training_issues, reporters_config):
    """
    Extracts development team information from the training dataset.

    :param training_issues: Dataframe with the issues for training
    :return: A develoment team size generator, and another one for the bandwith.
    """

    training_in_batches = simdata.include_batch_information(training_issues, target_fixes=TARGET_FIXES)
    dev_team_sizes = []
    dev_team_bandwiths = []

    unique_batches = training_in_batches[simdata.BATCH_COLUMN].unique()
    print len(training_in_batches.index), " training issues where grouped in ", len(
        unique_batches), " batches with ", TARGET_FIXES, " fixed reports ..."

    metrics_on_training = []

    excluded_counter = 0
    for train_batch in unique_batches:

        issues_for_batch = training_in_batches[training_in_batches[simdata.BATCH_COLUMN] == train_batch]
        if is_valid_period(issues_for_batch, train_batch):
            dev_team_size, _, resolved_batch, dev_team_bandwith = get_dev_team_production(issues_for_batch)

            dev_team_sizes.append(dev_team_size)
            dev_team_bandwiths.append(dev_team_bandwith)

            reporting_metrics = get_reporting_metrics(issues_for_batch, resolved_batch, reporters_config)
            metrics_on_training.append(reporting_metrics)
        else:
            excluded_counter += 1

    print excluded_counter, " batches were excluded from a total of ", len(unique_batches)

    dev_team_series = pd.Series(data=dev_team_sizes)
    dev_bandwith_series = pd.Series(data=dev_team_bandwiths)

    # TODO: Maybe fit theoretical distributions?

    print "Training - Development Team Size: ", dev_team_series.describe()
    print "Training - Development Team Bandwith: ", dev_bandwith_series.describe()

    return dev_team_series, dev_bandwith_series, metrics_on_training


def get_reporter_generator(reporters_config, symmetric=False):
    """
    Generates a probability distribution for bug reporters.
    :param reporters_config: List with reporter behaviour information.
    :return: A DiscreteEmpiricalDistribution instance
    """
    report_values = [config['reports'] for config in reporters_config]
    total_reports = float(sum(report_values))
    probability_values = [reports / total_reports for reports in report_values]

    if symmetric:
        print "THIS IS A SYMMETRIC GENERATOR: All reporter's have the same probability."
        probability_values = [1.0 / len(reporters_config) for _ in reporters_config]

    reporter_gen = simutils.DiscreteEmpiricalDistribution(name="Reporter_Generator",
                                                          values=[config['name'] for config in reporters_config],
                                                          probabilities=probability_values)
    return reporter_gen


def get_report_stream_params(training_issues, reporters_config, symmetric=False):
    """
    Returns the generators required for the bug report stream in the simulation.
    :param reporters_config: Reporter information.
    :return: A generator for reporter, for batch sizes and time between batches.
    """
    print "Getting global reporting information ..."

    reporter_gen = get_reporter_generator(reporters_config, symmetric=symmetric)

    all_reporters = [config['name'] for config in reporters_config]
    global_reporter_config = get_reporter_configuration(training_issues, [all_reporters], drive_by_filter=False)
    fit_reporter_distributions(global_reporter_config)

    batch_size_gen = global_reporter_config[0]['batch_size_gen']
    if gtconfig.report_stream_batching:
        print "Current batch size information: \n", global_reporter_config[0]['batch_size_sample'].describe()

    interarrival_time_gen = global_reporter_config[0]['interarrival_time_gen']

    return reporter_gen, batch_size_gen, interarrival_time_gen


def train_validate_simulation(project_key, issues_in_range, max_iterations, keys_train, keys_valid, parallel=True,
                              prefix=""):
    """

    Train the simulation model on a dataset and test it in another dataset.

    :param fold: Identifier of the train-test period.
    :param max_iterations: Iterations for the simulation.
    :param issues_in_range: Bug report dataframe.
    :param project_key:List of projects key.
    :param keys_train:Issues in training dataset.
    :param keys_valid: Issues in the validation dataset.
    :return: Consolidated simulation results.
    """

    simulate_func = simutils.launch_simulation_parallel
    if not parallel:
        print "Project ", project_key, ": Disabling parallel execution ..."
        simulate_func = simutils.launch_simulation

    training_issues = issues_in_range[issues_in_range[simdata.ISSUE_KEY_COLUMN].isin(keys_train)]
    print "Issues in training: ", len(training_issues.index)

    reporters_config = get_reporter_configuration(training_issues, drive_by_filter=True)
    if len(reporters_config) == 0:
        print "Project ", project_key, ": No reporters left on training dataset after drive-by filtering."
        return None

    try:
        simutils.assign_strategies(reporters_config, training_issues)
    except ValueError as e:
        print "Cannot perform strategy assignment for this project..."

    engaged_testers = [reporter_config['name'] for reporter_config in reporters_config]
    training_issues = simdata.filter_by_reporter(training_issues, engaged_testers)
    print "Issues in training after reporter filtering: ", len(training_issues.index)

    resolution_time_gen, ignored_gen, priority_generator = get_simulation_input(training_issues)
    if resolution_time_gen is None:
        print "Not enough resolution time info! ", project_key
        return

    valid_issues = issues_in_range[issues_in_range[simdata.ISSUE_KEY_COLUMN].isin(keys_valid)]
    print "Issues in Validation: ", len(valid_issues.index)
    valid_issues = simdata.filter_by_reporter(valid_issues, engaged_testers)
    print "Issues in validation after reporter filtering: ", len(valid_issues.index)
    print "Assigning Batch information to validation dataset ..."

    valid_issues = simdata.include_batch_information(valid_issues, target_fixes=TARGET_FIXES)

    unique_batches = valid_issues[simdata.BATCH_COLUMN].unique()
    print len(valid_issues.index), " reports where grouped in ", len(
        unique_batches), " batches with ", TARGET_FIXES, " fixed reports ..."

    dev_team_series, dev_bandwith_series, training_metrics = get_team_training_data(training_issues,
                                                                                    reporters_config)

    print "Development Team Size sample: \n", dev_team_series.describe(),
    dev_size_generator = simutils.DiscreteEmpiricalDistribution(observations=dev_team_series, inverse_cdf=True)

    reporter_gen, batch_size_gen, interarrival_time_gen = get_report_stream_params(training_issues, reporters_config)

    simulation_output = simulate_func(
        reporters_config=reporters_config,
        resolution_time_gen=resolution_time_gen,
        batch_size_gen=batch_size_gen,
        interarrival_time_gen=interarrival_time_gen,
        ignored_gen=ignored_gen,
        reporter_gen=reporter_gen,
        max_iterations=max_iterations,
        priority_generator=priority_generator,
        target_fixes=TARGET_FIXES,
        team_capacity=None,
        dev_size_generator=dev_size_generator)

    simulation_result = consolidate_results("SIMULATION", None, None,
                                            reporters_config,
                                            simulation_output["completed_per_reporter"],
                                            simulation_output["completed_per_priority"],
                                            simulation_output["reported_per_priotity"],
                                            simulation_output["reporting_times"],
                                            project_key)

    print "Project ", project_key, " - Assessing simulation on TRAINING DATASET: "
    training_results = pd.DataFrame(
        simvalid.analyse_input_output(training_metrics, simulation_result, prefix=prefix + "_TRAINING",
                                      difference=DIFFERENCE))
    training_results.to_csv("csv/" + prefix + "_training_val_results.csv")

    metrics_on_validation = []

    excluded_counter = 0
    for valid_period in unique_batches:
        issues_for_period = valid_issues[valid_issues[simdata.BATCH_COLUMN] == valid_period]

        _, issues_resolved, resolved_in_period, dev_team_bandwith = get_dev_team_production(
            issues_for_period)

        if is_valid_period(issues_for_period, valid_period):
            reporting_metrics = get_reporting_metrics(issues_for_period, resolved_in_period, reporters_config)
            metrics_on_validation.append(reporting_metrics)
        else:
            excluded_counter += 1

    print  excluded_counter, " batches where excluded from a total of ", len(unique_batches)

    return metrics_on_validation, simulation_result, training_results


def split_dataset(dataframe, set_size):
    """
    Splits a dataframe in two sets.

    :param dataframe: Dataframe to split.
    :param set_size: Size of the set located at the end. It is a number between 0 and 1.
    :return: The dataframe split in two sets.
    """
    if set_size:
        other_set_size = 1 - set_size
        split_point = int(len(dataframe) * other_set_size)

        set_keys = dataframe[:split_point]
        other_set_keys = dataframe[split_point:]

        return set_keys, other_set_keys

    return None, None


def get_experiment_prefix(project_key, test_size):
    """
    A convenient prefix to identify an experiment instance.
    :param project_key: Projects under analysis.
    :param test_size: Size of the test dataset.
    :return: The prefix.
    """
    return "_".join(project_key) + "_Test_" + str(test_size)


def simulate_project(project_key, enhanced_dataframe, parallel=True, test_size=None, max_iterations=1000):
    """
    Launches simulation analysis for an specific project.
    :param project_key: Project identifier.
    :param enhanced_dataframe: Dataframe with additional fields
    :return: None
    """

    print "Number of issues before valid filtering: ", len(enhanced_dataframe.index)
    print "Number of reporters before valid filtering: ", enhanced_dataframe[simdata.REPORTER_COLUMN].nunique()
    print "Report Start before valid filtering: ", enhanced_dataframe[simdata.CREATED_DATE_COLUMN].min()
    print "Report End before valid filtering: ", enhanced_dataframe[simdata.CREATED_DATE_COLUMN].max()
    print "Number of projects before valid filtering: ", enhanced_dataframe[simdata.PROJECT_KEY_COUMN].nunique()

    issues_in_range = get_valid_reports(project_key, enhanced_dataframe, exclude_priority=None)
    issues_in_range = issues_in_range.sort_values(by=simdata.CREATED_DATE_COLUMN, ascending=True)

    analytics.run_project_analysis(project_key, issues_in_range)

    keys_in_range = issues_in_range[simdata.ISSUE_KEY_COLUMN].unique()
    print "Number of issue keys after valid filtering: ", len(keys_in_range)

    simulation_results = []

    experiment_prefix = get_experiment_prefix(project_key, test_size)

    if test_size is not None:
        keys_train, keys_test = split_dataset(keys_in_range, test_size)
        keys_train, keys_valid = split_dataset(keys_train, test_size)

        print "Training simulation and validating: Keys in Train: ", len(
            keys_train), "Keys in Validation ", len(keys_valid), " Keys in Test: ", len(
            keys_test), " Test Size: ", test_size, " Simulation iterations: ", max_iterations

        training_output = train_validate_simulation(project_key, issues_in_range,
                                                    max_iterations,
                                                    keys_train,
                                                    keys_valid,
                                                    parallel=parallel,
                                                    prefix=experiment_prefix)
        if training_output is None:
            print "TRAINING FAILED for Project ", project_key
            return None

        metrics_on_valid, simulation_result, training_results = training_output
        simulation_results.extend((metrics_on_valid, simulation_result))

    if len(simulation_results) > 0:
        valid_results = pd.DataFrame(
            simvalid.analyse_input_output(simulation_results[0], simulation_results[1],
                                          prefix=experiment_prefix + "_VALIDATION",
                                          difference=DIFFERENCE))
        file_name = "csv/" + experiment_prefix + "_validation_val_results.csv"
        valid_results.to_csv(file_name)

        print "Project ", project_key, " - Assessing simulation on VALIDATION DATASET. Results written in ", file_name

    return training_results, valid_results


def get_valid_projects(enhanced_dataframe, threshold=0.3):
    """
    Selects the projects that will be considered in analysis.
    :param enhanced_dataframe: Bug Report dataframe.
    :return: Project key list.
    """

    project_dataframe = defaultabuse.get_default_usage_data(enhanced_dataframe)

    using_priorities = project_dataframe[project_dataframe['non_default_ratio'] >= threshold]
    project_keys = using_priorities['project_key'].unique()

    return project_keys


def get_simulation_results(project_list, enhanced_dataframe, test_size, max_iterations, parallel):
    """
    Applies the simulation and validation procedures to a project list.
    :param project_list: List of projects.
    :param enhanced_dataframe: Bug report dataframe.
    :param test_size: Percentage of bug reports for testing.
    :param max_iterations:Iterations per simulation.
    :param parallel: True for parallel simulation execution.
    :return:Validation results.
    """
    simulation_output = simulate_project(project_list, enhanced_dataframe,
                                         test_size=test_size,
                                         max_iterations=max_iterations,
                                         parallel=parallel)

    if simulation_output is None:
        return [{'test_size': test_size,
                 'project_list': "_".join(project_list),
                 'meassure': 'ERROR_COULDNT_TRAIN',
                 'simulation_value': 0.0,
                 'training_value': 0.0,
                 'validation_value': 0.0,
                 'accept_simulation_training': False,
                 'accept_simulation_validation': False}]

    training_results, validation_results = simulation_output
    performance_meassures = ['RESOLVED_BUGS_FROM_PRIORITY_1', 'RESOLVED_BUGS_FROM_PRIORITY_3']

    results = []
    for meassure in performance_meassures:
        column_value = get_experiment_prefix(project_list, test_size) + "_TRAINING_" + meassure
        training_series = training_results.loc[training_results['desc'] == column_value].iloc[0]
        simulation_value = training_series['sample_mean']
        training_value = training_series['population_mean']
        accept_simulation_training = training_series['ci_accept_simulation']

        column_value = get_experiment_prefix(project_list, test_size) + "_VALIDATION_" + meassure
        validation_series = validation_results.loc[validation_results['desc'] == column_value].iloc[0]
        validation_value = validation_series['population_mean']
        accept_simulation_validation = validation_series['ci_accept_simulation']

        results.append({'test_size': test_size,
                        'project_list': "_".join(project_list),
                        'meassure': meassure,
                        'simulation_value': simulation_value,
                        'training_value': training_value,
                        'validation_value': validation_value,
                        'accept_simulation_training': accept_simulation_training,
                        'accept_simulation_validation': accept_simulation_validation})

    return results


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    max_iterations = gtconfig.replications_per_profile
    valid_projects = get_valid_projects(enhanced_dataframe, threshold=0.3)
    parallel = gtconfig.parallel
    test_sizes = [.2]
    per_project = False
    consolidated = True

    consolidated_results = []

    try:
        project_name = None

        if consolidated:
            print "Starting consolidated analysis ..."
            project_name = "ALL"
            for test_size in test_sizes:
                consolidated_results += get_simulation_results(project_list=valid_projects,
                                                               max_iterations=max_iterations,
                                                               parallel=parallel,
                                                               test_size=test_size,
                                                               enhanced_dataframe=enhanced_dataframe)

        if per_project:
            print "Starting per-project analysis ..."
            for test_size in test_sizes:
                for project in valid_projects:
                    project_name = project
                    results = get_simulation_results(project_list=[project], max_iterations=max_iterations,
                                                     parallel=parallel,
                                                     test_size=test_size, enhanced_dataframe=enhanced_dataframe)

                    consolidated_results += results

    except:
        print "ERROR!!!!: Could not simulate ", project_name
        traceback.print_exc()

    consolidated_results = [result for result in consolidated_results if result is not None]

    if len(consolidated_results) > 0:
        prefix = ""
        if consolidated:
            prefix += "ALL_"
        if per_project:
            prefix += "PROJECTS_"
        results_dataframe = pd.DataFrame(consolidated_results)
        file_name = "csv/" + prefix + str(TARGET_FIXES) + "_fixes_" + str(
            DIFFERENCE) + "_ci_difference_validation.csv"
        results_dataframe.to_csv(file_name)
        print "Consolidated validation results written to ", file_name


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
