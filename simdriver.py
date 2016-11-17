"""
This modules triggers the bug report simulation.
"""
import time
import datetime

from scipy import stats
import numpy as np

import pandas as pd

from sklearn.cross_validation import KFold

import defaultabuse
import simdata
import simvalid
import simutils
import siminput

import winsound

DEBUG = False

# According to  Modelling and Simulation Fundamentals by J. Sokolowski (Chapter 2)
MINIMUM_P_VALUE = 0.05

from collections import defaultdict


def get_period_start(bug_dataset):
    """
    Returns the period start date, for interrival time calculation.
    :param period_identifier: List of periods to consider.
    :param bug_dataset: Bug report dataset.
    :return: Date for the starting period.
    """

    period_start = bug_dataset[simdata.CREATED_DATE_COLUMN].min()
    return period_start


def get_reporter_configuration(training_dataset, debug=False):
    """
    Returns the reporting information required for the simulation to run.

    Includes drive-in tester removal.

    :param training_dataset: Bug report data frame
    :return: List containing reporter information.
    """

    issues_by_tester = training_dataset[simdata.REPORTER_COLUMN].value_counts()
    testers_in_order = [index for index, _ in issues_by_tester.iteritems()]

    reporters_config = []
    period_start = get_period_start(training_dataset)

    for index, reporter_list in enumerate([[tester] for tester in testers_in_order]):
        bug_reports = simdata.filter_by_reporter(training_dataset, reporter_list)

        batches = simdata.get_report_batches(bug_reports)
        arrival_times = [batch["batch_head"] for batch in batches]

        inter_arrival_sample = simdata.get_interarrival_times(arrival_times, period_start)
        batch_sizes_sample = [batch["batch_count"] for batch in batches]

        try:

            inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(observations=inter_arrival_sample)
            batch_size_gen = simutils.DiscreteEmpiricalDistribution(name="batch_dist",
                                                                    observations=pd.Series(data=batch_sizes_sample))

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
                print "Interrival-time for tester ", reporter_name, " mean: ", np.mean(
                    inter_arrival_sample), " std: ", np.std(
                    inter_arrival_sample), "Batch-size", reporter_name, " mean: ", np.mean(
                    batch_sizes_sample), " std: ", np.std(
                    batch_sizes_sample), " priority_map ", priority_map, " with_modified_priority ", with_modified_priority

            config = {'name': reporter_name,
                      'interarrival_time_gen': inter_arrival_time_gen,
                      'batch_size_gen': batch_size_gen,
                      'reporter_list': reporter_list,
                      'reports_per_priority': reports_per_priority,
                      'with_modified_priority': with_modified_priority,
                      'modified_details': inflation_records}

            reporters_config.append(config)

        except ValueError as _:
            if debug:
                print "Reporters ", reporter_list, " could not be added. Possible because insufficient samples."

    original_reporters = len(reporters_config)
    reporters_config = simutils.remove_drive_in_testers(reporters_config, min_reports=10)
    print "Original reporters: ", original_reporters, "Number of reporters after drive-by filtering: ", len(
        reporters_config)

    return reporters_config


def fit_reporter_distributions(reporters_config):
    """
    Fits theoretical probability distributions  used for modelling reporter behaviour.
    :param reporters_config: List of basic reporter configurations.
    :return: None.
    """
    for config in reporters_config:
        inter_arrival_sample = config['interarrival_time_gen'].observations
        reporter_list = config['name']

        best_fit = siminput.launch_input_analysis(inter_arrival_sample, "INTERRIVAL_TIME_" + str(reporter_list),
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
            print "Using an Empirical Distribution for Tester ", str(reporter_list), " Resolution Time"
            inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(observations=inter_arrival_sample)

        config['interarrival_time_gen'] = inter_arrival_time_gen


def get_reporting_metrics(reported_dataset, resolved_dataset, reporters_config):
    """
    Gathers data metrics from a list of resolved issues.
    :param resolved_dataset: Dataframe with resolved issues.
    :return: Map with the resolved metrics.
    """

    resolution_metrics = {"results_per_priority": [],
                          "results_per_reporter": [],
                          'true_resolved': len(resolved_dataset.index)}

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
                        completed_per_priority, reports_per_priority,
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
    simulation_result["predicted_resolved"] = np.mean(results)

    # TODO: This reporter/priority logic can be refactored.
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
                                                          'predicted_reported': predicted_reported})

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


def get_simulation_input(training_issues=None, fold=1):
    """
    Extract the simulation paramaters from the training dataset.
    :param training_issues: Training data set.
    :return: Variate generator for resolution times, priorities and reporter inter-arrival time.
    """

    priority_sample = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    counts_per_priority = priority_sample.value_counts()
    print "Simplified Priorities in Training Range: \n ", counts_per_priority

    resolution_per_priority = defaultdict(lambda: None)
    all_resolved_issues = simdata.filter_resolved(training_issues, only_with_commits=False,
                                                  only_valid_resolution=False)
    for priority in priority_sample.unique():
        if not np.isnan(priority):
            priority_resolved = all_resolved_issues[all_resolved_issues[simdata.SIMPLE_PRIORITY_COLUMN] == priority]
            resolution_time_sample = priority_resolved[simdata.RESOLUTION_TIME_COLUMN].dropna()

            print "Resolution times in Training Range for Priority", priority, ": \n", resolution_time_sample.describe()

            best_fit = siminput.launch_input_analysis(resolution_time_sample, "RESOL_TIME_" + str(fold),
                                                      show_data_plot=False, save_plot=False)
            resolution_time_gen = None

            # According to  Modelling and Simulation Fundamentals by J. Sokolowski (Chapter 2)
            if best_fit["ks_p_value"] >= MINIMUM_P_VALUE:
                print "Using ", best_fit["dist_name"], " for Priority ", priority, " Resolution Time with parameters ", \
                    best_fit["parameters"], " with p-value ", best_fit["ks_p_value"]
                resolution_time_gen = simutils.ContinuousEmpiricalDistribution(distribution=best_fit["distribution"],
                                                                               parameters=best_fit["parameters"],
                                                                               observations=resolution_time_sample)
            elif len(resolution_time_sample.index) >= simutils.MINIMUM_OBSERVATIONS:
                print "Using an Empirical Distribution for Priority ", priority, " Resolution Time"
                resolution_time_gen = simutils.ContinuousEmpiricalDistribution(observations=resolution_time_sample)

            resolution_per_priority[priority] = resolution_time_gen

    return resolution_per_priority


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


def get_continuous_chunks(periods_train):
    inflection_points = []

    last_period = None
    for index, period in enumerate(periods_train):
        if last_period:
            year, month = period.split("-")
            year, month = int(year), int(month)

            previous_year, previous_month = last_period.split("-")
            previous_year, previous_month = int(previous_year), int(previous_month)

            same_year = (year == previous_year) and (month - previous_month == 1)
            different_year = (year - previous_year == 1) and (previous_month == 12) and (month == 1)

            if not same_year and not different_year:
                inflection_points.append(index)
                last_period = None
            else:
                last_period = period
        else:
            last_period = period

    chunks = []
    chunk_start = 0
    for point in inflection_points:
        chunks.append(periods_train[chunk_start: point])
        chunk_start = point

    chunks.append(periods_train[chunk_start:])
    return chunks


def get_resolution_times(issues_for_period):
    """
    From an issues dataframe, it returns the list of resolution times contained, excluding NaN
    :param issues_for_period: Dataframe with issues.
    :return: Resolution time series
    """
    resolved_issues = simdata.filter_resolved(issues_for_period, only_with_commits=False,
                                              only_valid_resolution=False)

    return resolved_issues[simdata.RESOLUTION_TIME_COLUMN].dropna()


def is_valid_period(issues_for_period):
    """
    Determines the rule for launching the simulation for that period.

    :param issues_for_period: Total issues in the period.
    :param resolved_in_period: Resolved issues in the period.
    :return: True if valid for simulation. False otherwise.
    """
    return len(issues_for_period.index) == simdata.BATCH_SIZE


def get_dev_team_production(issues_for_period, time_after_last=None):
    """
    Returns the production of the development team for a specific period.
    :return: Developer Team Size and Developer Team Production.
    """

    print "Retrieving dev team parameters from ", len(issues_for_period), " reports by ", \
        issues_for_period['Reported By'].nunique(), " testers "

    start_date = issues_for_period[simdata.CREATED_DATE_COLUMN].min()

    resolved_issues = simdata.filter_resolved(issues_for_period, only_with_commits=False,
                                              only_valid_resolution=False)

    if time_after_last is None:
        time_after_last = get_resolution_times(issues_for_period).median()

    margin = datetime.timedelta(hours=time_after_last)

    # The commented line is when considering a month as simulation period.
    # period_end = start_date
    period_end = issues_for_period[simdata.CREATED_DATE_COLUMN].max()
    end_date = period_end + margin

    resolved_in_period = simdata.filter_by_date_range(simdata.RESOLUTION_DATE_COLUMN, resolved_issues,
                                                      start_date,
                                                      end_date)

    bug_resolvers = resolved_in_period['JIRA Resolved By']
    dev_team_size = bug_resolvers.nunique()
    issues_resolved = len(resolved_in_period.index)
    dev_team_bandwith = resolved_in_period[simdata.RESOLUTION_TIME_COLUMN]

    dev_team_bandwith = dev_team_bandwith.sum()
    return dev_team_size, issues_resolved, resolved_in_period, dev_team_bandwith


def get_dev_team_training(training_issues, time_after_last):
    """
    Extracts development team information from the training dataset.

    :param training_issues: Dataframe with the issues for training
    :return: A develoment team size generator, and another one for the bandwith.
    """

    training_in_batches = simdata.include_batch_information(training_issues)
    dev_team_sizes = []
    dev_team_bandwiths = []

    unique_batches = training_in_batches[simdata.BATCH_COLUMN].unique()
    print len(training_in_batches.index), " training issues where grouped in ", len(
        unique_batches), " batches of ", simdata.BATCH_SIZE, " reports ..."

    for train_batch in unique_batches:
        issues_for_batch = training_in_batches[training_in_batches[simdata.BATCH_COLUMN] == train_batch]

        dev_team_size, _, _, dev_team_bandwith = get_dev_team_production(issues_for_batch, time_after_last)
        dev_team_sizes.append(dev_team_size)
        dev_team_bandwiths.append(dev_team_bandwith)

    dev_team_series = pd.Series(data=dev_team_sizes)
    dev_bandwith_series = pd.Series(data=dev_team_bandwiths)

    # TODO: Maybe fit theoretical distributions?
    dev_team_size_training = simutils.DiscreteEmpiricalDistribution(observations=dev_team_series)
    dev_team_bandwith_training = simutils.ContinuousEmpiricalDistribution(observations=dev_bandwith_series)

    print "Training - Development Team Size: ", dev_team_series.describe()
    print "Training - Development Team Bandwith: ", dev_bandwith_series.describe()

    return dev_team_size_training, dev_team_bandwith_training


def train_test_simulation(project_key, issues_in_range, max_iterations, keys_train, keys_test, fold=0,
                          debug=False):
    """

    Train the simulation model on a dataset and test it in another dataset.

    :param fold: Identifier of the train-test period.
    :param max_iterations: Iterations for the simulation.
    :param issues_in_range: Bug report dataframe.
    :param project_key:List of projects key.
    :param periods_train:Periods for training.
    :param periods_test: Periods for testing.
    :return: Consolidated simulation results.
    """

    training_issues = issues_in_range[issues_in_range[simdata.ISSUE_KEY_COLUMN].isin(keys_train)]
    print "Issues in training: ", len(training_issues.index)

    reporters_config = get_reporter_configuration(training_issues)

    try:
        simutils.assign_strategies(reporters_config, training_issues)
    except ValueError as e:
        print "Cannot perform strategy assignment for this project..."

    fit_reporter_distributions(reporters_config)

    engaged_testers = [reporter_config['name'] for reporter_config in reporters_config]
    training_issues = simdata.filter_by_reporter(training_issues, engaged_testers)
    print "Issues in training after reporter filtering: ", len(training_issues.index)

    resolution_time_gen = get_simulation_input(training_issues, fold=fold)
    if resolution_time_gen is None:
        print "Not enough resolution time info! ", project_key
        return

    test_issues = issues_in_range[issues_in_range[simdata.ISSUE_KEY_COLUMN].isin(keys_test)]
    print "Issues in test: ", len(test_issues.index)
    test_issues = simdata.filter_by_reporter(test_issues, engaged_testers)
    print "Issues in test after reporter filtering: ", len(test_issues.index)
    print "Assigning Batch information to training dataset ..."
    test_issues = simdata.include_batch_information(test_issues)

    unique_batches = test_issues[simdata.BATCH_COLUMN].unique()
    print len(test_issues.index), " where grouped in ", len(
        unique_batches), " batches of ", simdata.BATCH_SIZE, " reports ..."

    time_after_last = get_resolution_times(training_issues).median()

    priorities_in_training = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    print "Priorities for defects in training: ", priorities_in_training.describe()

    priority_generator = simutils.DiscreteEmpiricalDistribution(observations=priorities_in_training)
    print "Training Priority Map : ", priority_generator.get_probabilities()

    dev_team_size_training, dev_team_bandwith_training = get_dev_team_training(training_issues, time_after_last)

    simulation_output = simutils.launch_simulation_parallel(
        reporters_config=reporters_config,
        resolution_time_gen=resolution_time_gen,
        max_iterations=max_iterations,
        bugs_by_priority=None,
        priority_generator=priority_generator,
        catalog_size=simdata.BATCH_SIZE,
        team_capacity=None,
        dev_size_generator=dev_team_size_training,
        dev_team_bandwidth=None,
        dev_bandwith_generator=dev_team_bandwith_training)

    simulation_result = consolidate_results("SIMULATION", None, None,
                                            reporters_config,
                                            simulation_output["completed_per_reporter"],
                                            simulation_output["completed_per_priority"],
                                            simulation_output["reported_per_priotity"])

    metrics_on_test = []

    for test_period in unique_batches:
        issues_for_period = test_issues[test_issues[simdata.BATCH_COLUMN] == test_period]

        bugs_by_priority = {index: value
                            for index, value in
                            issues_for_period[simdata.SIMPLE_PRIORITY_COLUMN].value_counts().iteritems()}

        dev_team_size, issues_resolved, resolved_in_period, dev_team_bandwith = get_dev_team_production(
            issues_for_period, time_after_last=time_after_last)

        bug_reporters = issues_for_period['Reported By']
        test_team_size = bug_reporters.nunique()

        print "Project ", project_key, " Test Period: ", test_period, " Testers: ", test_team_size, " Developers:", dev_team_size, \
            " Reports: ", bugs_by_priority, " Resolved in Period: ", issues_resolved, " Dev Team Bandwith: ", dev_team_bandwith

        if is_valid_period(issues_for_period):
            reporting_metrics = get_reporting_metrics(issues_for_period, resolved_in_period, reporters_config)
            metrics_on_test.append(reporting_metrics)
        else:
            print "PERIOD EXCLUDED!!! "

    return metrics_on_test, simulation_result


def simulate_project(project_key, enhanced_dataframe, debug=False, n_folds=5, test_size=None, max_iterations=1000):
    """
    Launches simulation analysis for an specific project.
    :param project_key: Project identifier.
    :param enhanced_dataframe: Dataframe with additional fields
    :return: None
    """
    issues_in_range = get_valid_reports(project_key, enhanced_dataframe, exclude_priority=None)
    issues_in_range = issues_in_range.sort(simdata.CREATED_DATE_COLUMN, ascending=True)

    keys_in_range = issues_in_range[simdata.ISSUE_KEY_COLUMN].unique()
    print "Original number of issue keys: ", len(keys_in_range)

    simulation_results = []

    name = ""
    if test_size is not None:
        name = "Test_Size_" + str(test_size)

        train_size = 1 - test_size
        split_point = int(len(keys_in_range) * train_size)

        keys_train = keys_in_range[:split_point]
        keys_test = keys_in_range[split_point:]

        print "Training simulation and validating: ", name, " Keys in Train: ", len(
            keys_train), " Keys in Test: ", len(keys_test)

        metrics_on_test, simulation_result = train_test_simulation(project_key, issues_in_range, max_iterations,
                                                                   keys_train,
                                                                   keys_test,
                                                                   fold=None)

        simulation_results.extend((metrics_on_test, simulation_result))
    else:
        k_fold = KFold(len(keys_in_range), n_folds=n_folds)

        for fold, (train_index, test_index) in enumerate(k_fold):
            print "Fold number: ", fold

            periods_train, periods_test = keys_in_range[train_index], keys_in_range[test_index]

            fold_results = train_test_simulation(project_key, issues_in_range, max_iterations, periods_train,
                                                 periods_test,
                                                 fold=fold)
            simulation_results.extend(fold_results)

    if len(simulation_results) > 0:
        # simvalid.analyse_results_regression(name=name, reporters_config=None, simulation_results=simulation_results,
        #                 project_key=project_key)

        simvalid.analyse_input_output(simulation_results[0], simulation_results[1])


def get_valid_projects(enhanced_dataframe):
    """
    Selects the projects that will be considered in analysis.
    :param enhanced_dataframe: Bug Report dataframe.
    :return: Project key list.
    """

    project_dataframe = defaultabuse.get_default_usage_data(enhanced_dataframe)

    threshold = 0.3
    using_priorities = project_dataframe[project_dataframe['non_default_ratio'] >= threshold]
    project_keys = using_priorities['project_key'].unique()

    return project_keys


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    max_iterations = 200
    valid_projects = get_valid_projects(enhanced_dataframe)
    test_sizes = [.4, .3, .2]

    # TODO: Remove later. Only for speeding testing
    test_sizes = [.5]

    for test_size in test_sizes:
        simulate_project(valid_projects, enhanced_dataframe, test_size=test_size, max_iterations=max_iterations)


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
