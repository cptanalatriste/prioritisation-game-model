"""
This modules triggers the bug report simulation.
"""
import time
import datetime
import pytz

from scipy import stats
import numpy as np

import pandas as pd

from sklearn.cross_validation import KFold

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

import simdata
import simutils
import simmodel

import winsound

DEBUG = False
PLOT = False


def get_reporters_configuration(max_chunk, training_dataset, debug=False):
    """
    Returns the reporting information required for the simulation to run.

    Includes drive-in tester removal.

    :param issues_in_range: Bug report data frame
    :return:
    """

    issues_by_tester = training_dataset[simdata.REPORTER_COLUMN].value_counts()
    testers_in_order = [index for index, _ in issues_by_tester.iteritems()]
    print "Reporters in training dataset: ", len(testers_in_order)

    reporters_config = []
    year, month = max_chunk[0].split("-")
    period_start = datetime.datetime(year=int(year), month=int(month), day=1, tzinfo=pytz.utc)

    for index, reporter_list in enumerate([[tester] for tester in testers_in_order]):
        bug_reports = simdata.filter_by_reporter(training_dataset, reporter_list)

        training_periods = bug_reports[simdata.PERIOD_COLUMN]
        bug_reports_for_batch = bug_reports[training_periods.isin(max_chunk)]
        batches = simdata.get_report_batches(bug_reports_for_batch)

        arrival_times = [batch["batch_head"] for batch in batches]

        inter_arrival_sample = simdata.get_interarrival_times(arrival_times, period_start)
        batch_sizes_sample = [batch["batch_count"] for batch in batches]

        try:
            inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(inter_arrival_sample)
            batch_size_gen = simutils.DiscreteEmpiricalDistribution(observations=pd.Series(data=batch_sizes_sample))

            reporter_name = "Consolidated Testers (" + str(len(reporter_list)) + ")"
            if len(reporter_list) == 1:
                reporter_name = reporter_list[0]

            priority_distribution = simutils.DiscreteEmpiricalDistribution(
                observations=bug_reports[simdata.SIMPLE_PRIORITY_COLUMN])
            priority_map = priority_distribution.get_probabilities()

            modified_priority = simdata.get_modified_priority_bugs(bug_reports)
            with_modified_priority = len(modified_priority.index)

            if debug:
                print "Interrival-time for tester ", reporter_name, " mean: ", np.mean(
                    inter_arrival_sample), " std: ", np.std(
                    inter_arrival_sample), "Batch-size", reporter_name, " mean: ", np.mean(
                    batch_sizes_sample), " std: ", np.std(
                    batch_sizes_sample), " priority_map ", priority_map, " with_modified_priority ", with_modified_priority

            reporters_config.append({'name': reporter_name,
                                     'interarrival_time_gen': inter_arrival_time_gen,
                                     'batch_size_gen': batch_size_gen,
                                     'reporter_list': reporter_list,
                                     'priority_map': priority_map,
                                     'with_modified_priority': with_modified_priority})
        except ValueError as _:
            if debug:
                print "Reporters ", reporter_list, " could not be added. Possible because insufficient samples."

    reporters_config = simutils.remove_drive_in_testers(reporters_config)
    return reporters_config


def consolidate_results(year_month, issues_for_period, resolved_in_month, reporters_config, completed_per_reporter,
                        completed_per_priority,
                        debug=False):
    """
    It consolidates the results from the simulation with the information contained in the data.

    :param debug: Detailed output messages
    :param year_month: Period identifier.
    :param issues_for_period: Issues reported on the same period of report.
    :param resolved_in_month: Issues resolved on the same period of report.
    :param reporters_config:   Reporter configuration.
    :param completed_reports: Simulation results.
    :return:
    """
    simulation_result = {"period": year_month,
                         "results_per_reporter": [],
                         "results_per_priority": [],
                         "true_resolved": len(resolved_in_month.index)}

    results = []
    for report in completed_per_reporter:
        total_resolved = 0
        for reporter_config in reporters_config:
            total_resolved += report[reporter_config['name']]
        results.append(total_resolved)

    simulation_result["predicted_resolved"] = np.median(results)

    # TODO: This reporter/priority logic can be refactored.
    for priority in [simdata.SEVERE_PRIORITY, simdata.NON_SEVERE_PRIORITY, simdata.NORMAL_PRIORITY]:
        resolved_per_priority = resolved_in_month[resolved_in_month[simdata.SIMPLE_PRIORITY_COLUMN] == priority]

        resolved_on_simulation = [report[priority] for report in completed_per_priority]
        predicted_resolved = np.median(resolved_on_simulation)

        simulation_result['results_per_priority'].append({'priority': priority,
                                                          'true_resolved': len(resolved_per_priority.index),
                                                          "predicted_resolved": predicted_resolved})

    for reporter_config in reporters_config:
        reporter_name = reporter_config['name']
        true_resolved = simdata.filter_by_reporter(resolved_in_month, reporter_config['reporter_list'])
        true_reported = simdata.filter_by_reporter(issues_for_period, reporter_config['reporter_list'])

        resolved_on_simulation = [report[reporter_name] for report in completed_per_reporter]
        predicted_resolved = np.median(resolved_on_simulation)

        sample_median, sample_std, sample_size = predicted_resolved, np.std(resolved_on_simulation), len(
            resolved_on_simulation)
        alpha = 0.95
        confidence_interval = stats.norm.interval(alpha, loc=sample_median, scale=sample_std / np.sqrt(sample_size))

        if debug:
            print "Reporter ", reporter_name, "sample_median ", sample_median, " sample_std ", sample_std, " confidence interval: ", \
                confidence_interval, " true_resolved ", len(true_resolved.index), " true_reported ", len(
                true_reported.index)

        simulation_result["results_per_reporter"].append({"reporter_name": reporter_name,
                                                          "true_resolved": len(true_resolved.index),
                                                          "predicted_resolved": predicted_resolved})

    if debug:
        print "simulation_result ", simulation_result

    return simulation_result


def analyse_results(reporters_config=None, simulation_results=None, project_key=None, debug=DEBUG, plot=PLOT):
    """
    Per each tester, it anaysis how close is simulation to real data.
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

            mse = mean_squared_error(completed_true, completed_predicted)
            rmse = np.sqrt(mse)
            msa = mean_absolute_error(completed_true, completed_predicted)

            print "Project ", project_key, " Tester ", reporter_name, " RMSE ", rmse, \
                " Mean Squared Error ", mse, " Mean Absolute Error ", msa

    total_completed = [result['true_resolved'] for result in simulation_results]
    total_predicted = [result['predicted_resolved'] for result in simulation_results]

    if debug:
        print "total_completed ", total_completed
        print "total_predicted ", total_predicted

    total_mse = mean_squared_error(total_completed, total_predicted)
    total_mar = mean_absolute_error(total_completed, total_predicted)
    total_medar = median_absolute_error(total_completed, total_predicted)
    total_mmre = simutils.mean_magnitude_relative_error(total_completed, total_predicted, balanced=False)
    total_bmmre = simutils.mean_magnitude_relative_error(total_completed, total_predicted, balanced=True)
    total_mdmre = simutils.median_magnitude_relative_error(total_completed, total_predicted)

    print "RMSE for total bug resolved in Project ", project_key, ": ", np.sqrt(
        total_mse), " Mean Squared Error ", total_mse, " Mean Absolute Error: ", total_mar, " Median Absolute Error: ", \
        total_medar, " Mean Magnitude Relative Error ", total_mmre, " Balanced MMRE ", total_bmmre, "Median Magnitude Relative Error ", total_mdmre

    simutils.plot_correlation(total_predicted, total_completed, "_".join(project_key) + "-Total Resolved", plot)

    for priority in [simdata.NON_SEVERE_PRIORITY, simdata.SEVERE_PRIORITY, simdata.NORMAL_PRIORITY]:
        completed_true = []
        completed_predicted = []

        for simulation_result in simulation_results:
            priority_true = [result['true_resolved'] for result in simulation_result['results_per_priority'] if
                             result['priority'] == priority][0]
            completed_true.append(priority_true)

            priority_predicted = [result['predicted_resolved'] for result in simulation_result['results_per_priority']
                                  if
                                  result['priority'] == priority][0]
            completed_predicted.append(priority_predicted)

        mse = mean_squared_error(completed_true, completed_predicted)
        rmse = np.sqrt(mse)

        print "Project ", project_key, " Priority ", priority, " RMSE ", rmse, " Mean Squared Error ", mse

        if debug:
            print "priority ", priority, " completed_true ", completed_true
            print "priority ", priority, " completed_predicted ", completed_predicted

        simutils.plot_correlation(completed_predicted, completed_true,
                                  "-".join(project_key) + "-Priority " + str(priority), plot)


def get_simulation_input(training_issues):
    """
    Extract the simulation paramaters from the training dataset.
    :param training_issues: Training data set.
    :return: Variate generator for resolution times, priorities and reporter inter-arrival time.
    """
    resolved_issues = simdata.filter_resolved(training_issues, only_with_commits=False,
                                              only_valid_resolution=False)
    resolution_time_sample = resolved_issues[simdata.RESOLUTION_TIME_COLUMN].dropna()

    print "Resolution times in Training Range: \n", resolution_time_sample.describe()
    resolution_time_gen = None
    if len(resolution_time_sample.index) >= simutils.MINIMUM_OBSERVATIONS:
        resolution_time_gen = simutils.ContinuousEmpiricalDistribution(resolution_time_sample)

    priority_sample = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    print "Simplified Priorities in Training Range: \n ", priority_sample.value_counts()

    priority_gen = simutils.DiscreteEmpiricalDistribution(observations=priority_sample)

    return resolution_time_gen, priority_gen


def get_valid_reports(project_keys, enhanced_dataframe):
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


def get_dev_team_production(test_period, issues_for_period, simulation_days):
    """
    Returns the production of the development team for a specific period.
    :return: Developer Team Size and Developer Team Production.
    """
    year, month_string = test_period.split('-')

    month = int(month_string)
    start_date = datetime.datetime(year=int(year), month=month, day=1, tzinfo=pytz.utc)

    margin = datetime.timedelta(days=simulation_days)
    end_date = start_date + margin

    resolved_issues = simdata.filter_resolved(issues_for_period, only_with_commits=False,
                                              only_valid_resolution=False)
    resolved_in_period = simdata.filter_by_date_range(simdata.RESOLUTION_DATE_COLUMN, resolved_issues,
                                                      start_date,
                                                      end_date)

    bug_resolvers = resolved_in_period['JIRA Resolved By']
    dev_team_size = bug_resolvers.nunique()
    issues_resolved = len(resolved_in_period.index)

    return dev_team_size, issues_resolved, resolved_in_period


def simulate_project(project_key, enhanced_dataframe, debug=True, n_folds=5, max_iterations=1000):
    """
    Launches simulation analysis for an specific project.
    :param project_key: Project identifier.
    :param enhanced_dataframe: Dataframe with additional fields
    :return: None
    """
    issues_in_range = get_valid_reports(project_key, enhanced_dataframe)

    period_in_range = issues_in_range[simdata.PERIOD_COLUMN].unique()
    print "Original number of periods: ", len(period_in_range)
    period_in_range.sort()

    simulation_results = []

    k_fold = KFold(len(period_in_range), n_folds=n_folds)

    for train_index, test_index in k_fold:
        periods_train, periods_test = period_in_range[train_index], period_in_range[test_index]

        continuous_chunks = get_continuous_chunks(periods_train)
        max_chunk = max(continuous_chunks, key=len)

        training_issues = issues_in_range[issues_in_range[simdata.PERIOD_COLUMN].isin(periods_train)]
        print "Issues in training: ", len(training_issues.index)

        reporters_config = get_reporters_configuration(max_chunk, training_issues)
        print "Number of reporters after drive-by filtering: ", len(reporters_config)

        simutils.assign_strategies(reporters_config, training_issues)

        engaged_testers = [reporter_config['name'] for reporter_config in reporters_config]
        training_issues = simdata.filter_by_reporter(training_issues, engaged_testers)
        print "Issues in training after reporter filtering: ", len(training_issues.index)

        resolution_time_gen, priority_gen = get_simulation_input(training_issues)
        if resolution_time_gen is None:
            print "Not enough resolution time info! ", project_key
            return

        test_issues = issues_in_range[issues_in_range[simdata.PERIOD_COLUMN].isin(periods_test)]
        print "Issues in test: ", len(test_issues.index)
        test_issues = simdata.filter_by_reporter(test_issues, engaged_testers)
        print "Issues in test after reporter filtering: ", len(test_issues.index)

        for test_period in periods_test:
            issues_for_period = test_issues[test_issues[simdata.PERIOD_COLUMN] == test_period]

            simulation_days = 30
            reports_per_month = len(issues_for_period.index)

            dev_team_size, issues_resolved, resolved_in_period = get_dev_team_production(test_period, issues_for_period,
                                                                                         simulation_days)

            bug_reporters = issues_for_period['Reported By']
            test_team_size = bug_reporters.nunique()

            if debug:
                print "Project ", project_key, " Test Period: ", test_period, " Testers: ", test_team_size, " Developers:", dev_team_size, \
                    " Reports: ", reports_per_month, " Resolved in Period: ", issues_resolved

            simulation_time = simulation_days * 24

            completed_per_reporter, completed_per_priority = simutils.launch_simulation(team_capacity=dev_team_size,
                                                                                        report_number=reports_per_month,
                                                                                        reporters_config=reporters_config,
                                                                                        resolution_time_gen=resolution_time_gen,
                                                                                        priority_gen=priority_gen,
                                                                                        max_time=simulation_time,
                                                                                        max_iterations=max_iterations)

            simulation_result = consolidate_results(test_period, issues_for_period, resolved_in_period,
                                                    reporters_config,
                                                    completed_per_reporter, completed_per_priority)
            simulation_results.append(simulation_result)

    if len(simulation_results) > 0:
        analyse_results(reporters_config=None, simulation_results=simulation_results, project_key=project_key)


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)
    # project_lists = [["CASSANDRA"], ["CLOUDSTACK"], ["OFBIZ"], ["CLOUDSTACK", "OFBIZ", "CASSANDRA"]]
    project_lists = [enhanced_dataframe["Project Key"].unique()]
    # project_lists = [[project] for project in enhanced_dataframe["Project Key"].unique()]
    project_lists = [["MESOS"]]
    for project_list in project_lists:
        n_folds = 3
        max_iterations = 100
        simulate_project(project_list, enhanced_dataframe, n_folds=n_folds, max_iterations=max_iterations)


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
