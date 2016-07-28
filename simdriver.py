"""
This modules triggers the bug report simulation.
"""
import time
import datetime
import pytz

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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


def launch_simulation(team_capacity, report_number, reporters_config, resolution_time_gen, priority_gen,
                      max_time):
    """
    Triggers the simulation according a given configuration.

    :param team_capacity: Number of developers in the team.
    :param report_number: Number of bugs for the period.
    :param reporters_config: Bug reporter configuration.
    :param resolution_time_gen: Resolution time required by developers.
    :param priority_gen: The priority contained on the bug reports.
    :param max_time: Simulation time.
    :return: List containing the number of fixed reports.
    """

    # max_iterations = 1000
    max_iterations = 200
    completed_per_reporter = []
    completed_per_priority = []

    for _ in range(max_iterations):
        np.random.seed()
        reporter_monitors, priority_monitors = simmodel.run_model(team_capacity=team_capacity,
                                                                  report_number=report_number,
                                                                  reporters_config=reporters_config,
                                                                  resolution_time_gen=resolution_time_gen,
                                                                  priority_gen=priority_gen,
                                                                  max_time=max_time)

        result_per_reporter = {reporter_name: monitor.count() for reporter_name, monitor in
                               reporter_monitors.iteritems()}
        completed_per_reporter.append(result_per_reporter)

        result_per_priority = {priority: monitor.count() for priority, monitor in priority_monitors.iteritems()}
        completed_per_priority.append(result_per_priority)

    return completed_per_reporter, completed_per_priority


def get_reporters_configuration(issues_in_range, participation_coef, debug=DEBUG):
    """
    Returns the reporting information required for the simulation to run.
    :param issues_in_range: Bug report data frame
    :return:
    """

    issues_by_tester = issues_in_range[simdata.REPORTER_COLUMN].value_counts()
    testers_in_order = [index for index, _ in issues_by_tester.iteritems()]

    reporters_config = []
    total_training_periods = issues_in_range[simdata.PERIOD_COLUMN].nunique()
    for index, reporter_list in enumerate([[tester] for tester in testers_in_order]):
        # for index, reporter_list in enumerate([first_class_testers, second_class_testers]):
        bug_reports = simdata.filter_by_reporter(issues_in_range, reporter_list)

        reporter_participation = bug_reports[simdata.PERIOD_COLUMN].nunique()

        if reporter_participation >= total_training_periods / participation_coef:
            batches = simdata.get_report_batches(bug_reports)

            arrival_times = [batch["batch_head"] for batch in batches]
            inter_arrival_sample = simdata.get_interarrival_times(arrival_times)
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
                print "Reporters ", reporter_list, " could not be added. Possible because insufficient samples."

    return reporters_config


def assign_strategies(reporters_config, training_issues, debug=DEBUG):
    """
    Assigns an inflation pattern to the reporter based on clustering.
    :param reporters_config: Reporter configuration.
    :param training_issues: Training dataset.
    :return: Reporting Configuration including inflation pattern.
    """
    reporter_records = [
        [config['priority_map'][simdata.NON_SEVERE_PRIORITY], config['priority_map'][simdata.NORMAL_PRIORITY],
         config['priority_map'][simdata.SEVERE_PRIORITY],
         config['with_modified_priority']] for config in reporters_config]

    global_priority_map = simutils.DiscreteEmpiricalDistribution(
        observations=training_issues[simdata.SIMPLE_PRIORITY_COLUMN]).get_probabilities()

    if debug:
        print "global_priority_map: ", global_priority_map

    reporter_dataframe = pd.DataFrame(reporter_records)
    correction_column = "Corrections"
    non_severe_column = "Non-Severe"
    severe_column = "Severe"
    normal_column = "Normal"

    reporter_dataframe.columns = [non_severe_column, normal_column, severe_column, correction_column]

    scaler = StandardScaler()

    # Removing scaling because of cluster quality.
    # report_features = scaler.fit_transform(reporter_dataframe.values)
    # global_features = scaler.transform(
    #     [global_priority_map[simdata.NON_SEVERE_PRIORITY], global_priority_map[simdata.NORMAL_PRIORITY],
    #      global_priority_map[simdata.SEVERE_PRIORITY], 0.0])

    global_features = [global_priority_map[simdata.NON_SEVERE_PRIORITY], global_priority_map[simdata.NORMAL_PRIORITY],
                       global_priority_map[simdata.SEVERE_PRIORITY], 0.0]
    report_features = reporter_dataframe.values

    print "Starting clustering algorithms ..."
    k_means = KMeans(n_clusters=2,
                     init='random',
                     max_iter=300,
                     tol=1e-04,
                     random_state=0)

    predicted_clusters = k_means.fit_predict(report_features)

    main_cluster = k_means.predict(global_features)

    strategy_column = 'strategy'
    reporter_dataframe[strategy_column] = [
        simmodel.NOT_INFLATE_STRATEGY if cluster == main_cluster else simmodel.INFLATE_STRATEGY for
        cluster in
        predicted_clusters]

    for index, strategy in enumerate(reporter_dataframe[strategy_column].values):
        reporters_config[index]['strategy'] = strategy

    for strategy in [simmodel.NOT_INFLATE_STRATEGY, simmodel.INFLATE_STRATEGY]:
        reporters_per_strategy = reporter_dataframe[reporter_dataframe[strategy_column] == strategy]
        print "Strategy: ", strategy, " reporters: ", len(reporters_per_strategy.index), " avg corrections: ", \
            reporters_per_strategy[correction_column].mean(), " avg non-severe prob: ", reporters_per_strategy[
            non_severe_column].mean(), " avg normal prob: ", reporters_per_strategy[
            normal_column].mean(), " avg severe prob: ", reporters_per_strategy[severe_column].mean()


def consolidate_results(year_month, issues_for_period, resolved_in_month, reporters_config, completed_per_reporter,
                        completed_per_priority,
                        debug=DEBUG):
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

    print "RMSE for total bug resolved in Project ", project_key, ": ", np.sqrt(
        total_mse), " Mean Squared Error ", total_mse, " Mean Absolute Error: ", total_mar, " Median Absolute Error: ", total_medar

    if plot:
        plot_correlation(total_predicted, total_completed, "Total Resolved")

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

        if plot:
            plot_correlation(completed_predicted, completed_true, "Priority " + str(priority))


def plot_correlation(total_predicted, total_completed, title):
    """
    A scatter plot for seeing how correlation goes.
    :param total_predicted: List of predicted values.
    :param total_completed: List of real values.
    :return:
    """
    plt.clf()

    plt.scatter(total_predicted, total_completed)
    plt.title(title)
    plt.xlabel("Predicted Resolved")
    plt.ylabel("Actual Resolved")
    plt.plot([min(total_completed), max(total_completed)], [[min(total_completed)], [max(total_completed)]])
    plt.show()


def get_simulation_input(training_issues):
    """
    Extract the simulation paramaters from the training dataset.
    :param training_issues: Training data set.
    :return: Variate generator for resolution times, priorities and reporter inter-arrival time.
    """
    resolved_issues = simdata.filter_resolved(training_issues, only_with_commits=False)
    resolution_time_sample = resolved_issues[simdata.RESOLUTION_TIME_COLUMN].dropna()
    print "Resolution times in Training Range: \n", resolution_time_sample.describe()

    priority_sample = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    print "Simplified Priorities in Training Range: \n ", priority_sample.value_counts()

    resolution_time_gen = simutils.ContinuousEmpiricalDistribution(resolution_time_sample)
    priority_gen = simutils.DiscreteEmpiricalDistribution(observations=priority_sample)

    return resolution_time_gen, priority_gen


def get_bug_reports(project_keys, enhanced_dataframe):
    """
    Returns the issues valid for simulation analysis. It includes:

    - Filtered by project
    - Only reports from top testers
    - Only reports while priorities where corrected.

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


def simulate_project(project_key, enhanced_dataframe, debug=DEBUG):
    """
    Launches simulation analysis for an specific project.
    :param project_key: Project identifier.
    :param enhanced_dataframe: Dataframe with additional fields
    :return: None
    """
    issues_in_range = get_bug_reports(project_key, enhanced_dataframe)

    period_in_range = issues_in_range[simdata.PERIOD_COLUMN].unique()
    print "Original number of periods: ", len(period_in_range)
    period_in_range.sort()

    simulation_results = []

    k_fold = KFold(len(period_in_range), n_folds=5)
    for train_index, test_index in k_fold:
        periods_train, periods_test = period_in_range[train_index], period_in_range[test_index]
        training_issues = issues_in_range[issues_in_range[simdata.PERIOD_COLUMN].isin(periods_train)]
        print "Issues in training: ", len(training_issues.index)

        participation_coef = 3
        reporters_config = get_reporters_configuration(training_issues, participation_coef)
        print "Number of reporters: ", len(reporters_config)

        assign_strategies(reporters_config, training_issues)

        engaged_testers = [reporter_config['name'] for reporter_config in reporters_config]
        training_issues = simdata.filter_by_reporter(training_issues, engaged_testers)
        print "Issues in training after reporter filtering: ", len(training_issues.index)

        resolution_time_gen, priority_gen = get_simulation_input(training_issues)

        test_issues = issues_in_range[issues_in_range[simdata.PERIOD_COLUMN].isin(periods_test)]
        print "Issues in test: ", len(test_issues.index)
        test_issues = simdata.filter_by_reporter(test_issues, engaged_testers)
        print "Issues in test after reporter filtering: ", len(test_issues.index)

        for test_period in periods_test:
            issues_for_period = test_issues[test_issues[simdata.PERIOD_COLUMN] == test_period]

            reports_per_month = len(issues_for_period.index)

            # year, month_string, week = test_period.split('-')
            # Excluding week since to few reports get fixed.
            year, month_string = test_period.split('-')

            month = int(month_string)
            simulation_days = 30
            start_date = datetime.datetime(year=int(year), month=month, day=1, tzinfo=pytz.utc)

            margin = datetime.timedelta(days=simulation_days)
            end_date = start_date + margin

            resolved_issues = simdata.filter_resolved(issues_for_period, only_with_commits=False)
            resolved_in_period = simdata.filter_by_date_range(simdata.RESOLUTION_DATE_COLUMN, resolved_issues,
                                                              start_date,
                                                              end_date)

            bug_resolvers = resolved_in_period['JIRA Resolved By']
            dev_team_size = bug_resolvers.nunique()

            issues_resolved = len(resolved_in_period.index)

            bug_reporters = issues_for_period['Reported By']
            test_team_size = bug_reporters.nunique()

            if debug:
                print "Project ", project_key, " Test Period: ", test_period, " Testers: ", test_team_size, " Developers:", dev_team_size, \
                    " Reports: ", reports_per_month, " Resolved in Period: ", issues_resolved

            simulation_time = simulation_days * 24

            completed_per_reporter, completed_per_priority = launch_simulation(team_capacity=dev_team_size,
                                                                               report_number=reports_per_month,
                                                                               reporters_config=reporters_config,
                                                                               resolution_time_gen=resolution_time_gen,
                                                                               priority_gen=priority_gen,
                                                                               max_time=simulation_time)

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
    for project_list in project_lists:
        simulate_project(project_list, enhanced_dataframe)


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
