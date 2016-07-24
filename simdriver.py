"""
This modules triggers the bug report simulation.
"""
import time
import datetime
import pandas as pd
import pytz
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import simdata
import simutils
import simmodel
import numpy as np

import winsound  # vibhavcarlos


def launch_simulation(team_capacity, report_number, reporters_config, resolution_time_sample, priority_sample,
                      max_time):
    """
    Triggers the simulation according a given configuration.

    :param team_capacity: Number of developers in the team.
    :param report_number: Number of bugs for the period.
    :param reporters_config: Bug reporter configuration.
    :param resolution_time_sample: Resolution time required by developers.
    :param priority_sample: The priority contained on the bug reports.
    :param max_time: Simulation time.
    :return: List containing the number of fixed reports.
    """

    resolution_time_gen = simutils.ContinuousEmpiricalDistribution(resolution_time_sample)
    priority_gen = simutils.DiscreteEmpiricalDistribution(priority_sample)

    # max_iterations = 1000
    max_iterations = 100
    completed_reports = []

    for a_seed in range(max_iterations):
        np.random.seed(a_seed)
        resol_time_monitors = simmodel.run_model(team_capacity=team_capacity, report_number=report_number,
                                                 reporters_config=reporters_config,
                                                 resolution_time_gen=resolution_time_gen,
                                                 priority_gen=priority_gen,
                                                 max_time=max_time)

        simulation_result = {reporter_name: monitor.count() for reporter_name, monitor in
                             resol_time_monitors.iteritems()}
        completed_reports.append(simulation_result)

    return completed_reports


def get_reporters_configuration(issues_in_range):
    """
    Returns the reporting information required for the simulation to run.
    :param issues_in_range: Bug report data frame
    :return:
    """

    issues_by_tester = issues_in_range[simdata.REPORTER_COLUMN].value_counts()
    testers_in_order = [index for index, _ in issues_by_tester.iteritems()]

    # threshold = issues_by_tester.quantile(0.90)
    #
    # first_class_testers = [index for index, value in issues_by_tester.iteritems() if value >= threshold]
    # second_class_testers = [index for index, value in issues_by_tester.iteritems() if value < threshold]

    reporters_config = []
    for index, reporter_list in enumerate(
            [[testers_in_order[0]], [testers_in_order[1]], [testers_in_order[2]], testers_in_order[3:]]):
        # for index, reporter_list in enumerate([first_class_testers, second_class_testers]):
        bug_reports = simdata.filter_by_reporter(issues_in_range, reporter_list)
        inter_arrival_sample = simdata.get_interarrival_times(bug_reports)
        inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(inter_arrival_sample)

        reporter_name = "Consolidated Testers (" + str(len(reporter_list)) + ")"
        if len(reporter_list) == 1:
            reporter_name = reporter_list[0]

        print "Interrival-time for tester ", reporter_name, " mean: ", np.mean(inter_arrival_sample), " std: ", np.std(
            inter_arrival_sample), " testers ", len(reporter_list)

        reporters_config.append({'name': reporter_name,
                                 'interarrival_time_gen': inter_arrival_time_gen,
                                 'reporter_list': reporter_list})

    return reporters_config


def get_bug_reports(project_key, enhanced_dataframe):
    """
    Returns the issues valid for simulation analysis. It includes:

    - Filtered by project
    - Only reports from top testers
    - Only reports while priorities where corrected.

    :param project_key: Project identifier.
    :param enhanced_dataframe: Bug report dataframe.
    :return: Filtered dataframe
    """
    print "Starting analysis for project ", project_key, " ..."

    project_bugs = simdata.filter_by_project(enhanced_dataframe, project_key)
    print "Total issues for project ", project_key, ": ", len(project_bugs.index)

    project_reporters = project_bugs[simdata.REPORTER_COLUMN].value_counts()

    quantile = 0.9
    tester_threshold = project_reporters.quantile(quantile)
    print "Minimum reports to be included: ", tester_threshold
    top_testers = [index for index, value in project_reporters.iteritems() if value >= tester_threshold]

    project_bugs = simdata.filter_by_reporter(project_bugs, top_testers)
    print "Top-tester production for this project: ", len(project_bugs.index), " Testers: ", len(top_testers)

    with_corrected_priority = simdata.get_modified_priority_bugs(project_bugs)
    min_create_date = with_corrected_priority[simdata.CREATED_DATE_COLUMN].min()
    max_create_date = with_corrected_priority[simdata.CREATED_DATE_COLUMN].max()

    print "With corrected priorities: ", len(
        with_corrected_priority.index), " between ", min_create_date, " and ", max_create_date

    issues_in_range = simdata.filter_by_create_date(project_bugs, min_create_date, max_create_date)
    print "All issues in that range: ", len(issues_in_range.index)

    return issues_in_range


def consolidate_results(year_month, issues_for_period, resolved_in_month, reporters_config, completed_reports,
                        debug=True):
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
                         "results_per_reporter": []}

    for reporter_config in reporters_config:
        reporter_name = reporter_config['name']
        true_resolved = simdata.filter_by_reporter(resolved_in_month, reporter_config['reporter_list'])
        true_reported = simdata.filter_by_reporter(issues_for_period, reporter_config['reporter_list'])

        resolved_on_simulation = [report[reporter_name] for report in completed_reports]
        predicted_resolved = np.mean(resolved_on_simulation)

        sample_mean, sample_std, sample_size = predicted_resolved, np.std(resolved_on_simulation), len(
            resolved_on_simulation)
        alpha = 0.95
        confidence_interval = stats.norm.interval(alpha, loc=sample_mean, scale=sample_std / np.sqrt(sample_size))

        if debug:
            print "Reporter ", reporter_name, "sample_mean ", sample_mean, " sample_std ", sample_std, " confidence interval: ", \
                confidence_interval, " true_resolved ", len(true_resolved.index), " true_reported ", len(
                true_reported.index)

        simulation_result["results_per_reporter"].append({"reporter_name": reporter_name,
                                                          "true_resolved": len(true_resolved.index),
                                                          "predicted_resolved": predicted_resolved})

    return simulation_result


def analyse_results(reporters_config, simulation_results, project_key):
    """
    Per each tester, it anaysis how close is simulation to real data.
    :param reporters_config: Tester configuration.
    :param simulation_results: Result from simulation.
    :return: None
    """
    for reporter_config in reporters_config:
        reporter_name = reporter_config['name']
        completed_true = []
        completed_predicted = []

        for simulation_result in simulation_results:
            completed_true.append([result['true_resolved'] for result in simulation_result["results_per_reporter"] if
                                   result["reporter_name"] == reporter_name])
            completed_predicted.append(
                [result['predicted_resolved'] for result in simulation_result["results_per_reporter"] if
                 result["reporter_name"] == reporter_name])
        coefficient_of_determination = r2_score(completed_true, completed_predicted)
        mse = mean_squared_error(completed_true, completed_predicted)
        msa = mean_absolute_error(completed_true, completed_predicted)

        print "Project ", project_key, " Tester ", reporter_name, " coefficient_of_determination ", coefficient_of_determination, \
            " Mean Squared Error ", mse, " Mean Absolute Error ", msa


def is_game_valid(reporters_config, issues_for_period):
    """
    Determines the conditions for the period to run start simulation.
    :param reporters_config: Reporter configuration.
    :param issues_for_period: Bug reports for the period.
    :return: True if valid, false otherwise.
    """
    minimum_reports = 10

    for reporter_config in reporters_config:
        reporter_name = reporter_config['name']
        reported_issues = simdata.filter_by_reporter(issues_for_period, reporter_config['reporter_list'])

        if len(reported_issues.index) < minimum_reports:
            return False

    return True


def simulate_project(project_key, enhanced_dataframe):
    """
    Launches simulation anallysis for an specific project.
    :param project_key: Project identifier.
    :param enhanced_dataframe: Dataframe with additional fields
    :return: None
    """
    issues_in_range = get_bug_reports(project_key, enhanced_dataframe)

    resolved_issues = simdata.filter_resolved(issues_in_range)
    resolution_times = resolved_issues[simdata.RESOLUTION_TIME_COLUMN].dropna()
    print "Resolution times in Range: \n", resolution_times.describe()
    # simdata.launch_histogram(resolution_times)

    interrival_times_range = simdata.get_interarrival_times(issues_in_range)
    print "Inter-arrival times in Range: \n ", interrival_times_range.describe()
    # simdata.launch_histogram(interrival_times_range)

    priorities_in_range = issues_in_range[simdata.SIMPLE_PRIORITY_COLUMN]
    print "Simplified Priorities in Range: \n ", priorities_in_range.value_counts()

    reporters_in_range = issues_in_range['Reported By']
    print "Reporters in Range: \n ", reporters_in_range.describe()

    period_in_range = issues_in_range[simdata.PERIOD_COLUMN].unique()

    reporters_config = get_reporters_configuration(issues_in_range)
    print "Number of reporters: ", len(reporters_config)

    simulation_results = []
    for period in period_in_range:
        issues_for_period = issues_in_range[issues_in_range[simdata.PERIOD_COLUMN] == period]

        reports_per_month = len(issues_for_period.index)

        if is_game_valid(reporters_config, issues_for_period):
            year, period_value = period.split('-')
            month = int(period_value) * 3 - 2
            simulation_days = 90
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

            print "Project ", project_key, " Period: ", period, " Testers: ", test_team_size, " Developers:", dev_team_size, \
                " Reports: ", reports_per_month, " Resolved in Period: ", issues_resolved

            simulation_time = simulation_days * 24

            completed_reports = launch_simulation(team_capacity=dev_team_size, report_number=reports_per_month,
                                                  reporters_config=reporters_config,
                                                  resolution_time_sample=resolution_times,
                                                  priority_sample=priorities_in_range, max_time=simulation_time)

            simulation_result = consolidate_results(period, issues_for_period, resolved_in_period, reporters_config,
                                                    completed_reports)
            simulation_results.append(simulation_result)

    analyse_results(reporters_config, simulation_results, project_key)


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    project_list = ["CLOUDSTACK", "OFBIZ", "CASSANDRA"]
    # project_list = ["OFBIZ"]

    for project_key in project_list:
        simulate_project(project_key, enhanced_dataframe)


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
