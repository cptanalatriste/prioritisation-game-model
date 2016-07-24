"""
This modules triggers the bug report simulation.
"""
import time
import datetime
import pandas as pd
import pytz
from scipy import stats
from sklearn.metrics import r2_score

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

        completed_reports.append(np.sum([monitor.count() for monitor in resol_time_monitors]))

    return completed_reports


def get_reporters_configuration(issues_in_range):
    """
    Returns the reporting information required for the simulation to run.
    :param issues_in_range: Bug report data frame
    :return:
    """

    issues_by_tester = issues_in_range[simdata.REPORTER_COLUMN].value_counts()
    threshold = issues_by_tester.quantile(0.90)

    first_class_testers = [index for index, value in issues_by_tester.iteritems() if value >= threshold]
    second_class_testers = [index for index, value in issues_by_tester.iteritems() if value < threshold]

    reporters_config = []
    for index, reporter_list in enumerate([first_class_testers, second_class_testers]):
        bug_reports = simdata.filter_by_reporter(issues_in_range, reporter_list)
        inter_arrival_sample = simdata.get_interarrival_times(bug_reports)
        inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(inter_arrival_sample)

        print "Interrival-time for tester ", str(index + 1), " mean: ", np.mean(inter_arrival_sample), " std: ", np.std(
            inter_arrival_sample), " testers ", len(reporter_list)

        reporters_config.append({'name': "Consolidated Tester " + str(index + 1),
                                 'interarrival_time_gen': inter_arrival_time_gen})

    return reporters_config


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    # project_key = "CASSANDRA"
    # project_key = "CLOUDSTACK"
    project_key = "OFBIZ"

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

    months_in_range = issues_in_range[simdata.CREATED_MONTH_COLUMN].unique()

    team_sizes = []
    period_reports = []

    completed_predicted = []
    completed_true = []

    reporters_config = get_reporters_configuration(issues_in_range)
    print "Number of reporters: ", len(reporters_config)

    overestimate, understimate, in_interval = 0, 0, 0
    for year_month in months_in_range:
        issues_for_month = issues_in_range[issues_in_range[simdata.CREATED_MONTH_COLUMN] == year_month]

        reports_per_month = len(issues_for_month.index)
        period_reports.append(reports_per_month)

        year, month = year_month.split('-')
        start_date = datetime.datetime(year=int(year), month=int(month), day=1, tzinfo=pytz.utc)
        margin = datetime.timedelta(days=30)
        end_date = start_date + margin

        resolved_issues = simdata.filter_resolved(issues_for_month, only_with_commits=False)
        resolved_in_month = simdata.filter_by_date_range(simdata.RESOLUTION_DATE_COLUMN, resolved_issues, start_date,
                                                         end_date)

        bug_resolvers = resolved_in_month['JIRA Resolved By']
        dev_team_size = bug_resolvers.nunique()
        team_sizes.append(dev_team_size)

        issues_resolved = len(resolved_in_month.index)

        bug_reporters = issues_for_month['Reported By']
        test_team_size = bug_reporters.nunique()

        print "Period: ", year_month, " Testers: ", test_team_size, " Developers:", dev_team_size, \
            " Reports: ", reports_per_month, " Resolved in Month: ", issues_resolved

        simulation_time = 30 * 24
        alpha = 0.95

        completed_reports = launch_simulation(team_capacity=dev_team_size, report_number=reports_per_month,
                                              reporters_config=reporters_config,
                                              resolution_time_sample=resolution_times,
                                              priority_sample=priorities_in_range, max_time=simulation_time)

        sample_mean, sample_std, sample_size = np.mean(completed_reports), np.std(completed_reports), len(
            completed_reports)
        confidence_interval = stats.norm.interval(alpha, loc=sample_mean, scale=sample_std / np.sqrt(sample_size))
        print "sample_size", sample_size, "sample_mean ", sample_mean, " sample_std ", sample_std, " confidence interval: ", \
            confidence_interval

        completed_predicted.append(sample_mean)
        completed_true.append(issues_resolved)

        if confidence_interval[0] <= issues_resolved <= confidence_interval[1]:
            in_interval += 1
        elif issues_resolved < confidence_interval[0]:
            overestimate += 1
        elif issues_resolved > confidence_interval[1]:
            understimate += 1

    # simdata.launch_histogram(period_reports)

    coefficient_of_determination = r2_score(completed_true, completed_predicted)
    print "Simulation finished! coefficient_of_determination ", coefficient_of_determination
    print "in_interval ", in_interval, " overestimate ", overestimate, " understimate ", understimate


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
