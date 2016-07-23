"""
This modules triggers the bug report simulation.
"""
import datetime
import pandas as pd
import pytz
from scipy import stats
from sklearn.metrics import r2_score

import simdata
import simutils
import simmodel
import numpy as np

import winsound


def launch_simulation(team_capacity, report_number, inter_arrival_sample, resolution_time_sample, priority_sample,
                      max_time):
    """
    Triggers the simulation according a given configuration.

    :param team_capacity: Number of developers in the team.
    :param report_number: Number of bugs for the period.
    :param inter_arrival_sample: Interrival time for bug reports.
    :param resolution_time_sample: Resolution time required by developers.
    :param priority_sample: The priority contained on the bug reports.
    :param max_time: Simulation time.
    :return: List containing the number of fixed reports.
    """
    print "Launching simulation-> team_capacity:", team_capacity, " report_number: ", report_number, " max_time ", max_time

    inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(inter_arrival_sample)
    resolution_time_gen = simutils.ContinuousEmpiricalDistribution(resolution_time_sample)
    priority_gen = simutils.DiscreteEmpiricalDistribution(priority_sample)

    # max_iterations = 1000
    max_iterations = 100
    completed_reports = []
    for a_seed in range(max_iterations):
        np.random.seed(a_seed)
        resol_time_monitor = simmodel.run_model(team_capacity=team_capacity, report_number=report_number,
                                                interarrival_time_gen=inter_arrival_time_gen,
                                                resolution_time_gen=resolution_time_gen,
                                                priority_gen=priority_gen,
                                                max_time=max_time)

        resolved_issues = resol_time_monitor.count()
        completed_reports.append(resolved_issues)

        avg_resol_time = 0
        if resolved_issues > 0:
            avg_resol_time = resol_time_monitor.mean()
            # print "Completed bug reports: ", resolved_issues, " avg resolution time ", avg_resol_time

    return completed_reports


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    project_key = "CASSANDRA"
    print "Starting analysis for project ", project_key, " ..."

    project_bugs = simdata.filter_by_project(enhanced_dataframe, project_key)
    print "Total issues for project ", project_key, ": ", len(project_bugs.index)

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

    months_in_range = issues_in_range[simdata.CREATED_MONTH_COLUMN].unique()

    team_sizes = []
    period_reports = []

    completed_predicted = []
    completed_true = []
    for year_month in months_in_range:
        issues_for_month = issues_in_range[issues_in_range[simdata.CREATED_MONTH_COLUMN] == year_month]

        bug_resolvers = issues_for_month['JIRA Resolved By']
        team_size = bug_resolvers.nunique()
        reports_per_month = len(issues_for_month.index)

        team_sizes.append(team_size)
        period_reports.append(reports_per_month)

        year, month = year_month.split('-')
        start_date = datetime.datetime(year=int(year), month=int(month), day=1, tzinfo=pytz.utc)
        margin = datetime.timedelta(days=30)
        end_date = start_date + margin

        resolved_issues = simdata.filter_resolved(issues_for_month, only_with_commits=False)
        resolved_in_month = simdata.filter_by_date_range(simdata.RESOLUTION_DATE_COLUMN, resolved_issues, start_date,
                                                         end_date)
        issues_resolved = len(resolved_in_month.index)

        print "Period: ", year_month, " Developers:", team_size, " Reports: ", reports_per_month, " Resolved in Month: ", issues_resolved

        simulation_time = 30 * 24
        alpha = 0.95

        completed_reports = launch_simulation(team_capacity=team_size, report_number=reports_per_month,
                                              inter_arrival_sample=interrival_times_range,
                                              resolution_time_sample=resolution_times,
                                              priority_sample=priorities_in_range, max_time=simulation_time)

        sample_mean, sample_std, sample_size = np.mean(completed_reports), np.std(completed_reports), len(
            completed_reports)
        confidence_interval = stats.norm.interval(alpha, loc=sample_mean, scale=sample_std / np.sqrt(sample_size))
        print "sample_size", sample_size, "sample_mean ", sample_mean, " sample_std ", sample_std, " confidence interval: ", confidence_interval

        completed_predicted.append(sample_mean)
        completed_true.append(issues_resolved)

    # simdata.launch_histogram(period_reports)

    coefficient_of_determination = r2_score(completed_true, completed_predicted)
    print "Simulation finished! coefficient_of_determination ", coefficient_of_determination


if __name__ == "__main__":
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)
