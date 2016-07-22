"""
This modules triggers the bug report simulation.
"""
import pandas as pd
import simdata
import simutils
import simmodel
import numpy as np

import winsound


def launch_simulation(team_capacity, report_number, inter_arrival_sample, resolution_time_sample, priority_sample,
                      max_time):
    random_seeds = [393939, 31555999, 777999555, 319999771]

    inter_arrival_time_gen = simutils.ContinuousEmpiricalDistribution(inter_arrival_sample)
    resolution_time_gen = simutils.ContinuousEmpiricalDistribution(resolution_time_sample)
    priority_gen = simutils.DiscreteEmpiricalDistribution(priority_sample)

    for a_seed in random_seeds:
        np.random.seed(a_seed)
        wait_monitor = simmodel.run_model(team_capacity=team_capacity, report_number=report_number,
                                          interarrival_time_gen=inter_arrival_time_gen,
                                          resolution_time_gen=resolution_time_gen,
                                          priority_gen=priority_gen,
                                          max_time=max_time)
        print "Average wait for ", wait_monitor.count(), " completitions is ", wait_monitor.mean()


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    bug_reports = simdata.enhace_report_dataframe(all_issues)

    project_key = "CASSANDRA"
    print "Starting analysis for project ", project_key, " ..."

    project_bugs = simdata.filter_by_project(bug_reports, project_key)
    print "Total issues for project ", project_key, ": ", len(project_bugs.index)

    with_corrected_priority = simdata.get_modified_priority_bugs(project_bugs)
    min_create_date = with_corrected_priority[simdata.CREATED_DATE_COLUMN].min()
    max_create_date = with_corrected_priority[simdata.CREATED_DATE_COLUMN].max()

    print "With corrected priorities: ", len(
        with_corrected_priority.index), " between ", min_create_date, " and ", max_create_date

    issues_in_range = simdata.filter_by_create_date(bug_reports, min_create_date, max_create_date)
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
    for month in months_in_range:
        issues_for_month = issues_in_range[issues_in_range[simdata.CREATED_MONTH_COLUMN] == month]

        bug_resolvers = issues_for_month['JIRA Resolved By']
        team_size = bug_resolvers.nunique()
        reports_per_month = len(issues_for_month.index)

        team_sizes.append(team_size)
        period_reports.append(reports_per_month)
        print "Period: ", month, " Developers:", team_size, " Reports: ", reports_per_month

    simulation_time = 30 * 24

    print "Launching simulation ..."
    launch_simulation(team_capacity=team_sizes[0], report_number=period_reports[0],
                      inter_arrival_sample=interrival_times_range, resolution_time_sample=resolution_times,
                      priority_sample=priorities_in_range, max_time=simulation_time)

    print "Simulation finished!"


if __name__ == "__main__":
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)
