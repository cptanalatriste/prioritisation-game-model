"""
This modules is used to gather payoff values needed for equilibrium calculation.
"""
import pandas as pd

import random

import simdata
import simdriver
import simutils

SEED = 448


def main():
    random.seed(SEED)

    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    project_keys = ["MESOS"]
    valid_reports = simdriver.get_valid_reports(project_keys, enhanced_dataframe)
    periods = valid_reports[simdata.PERIOD_COLUMN].unique()
    reporter_configuration = simdriver.get_reporters_configuration(periods, valid_reports)
    random.shuffle(reporter_configuration)

    split_point = len(reporter_configuration) / 2
    first_reporter_team = reporter_configuration[: split_point]
    second_reporter_team = reporter_configuration[split_point:]

    print "First Team: ", len(first_reporter_team), " reporters "
    print "Second Team: ", len(second_reporter_team), " reporters "

    max_iterations = 100
    simulation_days = 30
    simulation_time = simulation_days * 24
    for period in periods:
        print "Starting simulation for project ", project_keys, " period: ", period
        issues_for_period = valid_reports[valid_reports[simdata.PERIOD_COLUMN] == period]
        reports_per_month = len(issues_for_period.index)

        resolution_time_gen, priority_gen = simdriver.get_simulation_input(valid_reports)
        dev_team_size, issues_resolved, resolved_in_period = simdriver.get_dev_team_production(period,
                                                                                               issues_for_period,
                                                                                               simulation_days)
        print "Reports for period: ", reports_per_month, " Developer Team Size: ", dev_team_size, \
            " Resolved in Period: ", issues_resolved

        completed_per_reporter, completed_per_priority = simutils.launch_simulation(team_capacity=dev_team_size,
                                                                                    report_number=reports_per_month,
                                                                                    reporters_config=reporter_configuration,
                                                                                    resolution_time_gen=resolution_time_gen,
                                                                                    priority_gen=priority_gen,
                                                                                    max_time=simulation_time,
                                                                                    max_iterations=max_iterations)


if __name__ == "__main__":
    main()
