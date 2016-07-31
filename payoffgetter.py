"""
This modules is used to gather payoff values needed for equilibrium calculation.
"""
import time
import winsound

import pandas as pd

import random

import simmodel
import simdata
import simdriver
import simutils

SEED = 448

FIRST_TEAM, SECOND_TEAM = "TEAM_1", "TEAM_2"
STRATEGY_MAP = {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY}


def consolidate_results(period, reporter_configuration, completed_per_reporter, bugs_per_reporter,
                        reports_per_reporter):
    if len(completed_per_reporter) != len(bugs_per_reporter):
        raise Exception("The output of the simulation doesn't match!")

    simulation_results = []

    for run in range(len(completed_per_reporter)):
        run_resolved = completed_per_reporter[run]
        run_found = bugs_per_reporter[run]
        run_reported = reports_per_reporter[run]

        for reporter_config in reporter_configuration:
            reporter_name = reporter_config['name']
            reporter_team = reporter_config['team']

            reported_completed = run_resolved[reporter_name]

            severe_found = run_found[reporter_name][simdata.SEVERE_PRIORITY]
            non_severe_found = run_found[reporter_name][simdata.NON_SEVERE_PRIORITY]
            normal_found = run_found[reporter_name][simdata.NORMAL_PRIORITY]

            severe_reported = run_reported[reporter_name][simdata.SEVERE_PRIORITY]
            non_severe_reported = run_reported[reporter_name][simdata.NON_SEVERE_PRIORITY]
            normal_reported = run_reported[reporter_name][simdata.NORMAL_PRIORITY]

            simulation_results.append({"period": period,
                                       "run": run,
                                       "reporter_name": reporter_name,
                                       'reporter_team': reporter_team,
                                       "reported_completed": reported_completed,
                                       "severe_found": severe_found,
                                       "non_severe_found": non_severe_found,
                                       "normal_found": normal_found,
                                       "avg_priority_found": get_avg_priority(normal_found, severe_found,
                                                                              non_severe_found),
                                       "severe_reported": severe_reported,
                                       "non_severe_reported": non_severe_reported,
                                       "normal_reported": normal_reported,
                                       "avg_priority_reported": get_avg_priority(normal_reported, severe_reported,
                                                                                 non_severe_reported)})

    return simulation_results


def assign_team(reporter_configuration, team_value):
    for config in reporter_configuration:
        config['team'] = team_value
        config['strategy'] = STRATEGY_MAP[team_value]


def get_avg_priority(team_normal_found, team_severe_found, team_nonsevere_found):
    denominator = team_normal_found + team_severe_found + team_nonsevere_found

    if denominator == 0:
        return None

    team_avg_priority = (
                            team_normal_found * simdata.NORMAL_PRIORITY + team_severe_found * simdata.SEVERE_PRIORITY + team_nonsevere_found * simdata.NON_SEVERE_PRIORITY) / float(
        denominator)

    return team_avg_priority


def get_team_metrics(game_period, overall_dataframe, optimal_threshold):
    runs = overall_dataframe['run'].unique()
    period_reports = overall_dataframe[overall_dataframe['period'] == game_period]

    consolidated_result = []
    for run in runs:
        reports_in_run = period_reports[period_reports['run'] == run]

        team_results = {}
        for team in [FIRST_TEAM, SECOND_TEAM]:
            team_run_reports = reports_in_run[reports_in_run['reporter_team'] == team]

            team_resolved = team_run_reports['reported_completed'].sum()
            team_normal_found = team_run_reports['normal_found'].sum()
            team_severe_found = team_run_reports['reported_completed'].sum()
            team_nonsevere_found = team_run_reports['severe_found'].sum()

            team_avg_priority = get_avg_priority(team_normal_found, team_severe_found, team_nonsevere_found)

            batch_quality = "OPTIMAL"
            if team_avg_priority < optimal_threshold:
                batch_quality = "SUB_OPTIMAL"

            team_results[team] = {"team_resolved": team_resolved,
                                  "batch_quality": batch_quality}

        consolidated_result.append({"run": run,
                                    "team_1_quality": team_results[FIRST_TEAM]['batch_quality'],
                                    "team_1_results": team_results[FIRST_TEAM]['team_resolved'],
                                    "team_2_quality": team_results[SECOND_TEAM]['batch_quality'],
                                    "team_2_results": team_results[SECOND_TEAM]['team_resolved'],
                                    "scenario": team_results[FIRST_TEAM]['batch_quality'] + "-" +
                                                team_results[SECOND_TEAM]['batch_quality']
                                    })

    consolidated_dataframe = pd.DataFrame(consolidated_result)
    scenarios = consolidated_dataframe["scenario"].value_counts(normalize=True)
    # print "scenarios: \n", scenarios

    for scenario, frecuency in scenarios.iteritems():
        print "scenario: ", scenario, " frecuency: ", frecuency
        scenario_reports = consolidated_dataframe[consolidated_dataframe["scenario"] == scenario]
        print "counts: ", len(scenario_reports.index)

        team_1_resolved = scenario_reports["team_1_results"].mean()
        team_2_resolved = scenario_reports["team_2_results"].mean()

        print "team_1_resolved ", team_1_resolved, " team_2_resolved ", team_2_resolved

    consolidated_dataframe.to_csv("consolidated_result.csv", index=False)


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

    assign_team(reporter_configuration[: split_point], FIRST_TEAM)
    assign_team(reporter_configuration[split_point:], SECOND_TEAM)

    max_iterations = 10000
    simulation_days = 30
    simulation_time = simulation_days * 24

    overall_results = []

    # Only for testing
    game_period = "2015-03"
    periods = [game_period]
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

        completed_per_reporter, completed_per_priority, bugs_per_reporter, reports_per_reporter = simutils.launch_simulation(
            team_capacity=dev_team_size,
            report_number=reports_per_month,
            reporters_config=reporter_configuration,
            resolution_time_gen=resolution_time_gen,
            priority_gen=priority_gen,
            max_time=simulation_time,
            max_iterations=max_iterations)

        results = consolidate_results(period, reporter_configuration, completed_per_reporter, bugs_per_reporter,
                                      reports_per_reporter)
        overall_results.extend(results)

    overall_dataframe = pd.DataFrame(overall_results)
    overall_dataframe.to_csv('simulation_results.csv', index=False)

    optimal_threshold = overall_dataframe["avg_priority_found"].mean()
    print "Mean Batch Priority: ", optimal_threshold
    get_team_metrics(game_period, overall_dataframe, optimal_threshold)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
