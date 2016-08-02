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


def consolidate_results(period, reporter_configuration, completed_per_reporter, bugs_per_reporter,
                        reports_per_reporter):
    """
    Gather per-run metrics according to a simulation result.
    :param period: Description of the period.
    :param reporter_configuration: List of reporter configuration.
    :param completed_per_reporter: List containing completed reports per reporter per run.
    :param bugs_per_reporter: List containing found reports per reporter per priority per run.
    :param reports_per_reporter: ist containing reported (sic) reports per reporter per priority per run.
    :return: Consolidated metrics in a list.
    """
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
                                       "reporter_team": reporter_team,
                                       "reported": severe_reported + non_severe_reported + normal_reported,
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


def assign_team(reporter_configuration, team_value, strategy_map):
    """
    Assigns teams to a reporter config list, including a strategy per team.
    :param reporter_configuration: List of reporter configuration in the team.
    :param team_value: Team name.
    :param strategy_map: Strategies per team.
    :return: None.
    """
    for config in reporter_configuration:
        config['team'] = team_value
        config['strategy'] = strategy_map[team_value]


def get_avg_priority(team_normal_found, team_severe_found, team_nonsevere_found):
    """
    Returns the average priority of a bug report batch.
    :param team_normal_found: Normal issues in batch
    :param team_severe_found: Severe issues in batch
    :param team_nonsevere_found: Non-Severe issues in batch
    :return: Average priority.
    """
    denominator = team_normal_found + team_severe_found + team_nonsevere_found

    if denominator == 0:
        return None

    team_avg_priority = (
                            team_normal_found * simdata.NORMAL_PRIORITY + team_severe_found * simdata.SEVERE_PRIORITY + team_nonsevere_found * simdata.NON_SEVERE_PRIORITY) / float(
        denominator)

    return team_avg_priority


def get_team_metrics(file_prefix, game_period, overall_dataframe, optimal_threshold):
    """
    Analises the performance of the team based on fixed issues, according to a scenario description.

    :param file_prefix: Strategy profile descripcion.
    :param game_period: Game period description.
    :param overall_dataframe: Dataframe with run information.
    :param optimal_threshold: Threshold, for the definition of optimal and suboptimal scenarios.
    :return: None
    """
    runs = overall_dataframe['run'].unique()
    period_reports = overall_dataframe[overall_dataframe['period'] == game_period]

    consolidated_result = []
    for run in runs:
        reports_in_run = period_reports[period_reports['run'] == run]

        team_results = {}
        for team in [FIRST_TEAM, SECOND_TEAM]:
            team_run_reports = reports_in_run[reports_in_run['reporter_team'] == team]

            team_resolved = team_run_reports['reported_completed'].sum()
            team_reported = team_run_reports['reported'].sum()

            team_normal_found = team_run_reports['normal_found'].sum()
            team_severe_found = team_run_reports['reported_completed'].sum()
            team_nonsevere_found = team_run_reports['severe_found'].sum()

            team_avg_priority = get_avg_priority(team_normal_found, team_severe_found, team_nonsevere_found)

            batch_quality = "OPTIMAL"
            if team_avg_priority < optimal_threshold:
                batch_quality = "SUB_OPTIMAL"

            team_results[team] = {"team_resolved": team_resolved,
                                  "team_reported": team_reported,
                                  "batch_quality": batch_quality}

        consolidated_result.append({"run": run,
                                    "team_1_quality": team_results[FIRST_TEAM]['batch_quality'],
                                    "team_1_results": team_results[FIRST_TEAM]['team_resolved'],
                                    "team_1_reports": team_results[FIRST_TEAM]['team_reported'],
                                    "team_2_quality": team_results[SECOND_TEAM]['batch_quality'],
                                    "team_2_results": team_results[SECOND_TEAM]['team_resolved'],
                                    "team_2_reports": team_results[SECOND_TEAM]['team_reported'],
                                    "scenario": team_results[FIRST_TEAM]['batch_quality'] + "-" +
                                                team_results[SECOND_TEAM]['batch_quality']
                                    })

    consolidated_dataframe = pd.DataFrame(consolidated_result)
    scenarios = consolidated_dataframe["scenario"].value_counts(normalize=True)
    # print "scenarios: \n", scenarios

    for scenario, frecuency in scenarios.iteritems():
        scenario_reports = consolidated_dataframe[consolidated_dataframe["scenario"] == scenario]

        team_1_results = scenario_reports["team_1_results"]
        team_2_results = scenario_reports["team_2_results"]

        team_1_resolved = team_1_results.median()
        team_2_resolved = team_2_results.median()

        team_1_reports = scenario_reports["team_1_reports"]
        team_2_reports = scenario_reports["team_2_reports"]

        dev_fix_ratio = (team_1_results.sum() + team_2_results.sum()) / float(
            team_1_reports.sum() + team_2_reports.sum())

        print "scenario: ", scenario, "profile ", file_prefix, " frecuency: ", frecuency, "counts: ", len(
            scenario_reports.index), "team_1_resolved ", team_1_resolved, " team_2_resolved ", team_2_resolved, \
            " dev_fix_ratio ", dev_fix_ratio

    consolidated_dataframe.to_csv("csv/" + file_prefix + "_consolidated_result.csv", index=False)


def main(file_prefix, strategy_map, enhanced_dataframe, project_keys):
    print "Starting ", file_prefix, " on project ", project_keys

    valid_reports = simdriver.get_valid_reports(project_keys, enhanced_dataframe)
    periods = valid_reports[simdata.PERIOD_COLUMN].unique()
    reporter_configuration = simdriver.get_reporters_configuration(periods, valid_reports)
    random.shuffle(reporter_configuration)

    split_point = len(reporter_configuration) / 2

    assign_team(reporter_configuration[: split_point], FIRST_TEAM, strategy_map)
    assign_team(reporter_configuration[split_point:], SECOND_TEAM, strategy_map)

    engaged_testers = [reporter_config['name'] for reporter_config in reporter_configuration]
    valid_reports = simdata.filter_by_reporter(valid_reports, engaged_testers)
    print "Issues in training after reporter filtering: ", len(valid_reports.index)

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

        gatekeeper_config = {'review_time': 2,
                             'capacity': 1,
                             'throttling': True}

        gatekeeper_config = False
        #
        # gatekeeper_config = {'review_time': 2,
        #                      'capacity': 1,
        #                      'throttling': False}

        completed_per_reporter, completed_per_priority, bugs_per_reporter, reports_per_reporter = simutils.launch_simulation(
            team_capacity=dev_team_size,
            report_number=reports_per_month,
            reporters_config=reporter_configuration,
            resolution_time_gen=resolution_time_gen,
            priority_gen=priority_gen,
            max_time=simulation_time,
            max_iterations=max_iterations,
            gatekeeper_config=gatekeeper_config)

        results = consolidate_results(period, reporter_configuration, completed_per_reporter, bugs_per_reporter,
                                      reports_per_reporter)
        overall_results.extend(results)

    overall_dataframe = pd.DataFrame(overall_results)
    overall_dataframe.to_csv("csv/" + file_prefix + '_simulation_results.csv', index=False)

    optimal_threshold = overall_dataframe["avg_priority_found"].mean()
    print "Mean Batch Priority: ", optimal_threshold

    mean_reported = overall_dataframe["avg_priority_reported"].mean()
    print "Mean Reported Priority: ", mean_reported

    get_team_metrics(file_prefix, game_period, overall_dataframe, optimal_threshold)


if __name__ == "__main__":
    start_time = time.time()
    try:
        random.seed(SEED)

        print "Loading information from ", simdata.ALL_ISSUES_CSV
        all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

        print "Adding calculated fields..."
        enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

        project_keys = ["MESOS"]

        strategy_profiles = [{'name': 'INFLATE_NOTINFLATE',
                              'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                      SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                             {'name': 'NOTINFLATE_NOTINFLATE',
                              'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                      SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                             {'name': 'INFLATE_INFLATE',
                              'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                      SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                             {'name': 'NOTINFLATE_INFLATE',
                              'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                      SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}}
                             ]

        for profile in strategy_profiles:
            main(profile['name'], profile['map'], enhanced_dataframe, project_keys)
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
