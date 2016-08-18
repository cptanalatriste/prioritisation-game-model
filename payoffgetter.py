"""
This modules is used to gather payoff values needed for equilibrium calculation.
"""
import time
import sys
import winsound

import pandas as pd

import random

import simmodel
import simdata
import simdriver
import simutils

SEED = 448

TEAM_PREFIX = "TEAM_"
FIRST_TEAM, SECOND_TEAM, THIRD_TEAM, FORTH_TEAM, FIFTH_TEAM = TEAM_PREFIX + "1", TEAM_PREFIX + "2", TEAM_PREFIX + "3", \
                                                              TEAM_PREFIX + "4", TEAM_PREFIX + "5"


def consolidate_payoff_results(period, reporter_configuration, completed_per_reporter, bugs_per_reporter,
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
                                       "severe_reported": severe_reported,
                                       "non_severe_reported": non_severe_reported,
                                       "normal_reported": normal_reported})

    return simulation_results


def get_team_metrics(file_prefix, game_period, overall_dataframe):
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
        for team in [FIRST_TEAM, SECOND_TEAM, THIRD_TEAM, FORTH_TEAM, FIFTH_TEAM]:
            team_run_reports = reports_in_run[reports_in_run['reporter_team'] == team]

            team_resolved = team_run_reports['reported_completed'].sum()
            team_reported = team_run_reports['reported'].sum()

            team_results[team] = {"team_resolved": team_resolved,
                                  "team_reported": team_reported}

        consolidated_result.append({"run": run,
                                    "team_1_results": team_results[FIRST_TEAM]['team_resolved'],
                                    "team_1_reports": team_results[FIRST_TEAM]['team_reported'],
                                    "team_2_results": team_results[SECOND_TEAM]['team_resolved'],
                                    "team_2_reports": team_results[SECOND_TEAM]['team_reported'],
                                    "team_3_results": team_results[THIRD_TEAM]['team_resolved'],
                                    "team_3_reports": team_results[THIRD_TEAM]['team_reported'],
                                    "team_4_results": team_results[FORTH_TEAM]['team_resolved'],
                                    "team_4_reports": team_results[FORTH_TEAM]['team_reported'],
                                    "team_5_results": team_results[FIFTH_TEAM]['team_resolved'],
                                    "team_5_reports": team_results[FIFTH_TEAM]['team_reported'],
                                    })

    consolidated_dataframe = pd.DataFrame(consolidated_result)
    consolidated_dataframe.to_csv("csv/" + file_prefix + "_consolidated_result.csv", index=False)

    team_1_avg = int(consolidated_dataframe['team_1_results'].mean())
    team_2_avg = int(consolidated_dataframe['team_2_results'].mean())
    team_3_avg = int(consolidated_dataframe['team_3_results'].mean())
    team_4_avg = int(consolidated_dataframe['team_4_results'].mean())
    team_5_avg = int(consolidated_dataframe['team_5_results'].mean())

    print "file_prefix: ", file_prefix, " team_1_avg ", team_1_avg, " team_2_avg ", team_2_avg, " team_3_avg ", team_3_avg, \
        " team_4_avg ", team_4_avg, " team_5_avg ", team_5_avg, " for gambit", ", ".join(
        [str(team_1_avg), str(team_2_avg), str(team_3_avg), str(team_4_avg), str(team_5_avg)])


def select_game_players(reporter_configuration):
    """
    Selects which of the players available will be the ones playing the game.
    :param reporter_configuration: List of non drive-by testers.
    :return:
    """
    sorted_reporters = sorted(reporter_configuration,
                              key=lambda config: len(config['interarrival_time_gen'].observations), reverse=True)
    number_of_players = 5

    return sorted_reporters[:number_of_players]


def main(strategy_maps, enhanced_dataframe, project_keys):
    print "Starting simulation on projects ", project_keys
    total_issues = len(enhanced_dataframe.index)
    reducing_factor = 0.10
    enhanced_dataframe = enhanced_dataframe[:int(total_issues * reducing_factor)]

    print "Original issues ", total_issues, " Issues remaining after reduction: ", len(enhanced_dataframe.index)

    valid_reports = simdriver.get_valid_reports(project_keys, enhanced_dataframe)
    periods = valid_reports[simdata.PERIOD_COLUMN].unique()
    reporter_configuration = simdriver.get_reporter_configuration(periods, valid_reports)

    reporter_configuration = select_game_players(reporter_configuration)
    simdriver.fit_reporter_distributions(reporter_configuration)

    for index, config in enumerate(reporter_configuration):
        config['team'] = TEAM_PREFIX + str(index + 1)

    engaged_testers = [reporter_config['name'] for reporter_config in reporter_configuration]
    valid_reports = simdata.filter_by_reporter(valid_reports, engaged_testers)
    print "Issues in training after reporter filtering: ", len(valid_reports.index)

    max_iterations = 100
    simulation_time = sys.maxint

    print "Starting simulation for project ", project_keys

    reports_for_simulation = len(valid_reports.index)

    resolution_time_gen, priority_gen = simdriver.get_simulation_input("_".join(project_keys), valid_reports)
    dev_team_size, issues_resolved, resolved_in_period, dev_team_bandwith = simdriver.get_dev_team_production(
        valid_reports)

    bug_reporters = valid_reports['Reported By']
    test_team_size = bug_reporters.nunique()

    print "Project ", project_keys, " Test Period: ", "ALL", " Testers: ", test_team_size, " Developers:", dev_team_size, \
        " Reports: ", reports_for_simulation, " Resolved in Period: ", issues_resolved, " Dev Team Bandwith: ", dev_team_bandwith

    for map_info in strategy_maps:

        file_prefix, strategy_map = map_info['name'], map_info['map']

        for config in reporter_configuration:
            config['strategy'] = strategy_map[config['team']]

        print "Launching simulation of resolved issues for profile ", file_prefix
        completed_per_reporter, completed_per_priority, bugs_per_reporter, reports_per_reporter, reported_per_priotity = simutils.launch_simulation(
            team_capacity=dev_team_size,
            report_number=reports_for_simulation,
            reporters_config=reporter_configuration,
            resolution_time_gen=resolution_time_gen,
            priority_gen=priority_gen,
            max_time=simulation_time,
            max_iterations=max_iterations,
            dev_team_bandwith=dev_team_bandwith,
            quota_system=True)

        print "Simulation ended"

        simulation_result = consolidate_payoff_results("ALL", reporter_configuration, completed_per_reporter,
                                                       bugs_per_reporter,
                                                       reports_per_reporter)

        overall_dataframe = pd.DataFrame(simulation_result)
        overall_dataframe.to_csv("csv/" + file_prefix + '_simulation_results.csv', index=False)

        get_team_metrics(file_prefix, "ALL", overall_dataframe)


if __name__ == "__main__":
    start_time = time.time()
    try:
        random.seed(SEED)

        print "Loading information from ", simdata.ALL_ISSUES_CSV
        all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

        print "Adding calculated fields..."
        enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

        valid_projects = simdriver.get_valid_projects(enhanced_dataframe)

        strategy_profiles_1 = [{'name': 'INFLATE_INFLATE_INFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_INFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_INFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_INFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_INFLATE_NOTINFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_NOTINFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_NOTINFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_NOTINFLATE_INFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_INFLATE_INFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_INFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}}
                               ]

        strategy_profiles_2 = [{'name': 'INFLATE_INFLATE_NOTINFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_INFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_INFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_NOTINFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE_INFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_INFLATE_INFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_INFLATE_NOTINFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_INFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}}

                               ]

        strategy_profiles_3 = [{'name': 'INFLATE_INFLATE_INFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_INFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_INFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_NOTINFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_INFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_NOTINFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_INFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'NOTINFLATE_NOTINFLATE_INFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_NOTINFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_INFLATE_NOTINFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}},
                               {'name': 'INFLATE_NOTINFLATE_INFLATE_INFLATE_NOTINFLATE',
                                'map': {FIRST_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        SECOND_TEAM: simmodel.NOT_INFLATE_STRATEGY,
                                        THIRD_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FORTH_TEAM: simmodel.SIMPLE_INFLATE_STRATEGY,
                                        FIFTH_TEAM: simmodel.NOT_INFLATE_STRATEGY}}
                               ]

        strategy_profiles = strategy_profiles_1 + strategy_profiles_2 + strategy_profiles_3
        main(strategy_profiles, enhanced_dataframe, valid_projects)

    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
