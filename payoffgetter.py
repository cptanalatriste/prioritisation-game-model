"""
This modules is used to gather payoff values needed for equilibrium calculation.
"""
import time
import sys
import winsound

import pandas as pd

import itertools
from sklearn.cluster import KMeans

import simmodel
import simdata
import simdriver
import simutils

REDUCING_FACTOR = 1.0
FIRST_TEAM, SECOND_TEAM, THIRD_TEAM, FORTH_TEAM, FIFTH_TEAM = 0, 1, 2, 3, 4


def get_payoff_score(severe_completed, non_severe_completed, normal_completed):
    """
    Calculates the payoff per user after an specific run.
    :param severe_completed:
    :param non_severe_completed:
    :param normal_completed:
    :return:
    """
    score_map = {
        simdata.NON_SEVERE_PRIORITY: 10,
        simdata.NORMAL_PRIORITY: 10 * 2,
        simdata.SEVERE_PRIORITY: 10 * 5
    }

    return severe_completed * score_map[simdata.SEVERE_PRIORITY] + non_severe_completed * score_map[
        simdata.NON_SEVERE_PRIORITY] + normal_completed * score_map[simdata.NORMAL_PRIORITY]


def consolidate_payoff_results(period, reporter_configuration, completed_per_reporter, bugs_per_reporter,
                               reports_per_reporter, resolved_per_reporter):
    """
    Gather per-run metrics according to a simulation result.
    :param resolved_per_reporter: Resolved issues per priority, including a priority detail.
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
        run_resolved_priority = resolved_per_reporter[run]

        for reporter_config in reporter_configuration:
            reporter_name = reporter_config['name']
            reporter_team = reporter_config['team']

            reported_completed = run_resolved[reporter_name]

            severe_completed = run_resolved_priority[reporter_name][simdata.SEVERE_PRIORITY]
            non_severe_completed = run_resolved_priority[reporter_name][simdata.NON_SEVERE_PRIORITY]
            normal_completed = run_resolved_priority[reporter_name][simdata.NORMAL_PRIORITY]

            severe_found = run_found[reporter_name][simdata.SEVERE_PRIORITY]
            non_severe_found = run_found[reporter_name][simdata.NON_SEVERE_PRIORITY]
            normal_found = run_found[reporter_name][simdata.NORMAL_PRIORITY]

            severe_reported = run_reported[reporter_name][simdata.SEVERE_PRIORITY]
            non_severe_reported = run_reported[reporter_name][simdata.NON_SEVERE_PRIORITY]
            normal_reported = run_reported[reporter_name][simdata.NORMAL_PRIORITY]

            payoff_score = get_payoff_score(severe_completed, non_severe_completed, normal_completed)

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
                                       "normal_reported": normal_reported,
                                       "severe_completed": severe_completed,
                                       "non_severe_completed": non_severe_completed,
                                       "normal_completed": normal_completed,
                                       "payoff_score": payoff_score})

    return simulation_results


def get_team_metrics(file_prefix, game_period, teams, overall_dataframe):
    """
    Analises the performance of the team based on fixed issues, according to a scenario description.

    :param teams: Number of teams in the game.
    :param file_prefix: Strategy profile descripcion.
    :param game_period: Game period description.
    :param overall_dataframe: Dataframe with run information.
    :return: List of outputs per team
    """
    runs = overall_dataframe['run'].unique()
    period_reports = overall_dataframe[overall_dataframe['period'] == game_period]

    consolidated_result = []
    for run in runs:
        reports_in_run = period_reports[period_reports['run'] == run]

        team_results = {}
        for team in range(teams):
            team_run_reports = reports_in_run[reports_in_run['reporter_team'] == team]

            team_resolved = team_run_reports['reported_completed'].sum()
            team_reported = team_run_reports['reported'].sum()
            team_score = team_run_reports['payoff_score'].sum()

            team_results[team] = {"team_resolved": team_resolved,
                                  "team_reported": team_reported,
                                  "team_score": team_score}

        consolidated_result.append({"run": run,
                                    "team_1_results": team_results[FIRST_TEAM]['team_resolved'],
                                    "team_1_reports": team_results[FIRST_TEAM]['team_reported'],
                                    "team_1_score": team_results[FIRST_TEAM]['team_score'],
                                    "team_2_results": team_results[SECOND_TEAM]['team_resolved'],
                                    "team_2_reports": team_results[SECOND_TEAM]['team_reported'],
                                    "team_2_score": team_results[SECOND_TEAM]['team_score'],
                                    "team_3_results": team_results[THIRD_TEAM]['team_resolved'],
                                    "team_3_reports": team_results[THIRD_TEAM]['team_reported'],
                                    "team_3_score": team_results[THIRD_TEAM]['team_score'],
                                    "team_4_results": team_results[FORTH_TEAM]['team_resolved'],
                                    "team_4_reports": team_results[FORTH_TEAM]['team_reported'],
                                    "team_4_score": team_results[FORTH_TEAM]['team_score'],
                                    "team_5_results": team_results[FIFTH_TEAM]['team_resolved'],
                                    "team_5_reports": team_results[FIFTH_TEAM]['team_reported'],
                                    "team_5_score": team_results[FIFTH_TEAM]['team_score']
                                    })

    consolidated_dataframe = pd.DataFrame(consolidated_result)
    consolidated_dataframe.to_csv("csv/" + file_prefix + "_consolidated_result.csv", index=False)

    team_1_avg = int(consolidated_dataframe['team_1_score'].mean())
    team_2_avg = int(consolidated_dataframe['team_2_score'].mean())
    team_3_avg = int(consolidated_dataframe['team_3_score'].mean())
    team_4_avg = int(consolidated_dataframe['team_4_score'].mean())
    team_5_avg = int(consolidated_dataframe['team_5_score'].mean())

    print "file_prefix: ", file_prefix, " team_1_avg\t", team_1_avg, " team_2_avg\t", team_2_avg, " team_3_avg\t", team_3_avg, \
        " team_4_avg\t", team_4_avg, " team_5_avg\t", team_5_avg

    return [str(team_1_avg), str(team_2_avg), str(team_3_avg), str(team_4_avg), str(team_5_avg)]


def select_game_players(reporter_configuration, number_of_players=5):
    """
    Selects which of the players available will be the ones playing the game.
    :param reporter_configuration: List of non drive-by testers.
    :return: List of selected players.
    """
    sorted_reporters = sorted(reporter_configuration,
                              key=lambda config: len(config['interarrival_time_gen'].observations), reverse=True)

    return sorted_reporters[:number_of_players]


def get_strategy_catalog(reporter_configuration, n_clusters=3):
    """
    It will group a list of reporters in a predefined number of clusters
    :param reporter_configuration: List of reporter configuration
    :return: The representative strategy per team.
    """

    print "Gathering strategies from reporter behaviour ..."
    reporter_dataframe = simutils.get_reporter_behavior_dataframe(reporter_configuration)

    kmeans = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)

    predicted_clusters = kmeans.fit_predict(reporter_dataframe)
    cluster_column = 'cluster'
    reporter_dataframe[cluster_column] = predicted_clusters

    centroids = kmeans.cluster_centers_
    print "Clustering centroids ..."

    strategies_per_team = []
    for index, centroid in enumerate(centroids):
        nonsevere_correction_index = 5
        severe_correction_index = 6
        nonsevere_inflation_index = 1
        severe_deflation_index = 3

        print " ", simutils.REPORTER_COLUMNS[0], ": ", "{0:.0f}%".format(centroid[0] * 100), \
            " ", simutils.REPORTER_COLUMNS[nonsevere_inflation_index], ": ", "{0:.0f}%".format(
            centroid[nonsevere_inflation_index] * 100), \
            " ", simutils.REPORTER_COLUMNS[2], ": ", "{0:.0f}%".format(centroid[2]), \
            " ", simutils.REPORTER_COLUMNS[severe_deflation_index], ": ", "{0:.0f}%".format(
            centroid[severe_deflation_index] * 100), \
            " ", simutils.REPORTER_COLUMNS[4], ": ", "{0:.0f}%".format(centroid[4] * 100), \
            " ", simutils.REPORTER_COLUMNS[nonsevere_correction_index], ": ", "{0:.0f}%".format(
            centroid[nonsevere_correction_index] * 100), \
            " ", simutils.REPORTER_COLUMNS[severe_correction_index], ": ", "{0:.0f}%".format(
            centroid[severe_correction_index] * 100)

        strategies_per_team.append({'name': 'Centroid_' + str(index),
                                    simutils.NONSEVERE_CORRECTION_COLUMN: centroid[nonsevere_correction_index],
                                    simutils.SEVERE_CORRECTION_COLUMN: centroid[severe_correction_index],
                                    simutils.NON_SEVERE_INFLATED_COLUMN: centroid[nonsevere_inflation_index],
                                    simutils.SEVERE_DEFLATED_COLUMN: centroid[severe_deflation_index]
                                    })

    print "Cluster distribution: \n", reporter_dataframe[cluster_column].value_counts()

    for index, cluster in enumerate(reporter_dataframe[cluster_column].values):
        reporter_configuration[index][cluster_column] = cluster_column

    return strategies_per_team


def get_strategy_map(strategies_per_team, teams):
    """
    Creates a strategy map, with all the possible strategy profiles on the game.
    :return:
    """
    strategy_maps = []

    strategy_list = [simmodel.EmpiricalInflationStrategy(strategy_config=strategy_config) for strategy_config in
                     strategies_per_team]
    strategy_profiles = list(itertools.product(strategy_list, repeat=teams))

    for profile in strategy_profiles:
        strategy_map = {'name': '',
                        'map': {}}

        # To keep the order preferred by Gambit

        for index, strategy in enumerate(reversed(list(profile))):
            strategy_map['name'] += strategy.strategy_config['name'] + "_"
            strategy_map['map'][index] = strategy

        strategy_maps.append(strategy_map)

    return strategy_maps


def group_in_teams(reporter_configuration):
    """
    Assigns a team to each of the players
    :param reporter_configuration: Configuration of the selected players.
    :return: Number of teams.
    """
    for index, config in enumerate(reporter_configuration):
        config['team'] = index

    return len(reporter_configuration)


def start_payoff_calculation(enhanced_dataframe, project_keys):
    """
    Given a strategy profile list, calculates payoffs per player thorugh simulation.
    :param enhanced_dataframe: Report data to gather simulation input.
    :param project_keys: Projects to be considered.
    :return: Payoffss per player per profile.
    """
    print "Starting simulation on projects ", project_keys
    total_issues = len(enhanced_dataframe.index)
    enhanced_dataframe = enhanced_dataframe[:int(total_issues * REDUCING_FACTOR)]

    print "Original issues ", total_issues, " Issues remaining after reduction: ", len(enhanced_dataframe.index)

    valid_reports = simdriver.get_valid_reports(project_keys, enhanced_dataframe)
    reporter_configuration = simdriver.get_reporter_configuration(valid_reports)
    print "Reporters after drive-in tester removal ..."

    print "Generating elbow-method plot..."
    simutils.elbow_method_for_reporters(reporter_configuration)

    clusters = 2
    strategies_catalog = get_strategy_catalog(reporter_configuration, n_clusters=clusters)

    reporter_configuration = select_game_players(reporter_configuration)
    print "Reporters selected for playing the game ", len(reporter_configuration)
    simdriver.fit_reporter_distributions(reporter_configuration)

    teams = group_in_teams(reporter_configuration)
    strategy_maps = get_strategy_map(strategies_catalog, teams)

    engaged_testers = [reporter_config['name'] for reporter_config in reporter_configuration]
    valid_reports = simdata.filter_by_reporter(valid_reports, engaged_testers)
    print "Issues in training after reporter filtering: ", len(valid_reports.index)

    print "Starting simulation for project ", project_keys

    bugs_by_priority = {index: value
                        for index, value in
                        valid_reports[simdata.SIMPLE_PRIORITY_COLUMN].value_counts().iteritems()}

    resolution_time_gen = simdriver.get_simulation_input(training_issues=valid_reports)
    dev_team_size, issues_resolved, resolved_in_period, dev_team_bandwith = simdriver.get_dev_team_production(
        valid_reports)

    bug_reporters = valid_reports['Reported By']
    test_team_size = bug_reporters.nunique()

    print "Project ", project_keys, " Test Period: ", "ALL", " Testers: ", test_team_size, " Developers:", dev_team_size, \
        " Reports: ", bugs_by_priority, " Resolved in Period: ", issues_resolved, " Dev Team Bandwith: ", dev_team_bandwith

    max_iterations = 30
    simulation_time = sys.maxint
    quota_system = True

    gambit_lines = []

    print "Simulation configuration: max_iterations ", max_iterations, " quota_system ", quota_system

    for index, map_info in enumerate(strategy_maps):

        file_prefix, strategy_map = map_info['name'], map_info['map']

        for config in reporter_configuration:
            config['strategy'] = strategy_map[config['team']]

        completed_per_reporter, _, bugs_per_reporter, reports_per_reporter, resolved_per_reporter = simutils.launch_simulation(
            team_capacity=dev_team_size,
            bugs_by_priority=bugs_by_priority,
            reporters_config=reporter_configuration,
            resolution_time_gen=resolution_time_gen,
            max_time=simulation_time,
            max_iterations=max_iterations,
            dev_team_bandwidth=dev_team_bandwith,
            quota_system=quota_system)

        simulation_result = consolidate_payoff_results("ALL", reporter_configuration, completed_per_reporter,
                                                       bugs_per_reporter,
                                                       reports_per_reporter,
                                                       resolved_per_reporter)

        overall_dataframe = pd.DataFrame(simulation_result)
        overall_dataframe.to_csv("csv/" + file_prefix + '_simulation_results.csv', index=False)

        payoffs = get_team_metrics(str(index) + "-" + file_prefix, "ALL", teams, overall_dataframe)
        gambit_line = '{ "' + file_prefix + '" ' + ",".join(payoffs) + " }"
        gambit_lines.append(gambit_line)

    print "Profiles for Gambit: \n"
    print "\n".join(gambit_lines)


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    valid_projects = simdriver.get_valid_projects(enhanced_dataframe)
    start_payoff_calculation(enhanced_dataframe, valid_projects)


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
