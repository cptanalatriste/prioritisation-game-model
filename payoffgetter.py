"""
This modules is used to gather payoff values needed for equilibrium calculation. Now it is also capable of triggering
gambit and calculating the equilibrium.
"""

from collections import defaultdict
import time
import sys
import winsound
import copy

import pandas as pd

import itertools
from sklearn.cluster import KMeans

import simmodel
import simdata
import simdriver
import simutils
import gtutils

# General game configuration
REDUCING_FACTOR = 1
REPLICATIONS_PER_PROFILE = 100  # On the Walsh paper, they do 2500 replications per profile.

# Payoff function parameters
PRIORITY_SCORING = True
SCORE_MAP = {
    simdata.NON_SEVERE_PRIORITY: 10,
    simdata.NORMAL_PRIORITY: 10 * 2,
    simdata.SEVERE_PRIORITY: 10 * 5
}

N_PLAYERS = None  # None, for not filtering among suitable players.
CLONE_PLAYER = 0  # None to disable cloning
CLONE_MEDIAN = True

# Throtling configuration parameters.
THROTTLING_ENABLED = False
FORCE_PENALTY = None

# Empirical Strategies parameters.
EMPIRICAL_STRATEGIES = False
N_CLUSTERS = 5

HEURISTIC_STRATEGIES = True

# Gatekeeper configuration.
GATEKEEPER_CONFIG = {'review_time': 8,  # False to disable the Gatekeeper on the simulation.
                     'capacity': 1}
GATEKEEPER_CONFIG = False

# Team Configuration
NUMBER_OF_TEAMS = 2
AGGREGATE_AGENT_TEAM = -1


def get_payoff_score(severe_completed, non_severe_completed, normal_completed, priority_based=True):
    """
    Calculates the payoff per user after an specific run.

    :param priority_based: If True, each defect solved has a specific score based on the priority.
    :param severe_completed: Severe bugs resolved.
    :param non_severe_completed: Non Severe bugs resolved.
    :param normal_completed: Normal bugs resolved.
    :return: Payoff score.
    """

    if priority_based:
        score = severe_completed * SCORE_MAP[simdata.SEVERE_PRIORITY] + non_severe_completed * SCORE_MAP[
            simdata.NON_SEVERE_PRIORITY] + normal_completed * SCORE_MAP[simdata.NORMAL_PRIORITY]
    else:
        score = severe_completed + non_severe_completed + normal_completed

    return score


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
            reporter_strategy = reporter_config['strategy']['name']

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

            payoff_score = get_payoff_score(severe_completed, non_severe_completed, normal_completed, PRIORITY_SCORING)

            simulation_results.append({"period": period,
                                       "run": run,
                                       "reporter_name": reporter_name,
                                       "reporter_team": reporter_team,
                                       "reporter_strategy": reporter_strategy,
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


def get_team_metrics(file_prefix, game_period, teams, overall_dataframes):
    """
    Analizes the performance of the team based on fixed issues, according to a scenario description.

    :param teams: Number of teams in the game.
    :param file_prefix: Strategy profile descripcion.
    :param game_period: Game period description.
    :param overall_dataframe: Dataframe with run information.
    :return: List of outputs per team
    """
    runs = overall_dataframes[0]['run'].unique()

    consolidated_result = []

    print "Dataframes under analysis: ", len(overall_dataframes)
    for run in runs:

        team_results = {}
        for team in range(teams):

            for index, overall_dataframe in enumerate(overall_dataframes):
                period_reports = overall_dataframe[overall_dataframe['period'] == game_period]
                reports_in_run = period_reports[period_reports['run'] == run]

                team_run_reports = reports_in_run[reports_in_run['reporter_team'] == team]
                if len(team_run_reports.index) > 0:
                    team_resolved = team_run_reports['reported_completed'].sum()
                    team_reported = team_run_reports['reported'].sum()
                    team_score = team_run_reports['payoff_score'].sum()

                    team_results[team] = {"team_resolved": team_resolved,
                                          "team_reported": team_reported,
                                          "team_score": team_score}

        simulation_result = {"run": run}

        for team_index in range(NUMBER_OF_TEAMS):
            team_prefix = "team_" + str(team_index + 1) + "_"
            simulation_result[team_prefix + "results"] = team_results[team_index]['team_resolved']
            simulation_result[team_prefix + "reports"] = team_results[team_index]['team_reported']
            simulation_result[team_prefix + "score"] = team_results[team_index]['team_score']

        consolidated_result.append(simulation_result)

    consolidated_dataframe = pd.DataFrame(consolidated_result)
    consolidated_dataframe.to_csv("csv/" + file_prefix + "_consolidated_result.csv", index=False)

    team_averages = []
    for team_index in range(NUMBER_OF_TEAMS):
        score_column = "team_" + str(team_index + 1) + "_score"
        team_averages.append(int(consolidated_dataframe[score_column].mean()))

    print "file_prefix: ", file_prefix, " team_averages\t", team_averages, "\n"
    return [str(team_avg) for team_avg in team_averages]


def select_game_players(reporter_configuration, number_of_players=5, clone_player=None):
    """
    Selects which of the players available will be the ones playing the game.
    :param clone_player: The index of the players. whose clones will be returned.
    :param number_of_players: Number of players to be playing the game. In case it is None, no filtering will be performed.
    :param reporter_configuration: List of non drive-by testers.
    :return: List of selected players.
    """
    sorted_reporters = sorted(reporter_configuration,
                              key=lambda config: len(config['interarrival_time_gen'].observations), reverse=True)

    if number_of_players is None:
        number_of_players = len(reporter_configuration)

    if clone_player is not None:
        source_player = [sorted_reporters[clone_player]]
        source_name = source_player[0]['name']

        print "Cloning player at index: ", clone_player, "(", source_name, ")"
        simdriver.fit_reporter_distributions(source_player)

        list_with_clones = []

        for index in range(number_of_players):
            clone = copy.deepcopy(source_player[0])
            clone['name'] = 'clone_' + source_name + '_' + str(index)
            list_with_clones.append(clone)

        return list_with_clones

    return sorted_reporters[:number_of_players]


def select_reporters_for_simulation(reporter_configuration):
    """
    The production of these reporters will be considered for simulation extraction parameters
    :param reporter_configuration: All valid reporters.
    :return: A filtered list of reporters.
    """

    reporters_with_corrections = [config for config in reporter_configuration if
                                  config['with_modified_priority'] > 0]

    print "Original Reporters: ", len(reporter_configuration), " Reporters with corrected priorities: ", len(
        reporters_with_corrections)

    return reporters_with_corrections


def get_heuristic_strategies():
    """
    Returns a set of strategies not related with how users are behaving in data.
    :return:
    """
    return [{'name': simmodel.HONEST_STRATEGY},
            {'name': simmodel.SIMPLE_INFLATE_STRATEGY}]


def get_empirical_strategies(reporter_configuration, n_clusters=3):
    """
    It will group a list of reporters in a predefined number of clusters
    :param reporter_configuration: List of reporter configuration
    :return: The representative strategy per team.
    """

    print "Gathering strategies from reporter behaviour ..."
    print "Original number of reporters: ", len(reporter_configuration)

    reporters_with_corrections = [config for config in reporter_configuration if
                                  config['with_modified_priority'] > 0]
    print "Reporters with corrections: ", len(reporters_with_corrections)

    correction_dataframe = simutils.get_reporter_behavior_dataframe(reporters_with_corrections)
    reporter_dataframe = simutils.get_reporter_behavior_dataframe(reporter_configuration)

    kmeans = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)

    kmeans.fit(correction_dataframe)

    predicted_clusters = kmeans.predict(reporter_dataframe)
    cluster_column = 'cluster'
    reporter_dataframe[cluster_column] = predicted_clusters

    centroids = kmeans.cluster_centers_
    print "Clustering centroids ..."

    strategies_per_team = []
    for index, centroid in enumerate(centroids):
        nonsevere_correction_index = 3
        severe_correction_index = 4
        nonsevere_inflation_index = 0
        severe_deflation_index = 1

        print  " ", simutils.REPORTER_COLUMNS[nonsevere_inflation_index], ": ", "{0:.0f}%".format(
            centroid[nonsevere_inflation_index] * 100), \
            " ", simutils.REPORTER_COLUMNS[severe_deflation_index], ": ", "{0:.0f}%".format(
            centroid[severe_deflation_index] * 100), \
            " ", simutils.REPORTER_COLUMNS[2], ": ", "{0:.0f}%".format(centroid[2] * 100), \
            " ", simutils.REPORTER_COLUMNS[nonsevere_correction_index], ": ", "{0:.0f}%".format(
            centroid[nonsevere_correction_index] * 100), \
            " ", simutils.REPORTER_COLUMNS[severe_correction_index], ": ", "{0:.0f}%".format(
            centroid[severe_correction_index] * 100)

        strategies_per_team.append({'name': 'EMPIRICAL' + str(index),
                                    simutils.NONSEVERE_CORRECTION_COLUMN: centroid[nonsevere_correction_index],
                                    simutils.SEVERE_CORRECTION_COLUMN: centroid[severe_correction_index],
                                    simutils.NON_SEVERE_INFLATED_COLUMN: centroid[nonsevere_inflation_index],
                                    simutils.SEVERE_DEFLATED_COLUMN: centroid[severe_deflation_index]
                                    })

    print "Cluster distribution: \n", reporter_dataframe[cluster_column].value_counts()

    for index, cluster in enumerate(reporter_dataframe[cluster_column].values):
        reporter_configuration[index][cluster_column] = cluster_column

    return strategies_per_team


def get_strategy_map(strategy_list, teams):
    """
    Creates a strategy map, with all the possible strategy profiles on the game.
    :return: A map with all the possible strategy profiles according the players and strategies available.
    """
    strategy_maps = []
    strategy_profiles = list(itertools.product(strategy_list, repeat=teams))

    for profile in strategy_profiles:
        strategy_map = {'name': '',
                        'map': {}}

        # To keep the order preferred by Gambit
        for index, strategy in enumerate(reversed(list(profile))):
            if isinstance(strategy, simmodel.EmpiricalInflationStrategy):
                strategy_name = strategy.name
            else:
                strategy_name = strategy['name']

            strategy_map['name'] += strategy_name + "_"
            strategy_map['map'][index] = strategy

        strategy_map['name'] = strategy_map['name'][:-1]
        strategy_maps.append(strategy_map)

    return strategy_maps


def aggregate_players(agent_team, reporter_configuration):
    """
    Assigns a team to each of the players, peformming player aggregation.
    :param reporter_configuration: Configuration of the selected players.
    :return: Number of teams.
    """

    # We are applying the rationale for a twins player game: For a player C, C represents a single agent in a cluster
    # while C' represents the cluster as an aggregate. This is the one-cluster approach for symmetric games.

    team_attribute = 'team'
    individual_agent_index = 0

    reporter_configuration[individual_agent_index][team_attribute] = agent_team

    for index in range(len(reporter_configuration)):
        if index != individual_agent_index:
            reporter_configuration[index][team_attribute] = AGGREGATE_AGENT_TEAM


def configure_strategies_per_team(player_configuration, strategy_map):
    """
    Assigns the strategies corresponding to teams according to an specific strategy profile.
    :return: Player index whose payoff value is of interest.
    """

    for config in player_configuration:
        config['strategy'] = strategy_map[config['team']]


def get_twins_strategy_map(agent_team, strategy_map):
    """
    Generates a strategy map for a twins reduction simulation.
    :param agent_team: The team that will be assigned to an individual agent.
    :param strategy_map: Original Strategy Map.
    :return:
    """

    twins_strategy_map = strategy_map.copy()

    opponent_strategy = None
    for team_in_copy, strategy_in_copy in twins_strategy_map.iteritems():
        if team_in_copy != agent_team:
            opponent_strategy = strategy_in_copy

    twins_strategy_map[AGGREGATE_AGENT_TEAM] = opponent_strategy
    return twins_strategy_map


def check_simulation_history(overall_dataframes, player_configuration):
    """
    Recycles a previous execution result in case it is consistent with the profile to execute. Specially usefull
    while simulating simetric games.

    :param overall_dataframes: Data from previous simulations.
    :param twins_strategy_map: Strategy map to execute
    :return: Recycled dataframe.
    """

    counter_key = 'counter'
    team_key = 'team'
    strategy_column = 'reporter_strategy'

    strategy_counters = defaultdict(lambda: {counter_key: 0,
                                             team_key: set()})

    for player in player_configuration:
        strategy_name = player['strategy']['name']
        strategy_counters[strategy_name][counter_key] += 1

        strategy_counters[strategy_name][team_key].add(player[team_key])

    for overall_dataframe in overall_dataframes:
        first_run = 0
        single_execution = overall_dataframe[overall_dataframe['run'] == first_run]

        chosen_for_recycling = True
        for strategy, strategy_info in strategy_counters.iteritems():
            strategy_on_dataframe = single_execution[single_execution[strategy_column] == strategy]

            if len(strategy_on_dataframe.index) != strategy_counters[strategy][counter_key]:
                chosen_for_recycling = False

        if chosen_for_recycling:
            recycled_dataframe = overall_dataframe.copy()

            if len(strategy_counters.keys()) == 1:
                filter = (recycled_dataframe[strategy_column] == strategy) & \
                         (recycled_dataframe['reporter_team'] != AGGREGATE_AGENT_TEAM)
                configured_team = None
                for team in strategy_counters[strategy_counters.keys()[0]][team_key]:
                    if team != AGGREGATE_AGENT_TEAM:
                        configured_team = team

                recycled_dataframe.loc[filter, 'reporter_team'] = configured_team
            else:
                for strategy, strategy_info in strategy_counters.iteritems():
                    print "strategy ", strategy, "strategy_info ", strategy_info
                    configured_team = strategy_counters[strategy][team_key].pop()

                    recycled_dataframe.loc[
                        recycled_dataframe[strategy_column] == strategy, 'reporter_team'] = configured_team

            return recycled_dataframe

    return None


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

    print "Original issues ", total_issues, "Reduction Factor: ", REDUCING_FACTOR, " Issues remaining after reduction: ", len(
        enhanced_dataframe.index)

    valid_reports = simdriver.get_valid_reports(project_keys, enhanced_dataframe)
    valid_reporters = simdriver.get_reporter_configuration(valid_reports)
    print "Reporters after drive-in tester removal ...", len(valid_reporters)

    print "Generating elbow-method plot..."
    simutils.elbow_method_for_reporters(valid_reporters)

    empirical_strategies = [simmodel.EmpiricalInflationStrategy(strategy_config=strategy_config) for strategy_config in
                            get_empirical_strategies(valid_reporters, n_clusters=N_CLUSTERS)]

    strategies_catalog = []

    if EMPIRICAL_STRATEGIES:
        strategies_catalog.extend(empirical_strategies)

    if HEURISTIC_STRATEGIES:
        strategies_catalog.extend(get_heuristic_strategies())

    print "strategies_catalog: ", strategies_catalog

    # This are the reporters whose reported bugs will be used to configure the simulation.
    reporter_configuration = select_reporters_for_simulation(valid_reporters)

    # This is the configuration of the actual game players.
    clone_player = CLONE_PLAYER
    if CLONE_MEDIAN:
        clone_player = len(reporter_configuration) / 2

    player_configuration = select_game_players(reporter_configuration, number_of_players=N_PLAYERS,
                                               clone_player=clone_player)

    print "Reporters selected for playing the game ", len(player_configuration)

    if clone_player is None:
        # When cloning is configured, distribution fitting happened before.
        simdriver.fit_reporter_distributions(player_configuration)

    teams = NUMBER_OF_TEAMS
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

    print "Project ", project_keys, " Test Period: ", "ALL", " Reporters: ", test_team_size, " Developers:", dev_team_size, \
        " Reports: ", bugs_by_priority, " Resolved in Period: ", issues_resolved, " Dev Team Bandwith: ", dev_team_bandwith

    simulation_time = sys.maxint

    profile_payoffs = []

    print "Simulation configuration: REPLICATIONS_PER_PROFILE ", REPLICATIONS_PER_PROFILE, " THROTTLING_ENABLED ", \
        THROTTLING_ENABLED, " FORCE_PENALTY ", FORCE_PENALTY, " PRIORITY_SCORING ", PRIORITY_SCORING, \
        " REDUCING_FACTOR ", REDUCING_FACTOR, " EMPIRICAL_STRATEGIES ", EMPIRICAL_STRATEGIES, \
        " HEURISTIC_STRATEGIES ", HEURISTIC_STRATEGIES, " N_CLUSTERS ", N_CLUSTERS, " N_PLAYERS ", N_PLAYERS, \
        " CLONE_PLAYER ", CLONE_PLAYER, " CLONE_MEDIAN ", CLONE_MEDIAN, " GATEKEEPER_CONFIG ", GATEKEEPER_CONFIG

    print "Simulating ", len(strategy_maps), " strategy profiles..."

    simulation_history = []
    for index, map_info in enumerate(strategy_maps):
        file_prefix, strategy_map = map_info['name'], map_info['map']

        overall_dataframes = []

        for team, strategy in strategy_map.iteritems():
            print "Getting payoff for team ", team, " on profile ", file_prefix
            aggregate_players(team, player_configuration)
            twins_strategy_map = get_twins_strategy_map(team, strategy_map)

            configure_strategies_per_team(player_configuration, twins_strategy_map)
            overall_dataframe = check_simulation_history(simulation_history, player_configuration)

            if overall_dataframe is None:
                completed_per_reporter, _, bugs_per_reporter, reports_per_reporter, resolved_per_reporter = simutils.launch_simulation(
                    team_capacity=dev_team_size,
                    bugs_by_priority=bugs_by_priority,
                    reporters_config=player_configuration,
                    resolution_time_gen=resolution_time_gen,
                    max_time=simulation_time,
                    max_iterations=REPLICATIONS_PER_PROFILE,
                    dev_team_bandwidth=dev_team_bandwith,
                    quota_system=THROTTLING_ENABLED,
                    gatekeeper_config=GATEKEEPER_CONFIG)

                simulation_result = consolidate_payoff_results("ALL", player_configuration, completed_per_reporter,
                                                               bugs_per_reporter,
                                                               reports_per_reporter,
                                                               resolved_per_reporter)
                overall_dataframe = pd.DataFrame(simulation_result)
                simulation_history.append(overall_dataframe)

            else:
                print "Profile ", twins_strategy_map, " has being already executed. Recycling dataframe."

            overall_dataframe.to_csv("csv/agent_team_" + str(team) + "_" + file_prefix + '_simulation_results.csv',
                                     index=False)
            overall_dataframes.append(overall_dataframe)

        payoffs = get_team_metrics(str(index) + "-" + file_prefix, "ALL", teams, overall_dataframes)
        profile_payoffs.append((file_prefix, payoffs))

    game_desc = "AS-IS" if not THROTTLING_ENABLED else "THROTTLING"
    game_desc = "GATEKEEPER" if GATEKEEPER_CONFIG else game_desc

    print "Generating Gambit NFG file ..."
    gambit_file = gtutils.get_strategic_game_format(game_desc, player_configuration, strategies_catalog,
                                                    profile_payoffs, teams)

    print "Executing Gambit for equilibrium calculation..."
    gtutils.calculate_equilibrium(player_configuration, strategies_catalog, gambit_file)


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
