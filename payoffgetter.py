"""
This modules is used to gather payoff values needed for equilibrium calculation. Now it is also capable of triggering
gambit and calculating the equilibrium.

The execution of this module triggers the equilibrium experiments for unsupervised prioritizaton.
Experiment parameters can be tuned in gtconfig.py.

"""
import logging
import time
import sys

from recordtype import recordtype
import pandas as pd

import itertools
from sklearn.cluster import KMeans

import simmodel
import simtwins
import simdata
import simdriver
import simutils
import simcruncher
import gtutils
import gtconfig

if gtconfig.is_windows:
    import winsound

DEFAULT_CONFIGURATION = {
    # General game configuration
    'REPLICATIONS_PER_PROFILE': 30,  # On the Walsh paper, they do 2500 replications per profile.

    # Payoff function parameters
    'PRIORITY_SCORING': True,
    'SCORE_MAP': {
        simdata.NON_SEVERE_PRIORITY: gtconfig.nonsevere_fix_weight,
        simdata.NORMAL_PRIORITY: 10 * 2,
        simdata.SEVERE_PRIORITY: gtconfig.severe_fix_weight
    },

    # Throtling configuration parameters.
    'THROTTLING_ENABLED': False,
    'INFLATION_FACTOR': None,

    # Empirical Strategies parameters.
    'EMPIRICAL_STRATEGIES': False,
    'N_CLUSTERS': 5,

    'HEURISTIC_STRATEGIES': True,

    # Gatekeeper configuration.
    # 'GATEKEEPER_CONFIG': {'review_time': 8,  # False to disable the Gatekeeper on the simulation.
    #                       'capacity': 1},
    'GATEKEEPER_CONFIG': None,
    'SUCCESS_RATE': 1.0,
    # Team Configuration
    'PLAYER_CRITERIA': '3RD_PARTY_CORRECTIONS',
    'NUMBER_OF_TEAMS': 2,
    'TWINS_REDUCTION': True,
    'AGGREGATE_AGENT_TEAM': -1,
    'ENABLE_RECYCLING': True,  # Remembers previous simulation execution. Currently working for symmetric with twins.
    'SYMMETRIC': True,  # If all the players have the same strategic vision, i.e  there are no advantages per player.
    'ALL_EQUILIBRIA': True  # Instructs gambit to find all equilibria. Only supported for 2 player games.
}

logger = gtconfig.get_logger("exp_equilibrium_results", "exp_equilibrium_results.txt", level=logging.INFO)


def select_reporters_for_simulation(reporter_configuration, game_configuration):
    """
    The production of these reporters will be considered for simulation extraction parameters
    :param reporter_configuration: All valid reporters.
    :return: A filtered list of reporters.
    """

    selection_criteria = game_configuration['PLAYER_CRITERIA']

    if selection_criteria == '3RD_PARTY_CORRECTIONS':
        reporters_with_corrections = [config for config in reporter_configuration if
                                      config['with_modified_priority'] > 0]

        corrections_size = len(reporters_with_corrections)
        logger.info("PLAYER SELECTION CRITERIA: 3RD_PARTY_CORRECTIONS Original Reporters: " + str(len(
            reporter_configuration)) + " Reporters with corrected priorities: " + str(corrections_size))

        if corrections_size == 0:
            logger.info("In the current dataset there is NO THIRD PARTY CORRECTIONS." +
                        " Using the unfiltered reporters for simulation")

            return reporter_configuration

    elif selection_criteria == 'TOP_FROM_TEAMS':
        logger.info("PLAYER SELECTION CRITERIA: Top " + str(
            game_configuration["NUMBER_OF_TEAMS"]) + " most productive testers.")
        sorted_by_productivity = sorted(reporter_configuration, key=lambda reporter: reporter['reports'], reverse=True)
        return sorted_by_productivity[: game_configuration["NUMBER_OF_TEAMS"]]

    return reporters_with_corrections


def get_heuristic_strategies():
    """
    Returns a set of strategies not related with how users are behaving in data.
    :return:
    """

    honest_strategy = simutils.EmpiricalInflationStrategy(strategy_config=simmodel.HONEST_CONFIG)
    simple_inflate_strategy = simutils.EmpiricalInflationStrategy(strategy_config=simmodel.SIMPLE_INFLATE_CONFIG)

    return [honest_strategy,
            simple_inflate_strategy]


def assign_empirical_strategy(reporter_configuration, correction_dataframe, strategy_catalog):
    """
    Adds a strategy instance per reporter, on the reporter_configuration catalog.

    :param reporter_configuration: Reporter catalog.
    :param correction_dataframe: Contain reporter behaviour information.
    :param strategies_per_team: Strategy catalog.
    :return: None.
    """

    print "Assigning each reporter the corresponding Empirical Strategy ..."

    for reporter in reporter_configuration:
        reporter_name = reporter['name']

        if reporter_name in correction_dataframe.index:
            strategy_index = int(correction_dataframe.loc[reporter_name]['cluster'])
            reporter[simmodel.STRATEGY_KEY] = strategy_catalog[strategy_index]


def get_empirical_strategies(reporter_configuration, n_clusters=3):
    """
    It will group a list of reporters in a predefined number of clusters
    :param n_clusters: Number of clusters, which will determine the number of strategies to extract.
    :param reporter_configuration: List of reporter configuration
    :return: The representative strategy per team.
    """

    print "Gathering strategies from reporter behaviour ..."
    print "Original number of reporters: ", len(reporter_configuration), " grouping in ", n_clusters, " clusters "

    reporters_with_corrections = [config for config in reporter_configuration if
                                  config['with_modified_priority'] > 0]
    print "Reporters with corrections: ", len(reporters_with_corrections)

    correction_dataframe = simutils.get_reporter_behavior_dataframe(reporters_with_corrections)

    kmeans = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)

    kmeans.fit(correction_dataframe)

    predicted_clusters = kmeans.predict(correction_dataframe)
    cluster_column = 'cluster'
    correction_dataframe[cluster_column] = predicted_clusters

    centroids = kmeans.cluster_centers_
    print "Clustering centroids ..."

    strategies_per_team = []
    for index, centroid in enumerate(centroids):
        nonsevere_inflation_index = 0
        severe_deflation_index = 1

        inflation_as_string = "{0:.0f}%".format(centroid[nonsevere_inflation_index] * 100)
        deflation_as_string = "{0:.0f}%".format(centroid[severe_deflation_index] * 100)

        print  " ", simutils.REPORTER_COLUMNS[nonsevere_inflation_index], ": ", inflation_as_string, \
            " ", simutils.REPORTER_COLUMNS[severe_deflation_index], ": ", deflation_as_string

        strategies_per_team.append(
            {'name': 'EMPIRICAL' + str(index) + "_INF" + inflation_as_string + "DEF" + deflation_as_string,
             simmodel.NON_SEVERE_INFLATED_COLUMN: centroid[nonsevere_inflation_index],
             simmodel.SEVERE_DEFLATED_COLUMN: centroid[severe_deflation_index]
             })

    print "Cluster distribution: \n", correction_dataframe[cluster_column].value_counts()
    return strategies_per_team, correction_dataframe


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
            strategy_name = strategy.name

            strategy_map['name'] += strategy_name + "_"
            strategy_map['map'][index] = strategy

        strategy_map['name'] = strategy_map['name'][:-1]
        strategy_maps.append(strategy_map)

    return strategy_maps


def start_payoff_calculation(enhanced_dataframe, project_keys, game_configuration, priority_queue=False,
                             dev_team_factor=1.0):
    """
    Given a strategy profile list, calculates payoffs per player thorugh simulation.
    :param enhanced_dataframe: Report data to gather simulation input.
    :param project_keys: Projects to be considered.
    :return: Payoffs per player per profile.
    """

    input_params = prepare_simulation_inputs(enhanced_dataframe, project_keys, game_configuration, priority_queue)

    return run_simulation(strategy_maps=input_params.strategy_maps, strategies_catalog=input_params.strategies_catalog,
                          player_configuration=input_params.player_configuration,
                          dev_team_size=input_params.dev_team_size,
                          resolution_time_gen=input_params.resolution_time_gen,
                          ignored_gen=input_params.ignored_gen, reporter_gen=input_params.reporter_gen,
                          target_fixes=input_params.target_fixes, batch_size_gen=input_params.batch_size_gen,
                          interarrival_time_gen=input_params.interarrival_time_gen,
                          priority_generator=input_params.priority_generator,
                          teams=input_params.teams,
                          game_configuration=game_configuration,
                          priority_queue=priority_queue,
                          dev_team_factor=dev_team_factor)


def prepare_simulation_inputs(enhanced_dataframe, all_project_keys, game_configuration, priority_queue=False):
    """
    Based on the provided dataframe, this functions produces the simulation inputs.

    :param enhanced_dataframe: Dataframe with bug reports.
    :param project_keys: Selected projects.
    :param game_configuration: Game configuration.
    :return: Simulation inputs
    """

    project_keys = all_project_keys
    logger.info("Project catalog: " + str(project_keys))
    total_projects = len(project_keys)

    if game_configuration["PROJECT_FILTER"] is not None and len(game_configuration["PROJECT_FILTER"]) >= 1:
        project_keys = game_configuration["PROJECT_FILTER"]

    logger.info(
        "Original projects " + str(total_projects) + " Project Filter: " + str(game_configuration["PROJECT_FILTER"]) + \
        " Projects remaining after reduction: " + str(len(project_keys)))

    valid_reports = simdriver.get_valid_reports(project_keys=project_keys, enhanced_dataframe=enhanced_dataframe,
                                                exclude_self_fix=gtconfig.exclude_self_fix)
    valid_reporters, _ = simdriver.get_reporter_configuration(valid_reports)
    logger.info("Reporters after drive-in tester removal ..." + str(len(valid_reporters)))

    strategies_catalog = []

    empirical_strategies = None
    reporter_behaviour = None
    if game_configuration["EMPIRICAL_STRATEGIES"]:
        logger.info("Empirical Strategy extraction over " + str(len(all_project_keys)) + " project datasets ...")
        all_reports = simdriver.get_valid_reports(project_keys=all_project_keys,
                                                  enhanced_dataframe=enhanced_dataframe,
                                                  exclude_self_fix=gtconfig.exclude_self_fix)
        all_reporters, _ = simdriver.get_reporter_configuration(all_reports)

        logger.info("Generating elbow-method plot...")
        simutils.elbow_method_for_reporters(all_reporters, file_prefix="_".join(all_project_keys))

        strategy_params, reporter_behaviour = get_empirical_strategies(all_reporters,
                                                                       n_clusters=game_configuration["N_CLUSTERS"])
        empirical_strategies = [simutils.EmpiricalInflationStrategy(strategy_config=strategy_config) for strategy_config
                                in
                                strategy_params]

        strategies_catalog.extend(empirical_strategies)

    if game_configuration["HEURISTIC_STRATEGIES"]:
        strategies_catalog.extend(get_heuristic_strategies())

    # This are the reporters whose reported bugs will be used to configure the simulation.
    reporter_configuration = select_reporters_for_simulation(valid_reporters, game_configuration)

    # This is the configuration of the actual game players.
    player_configuration = reporter_configuration

    logger.info("Reporters selected for playing the game " + str(len(player_configuration)))

    if game_configuration["EMPIRICAL_STRATEGIES"]:
        assign_empirical_strategy(player_configuration, reporter_behaviour, empirical_strategies)

    teams = game_configuration["NUMBER_OF_TEAMS"]
    strategy_maps = get_strategy_map(strategies_catalog, teams)

    engaged_testers = [reporter_config['name'] for reporter_config in reporter_configuration]
    valid_reports = simdata.filter_by_reporter(valid_reports, engaged_testers)
    logger.info("Issues in training after reporter filtering: " + str(len(valid_reports.index)))

    logger.info(str(len(valid_reports.index)) + " reports where considered for simulation. Those reports where made by"
                + str(len(engaged_testers)) + " reporters.")

    logger.info("Starting simulation for project " + str(project_keys))

    # When NOT dealing with a priority queue, the ignore behaviour is disabled
    disable_ignore = not priority_queue

    resolution_time_gen, ignored_gen, priority_generator = simdriver.get_simulation_input(training_issues=valid_reports,
                                                                                          disable_ignore=disable_ignore)
    dev_team_size, issues_resolved, resolved_in_period, dev_team_bandwith = simdriver.get_dev_team_production(
        valid_reports)

    logger.info("Obtaining priority change distribution ...")
    review_time_gen = simdriver.get_priority_change_gen(training_issues=valid_reports)

    target_fixes = issues_resolved
    bug_reporters = valid_reports['Reported By']
    test_team_size = bug_reporters.nunique()

    reporter_gen, batch_size_gen, interarrival_time_gen = simdriver.get_report_stream_params(valid_reports,
                                                                                             player_configuration,
                                                                                             symmetric=
                                                                                             game_configuration[
                                                                                                 'SYMMETRIC'])

    catcher_generator = None
    if game_configuration['SUCCESS_RATE'] is not None:
        success_rate = game_configuration['SUCCESS_RATE']
        logger.info("The inflation detection succes rate is: " + str(success_rate))
        catcher_generator = simutils.DiscreteEmpiricalDistribution(name="InflationCatcher",
                                                                   values=[True, False],
                                                                   probabilities=[success_rate, (1 - success_rate)])

    logger.info("Project " + str(project_keys) + " Test Period: " + "ALL" + " Reporters: " + str(
        test_team_size) + " Developers:" + str(dev_team_size) + \
                " Resolved in Period: " + str(issues_resolved) + " Dev Team Bandwith: " + str(dev_team_bandwith))

    input_params = recordtype('SimulationParams',
                              ['strategy_maps', 'strategies_catalog',
                               'player_configuration', 'dev_team_size',
                               'resolution_time_gen', 'teams', 'ignored_gen',
                               'reporter_gen', 'target_fixes', 'batch_size_gen',
                               'interarrival_time_gen', 'priority_generator', 'review_time_gen',
                               'catcher_generator', 'dev_time_budget'])

    return input_params(strategy_maps, strategies_catalog, player_configuration, dev_team_size,
                        resolution_time_gen, teams, ignored_gen, reporter_gen, target_fixes, batch_size_gen,
                        interarrival_time_gen, priority_generator, review_time_gen, catcher_generator,
                        dev_team_bandwith)


def assign_teams(player_configuration):
    """
    Regular team assignment without aggregation.
    :param player_configuration: List of player configurations.
    :return: None.
    """
    for team, player in enumerate(player_configuration):
        player['team'] = team


def configure_strategies_per_team(player_configuration, strategy_map):
    """
    Assigns the strategies corresponding to teams according to an specific strategy profile.
    :return: Player index whose payoff value is of interest.
    """

    for config in player_configuration:
        config[simmodel.STRATEGY_KEY] = strategy_map[config['team']]


def get_simulation_results(file_prefix, strategy_map, player_configuration, game_configuration, simfunction, simparams,
                           simulation_history):
    """
    For a given strategy profile, it returns the results of its simulation execution.
    :return: 
    """
    configure_strategies_per_team(player_configuration, strategy_map)

    simulation_output = simfunction(
        team_capacity=simparams['dev_team_size'],
        ignored_gen=simparams['ignored_gen'],
        reporter_gen=simparams['reporter_gen'],
        target_fixes=simparams['target_fixes'],
        batch_size_gen=simparams['batch_size_gen'],
        interarrival_time_gen=simparams['interarrival_time_gen'],
        priority_generator=simparams['priority_generator'],
        reporters_config=player_configuration,
        resolution_time_gen=simparams['resolution_time_gen'],
        max_time=simparams['simulation_time'],
        catcher_generator=simparams['catcher_generator'],
        max_iterations=game_configuration["REPLICATIONS_PER_PROFILE"],
        inflation_factor=game_configuration["INFLATION_FACTOR"],
        quota_system=game_configuration["THROTTLING_ENABLED"],
        gatekeeper_config=game_configuration["GATEKEEPER_CONFIG"])

    simulation_result = simcruncher.consolidate_payoff_results("ALL", player_configuration,
                                                               simulation_output,
                                                               game_configuration["SCORE_MAP"],
                                                               game_configuration["PRIORITY_SCORING"])
    overall_dataframe = pd.DataFrame(simulation_result)
    simulation_history.append(overall_dataframe)

    file_name = "csv/all_teams_" + file_prefix + '_simulation_results.csv'
    overall_dataframe.to_csv(file_name, index=False)
    logger.info("The simulation results for the strategy profile were stored at " + file)

    return overall_dataframe


def run_simulation(strategy_maps, strategies_catalog, player_configuration, dev_team_size, resolution_time_gen, teams,
                   game_configuration, ignored_gen=None, reporter_gen=None, target_fixes=None, batch_size_gen=None,
                   interarrival_time_gen=None, catcher_generator=None, priority_generator=None,
                   simfunction=simutils.launch_simulation_parallel, priority_queue=False, dev_team_factor=1.0):
    """

    :param strategy_maps: Strategy profiles of the game.
    :param strategies_catalog: List of strategies available for players.
    :param player_configuration: List of reporters with parameters.
    :param dev_team_size: Number of developers available for bug fixing.
    :param bugs_by_priority: Bugs to find in the system, according to their priority.
    :param resolution_time_gen: Generators for resolution time. Per priority.
    :param dev_team_bandwith: Number of dev time hours for bug fixing.
    :param teams: Number of teams available.
    :param game_configuration: Game configuration parameters.
    :return: List of equilibrium profiles.
    """

    simulation_time = sys.maxint

    profile_payoffs = []
    logger.info("Simulating " + str(len(strategy_maps)) + " strategy profiles...")

    simulation_history = []

    if not game_configuration['TWINS_REDUCTION'] and game_configuration["NUMBER_OF_TEAMS"] == len(player_configuration):
        logger.info("PLAYER AGGREGATION: Agents are not agregated. No player reduction is applied.")
        assign_teams(player_configuration)

    game_desc = get_game_description(game_configuration, priority_queue=priority_queue, dev_team_factor=dev_team_factor)

    for index, map_info in enumerate(strategy_maps):
        logger.info("Current scenario: " + game_desc + ". Simulating profile " + str((index + 1)) + " of " + str(
            len(strategy_maps)))

        file_prefix, strategy_map = map_info['name'], map_info['map']

        file_prefix = game_desc + file_prefix

        overall_dataframes = []

        team_capacity = int(dev_team_size * dev_team_factor)

        logger.info("Team capacity: " + str(team_capacity) + ". After applying the factor of " + str(
            dev_team_factor) + " to a team of " + str(dev_team_size))

        simulation_config = simutils.SimulationConfig(team_capacity=team_capacity,
                                                      ignored_gen=ignored_gen,
                                                      reporter_gen=reporter_gen,
                                                      target_fixes=target_fixes,
                                                      batch_size_gen=batch_size_gen,
                                                      interarrival_time_gen=interarrival_time_gen,
                                                      priority_generator=priority_generator,
                                                      reporters_config=player_configuration,
                                                      resolution_time_gen=resolution_time_gen,
                                                      max_time=simulation_time,
                                                      catcher_generator=catcher_generator,
                                                      priority_queue=priority_queue,
                                                      inflation_factor=game_configuration["INFLATION_FACTOR"],
                                                      quota_system=game_configuration["THROTTLING_ENABLED"],
                                                      gatekeeper_config=game_configuration["GATEKEEPER_CONFIG"])

        if not gtconfig.parallel:
            logger.info("PARALLEL EXECUTION: Has been disabled.")
            simfunction = simutils.launch_simulation

        if game_configuration['TWINS_REDUCTION']:
            overall_dataframes += simtwins.get_simulation_results(file_prefix, strategy_map, player_configuration,
                                                                  game_configuration, simfunction,
                                                                  simulation_config, simulation_history)
        else:
            overall_dataframes.append(get_simulation_results(file_prefix, strategy_map, player_configuration,
                                                             game_configuration, simfunction,
                                                             simulation_config, simulation_history))

        payoffs = simcruncher.get_team_metrics(str(index) + "-" + file_prefix, "ALL", teams, overall_dataframes,
                                               game_configuration["NUMBER_OF_TEAMS"])
        profile_payoffs.append((file_prefix, payoffs))

    logger.info("Generating Gambit NFG file ...")
    gambit_file = gtutils.get_strategic_game_format(game_desc, player_configuration, strategies_catalog,
                                                    profile_payoffs, teams)
    logger.info("NFG File created at " + gambit_file)

    print "Executing Gambit for equilibrium calculation..."
    equilibrium_list = gtutils.calculate_equilibrium(strategies_catalog=strategies_catalog, gambit_file=gambit_file,
                                                     all_equilibria=game_configuration['ALL_EQUILIBRIA'])

    logger.info("Equilibria found: " + str(len(equilibrium_list)) + str(equilibrium_list))
    return equilibrium_list


def get_game_description(game_configuration, priority_queue=False, dev_team_factor=1.0):
    """
    Returns a descriptive name for the game in place
    :param game_configuration: Game parameters
    :return: A name
    """
    game_desc = "AS-IS" if not game_configuration["THROTTLING_ENABLED"] else "THROTTLING"
    game_desc = "GATEKEEPER" if game_configuration["GATEKEEPER_CONFIG"] else game_desc

    if game_configuration["THROTTLING_ENABLED"]:
        game_desc += "_INF" + str(game_configuration['INFLATION_FACTOR'] * 100)

    if game_configuration["GATEKEEPER_CONFIG"] and game_configuration["SUCCESS_RATE"]:
        game_desc += "_RATE" + str(game_configuration['SUCCESS_RATE'] * 100)

    project_prefix = ""
    if game_configuration["PROJECT_FILTER"] is not None and len(game_configuration["PROJECT_FILTER"]) > 0:
        project_prefix = "_".join(game_configuration["PROJECT_FILTER"])

    final_description = project_prefix + "_" + game_desc + "_PRIQUEUE_" + str(priority_queue) + "_TEAMFACTOR_" + str(
        dev_team_factor)
    return final_description


def main():
    logger.info("Loading information from " + str(simdata.ALL_ISSUES_CSV))
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    logger.info("Adding calculated fields...")
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    all_valid_projects = simdriver.get_valid_projects(enhanced_dataframe=enhanced_dataframe,
                                                      exclude_self_fix=gtconfig.exclude_self_fix)

    per_project = False
    consolidated = True

    simulation_configuration = dict(DEFAULT_CONFIGURATION)
    simulation_configuration['REPLICATIONS_PER_PROFILE'] = gtconfig.replications_per_profile
    simulation_configuration['EMPIRICAL_STRATEGIES'] = gtconfig.use_empirical_strategies
    simulation_configuration['N_CLUSTERS'] = 5

    valid_projects = all_valid_projects

    for priority_queue in gtconfig.priority_queues:
        for dev_team_factor in gtconfig.dev_team_factors:

            logger.info("GAME CONFIGURATION: Priority Queue " + str(priority_queue) + " Dev Team Factor: " + str(
                dev_team_factor))

            equilibrium_catalog = []
            if per_project:
                for project in valid_projects:
                    logger.info("Calculating equilibria for project " + str(project))

                    configuration = dict(simulation_configuration)
                    configuration['PROJECT_FILTER'] = [project]
                    equilibrium_list = start_payoff_calculation(enhanced_dataframe, all_valid_projects,
                                                                configuration, priority_queue=priority_queue,
                                                                dev_team_factor=dev_team_factor)

                    equilibrium_catalog += [gtutils.get_equilibrium_as_dict(identifier=project, profile=profile) for
                                            profile in
                                            equilibrium_list]

            if consolidated:
                configuration = dict(simulation_configuration)
                configuration['PROJECT_FILTER'] = None
                equilibrium_list = start_payoff_calculation(enhanced_dataframe, all_valid_projects, configuration,
                                                            priority_queue=priority_queue,
                                                            dev_team_factor=dev_team_factor)
                equilibrium_catalog += [gtutils.get_equilibrium_as_dict(identifier="CONSOLIDATED", profile=profile) for
                                        profile
                                        in
                                        equilibrium_list]

            prefix = ""
            if consolidated:
                prefix += "ALL_"
            if per_project:
                prefix += "PROJECTS_"

            prefix += "_PRIQUEUE_" + str(priority_queue) + "_DEVFACTOR_" + str(dev_team_factor)
            results_dataframe = pd.DataFrame(equilibrium_catalog)
            file_name = "csv/" + prefix + "vanilla_equilibrium_results.csv"
            results_dataframe.to_csv(file_name)
            logger.info("Consolidated equilibrium results written to " + str(file_name))


if __name__ == "__main__":

    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
