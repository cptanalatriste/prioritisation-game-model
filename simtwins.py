"""
This module contains functions for simulating twin reduction games.
"""

from collections import defaultdict

import pandas as pd

import simcruncher


def aggregate_players(agent_team, reporter_configuration, aggregate_agent_team):
    """
    Assigns a team to each of the players, performing player aggregation.
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
            reporter_configuration[index][team_attribute] = aggregate_agent_team


def get_twins_strategy_map(agent_team, strategy_map, aggregate_agent_team):
    """
    Generates a strategy map for a twins reduction simulation.
    :param agent_team: The team that will be assigned to an individual agent.
    :param strategy_map: Original Strategy Map.
    :return: Strategy map for twins aggregation simulation.
    """

    twins_strategy_map = strategy_map.copy()

    opponent_strategy = None
    for team_in_copy, strategy_in_copy in twins_strategy_map.iteritems():
        if team_in_copy != agent_team:
            opponent_strategy = strategy_in_copy

    twins_strategy_map[aggregate_agent_team] = opponent_strategy
    return twins_strategy_map


def get_simulation_results(file_prefix, strategy_map, player_configuration, game_configuration,
                           simfunction, simparams, simulation_history):
    """
    Given an strategy profile, it returns the results of all the simulation runs, given the "rules" for twins aggregation
    :return: List of dataframes containing simulation execution information.
    """
    overall_dataframes = []
    for team, strategy in strategy_map.iteritems():
        print "Getting payoff for team ", team, " on profile ", file_prefix

        print "PLAYER AGGREGATION: Assigning players to teams according to profile "

        aggregate_players(team, player_configuration, game_configuration["AGGREGATE_AGENT_TEAM"])
        twins_strategy_map = get_twins_strategy_map(team, strategy_map,
                                                    game_configuration["AGGREGATE_AGENT_TEAM"])

        for config in player_configuration:
            config['strategy'] = twins_strategy_map[config['team']]

        aggregate_team = game_configuration["AGGREGATE_AGENT_TEAM"]
        overall_dataframe = check_simulation_history(simulation_history, player_configuration,
                                                     aggregate_team)

        if overall_dataframe is None:
            print "Preparing simulation for getting the payoff for team ", team, " in profile: ", twins_strategy_map

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
                                                                       simulation_output["completed_per_reporter"],
                                                                       simulation_output["bugs_per_reporter"],
                                                                       simulation_output["reports_per_reporter"],
                                                                       simulation_output["resolved_per_reporter"],
                                                                       game_configuration["SCORE_MAP"],
                                                                       game_configuration["PRIORITY_SCORING"])
            overall_dataframe = pd.DataFrame(simulation_result)
            simulation_history.append(overall_dataframe)

        else:
            print "Profile ", twins_strategy_map, " has being already executed. Team ", team, " payoff will be recycled."

        overall_dataframe.to_csv("csv/agent_team_" + str(team) + "_" + file_prefix + '_simulation_results.csv',
                                 index=False)
        overall_dataframes.append(overall_dataframe)

    return overall_dataframes


def check_simulation_history(overall_dataframes, player_configuration, aggregate_agent_team):
    """
    Recycles a previous execution result in case it is consistent with the profile to execute. Specially useful
    while simulating symmetric games.

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
        strategy_name = player['strategy'].name
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
                         (recycled_dataframe['reporter_team'] != aggregate_agent_team)
                configured_team = None
                for team in strategy_counters[strategy_counters.keys()[0]][team_key]:
                    if team != aggregate_agent_team:
                        configured_team = team

                recycled_dataframe.loc[filter, 'reporter_team'] = configured_team
            else:
                for strategy, strategy_info in strategy_counters.iteritems():
                    configured_team = strategy_counters[strategy][team_key].pop()

                    recycled_dataframe.loc[
                        recycled_dataframe[strategy_column] == strategy, 'reporter_team'] = configured_team

            return recycled_dataframe

    return None
