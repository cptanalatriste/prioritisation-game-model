"""
This module contains functions for simulating twin reduction games.
"""

from collections import defaultdict


def aggregate_players(agent_team, reporter_configuration, aggregate_agent_team):
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


def check_simulation_history(overall_dataframes, player_configuration, aggregate_agent_team):
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
                    print "strategy ", strategy, "strategy_info ", strategy_info
                    configured_team = strategy_counters[strategy][team_key].pop()

                    recycled_dataframe.loc[
                        recycled_dataframe[strategy_column] == strategy, 'reporter_team'] = configured_team

            return recycled_dataframe

    return None
