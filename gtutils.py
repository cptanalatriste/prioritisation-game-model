"""
This modules provides functions for equilibrium calculation through the Gambit Software.
"""

import subprocess
from string import Template
from collections import defaultdict
import fractions
import gtconfig

GAMBIT_FOLDER = gtconfig.gambit_folder

ENUMERATE_EQUILIBRIA_SOLVER = gtconfig.enumerate_equilibria_solver  # It only works with two players.
NO_BANNER_OPTION = "-q"

QUANTAL_RESPONSE_SOLVER = gtconfig.quantal_response_solver
ONLY_NASH_OPTION = "-e"


def get_strategic_game_format(game_desc, reporter_configuration, strategies_catalog, profile_payoffs, players):
    """
    Generates the content of a Gambit NFG file.
    :return: Name of the generated file.
    """

    template = 'NFG 1 R "$num_reporters reporters - $num_strategies strategies - $game_desc" { $player_catalog } \n\n ' \
               '{ $actions_per_player \n}\n""\n\n' \
               '{\n$payoff_per_profile\n}\n$profile_ordering'

    nfg_template = Template(template)
    teams = set(['"Team_' + str(team_number) + '"' for team_number in range(players)])
    actions = " ".join(['"' + strategy.name + '"' for strategy in strategies_catalog])
    action_list = ["{ " + actions + " }" for _ in teams]

    profile_lines = []
    profile_ordering = []

    for index, profile_info in enumerate(profile_payoffs):
        profile_name, payoffs = profile_info
        payoff_line = '{ "' + profile_name + '" ' + ",".join(payoffs) + " }"

        profile_lines.append(payoff_line)
        profile_ordering.append(str(index + 1))

    num_reporters = len(reporter_configuration)
    num_strategies = len(strategies_catalog)
    player_catalog = " ".join(teams)
    actions_per_player = "\n".join(action_list)
    payoff_per_profile = "\n".join(profile_lines)
    profile_ordering = " ".join(profile_ordering)

    file_content = nfg_template.substitute({'num_reporters': num_reporters,
                                            'num_strategies': num_strategies,
                                            'game_desc': game_desc,
                                            'player_catalog': player_catalog,
                                            'actions_per_player': actions_per_player,
                                            'payoff_per_profile': payoff_per_profile,
                                            'profile_ordering': profile_ordering})

    file_name = "nfg/" + str(num_reporters) + "_players_" + str(
        num_strategies) + "_strategies_" + game_desc + "_game.nfg"
    with open(file_name, "w") as gambit_file:
        gambit_file.write(file_content)

    return file_name


def calculate_equilibrium(strategies_catalog, gambit_file, all_equilibria=True, debug=False):
    """
    Executes Gambit for equilibrium calculation.
    :param strategies_catalog: Catalog of available strategies.
    :param gambit_file:
    :return: List of equilibrium profiles.
    """

    if all_equilibria:
        process = GAMBIT_FOLDER + ENUMERATE_EQUILIBRIA_SOLVER
        command_line = [process, NO_BANNER_OPTION, gambit_file]
    else:
        print "NOTE: Only one equilibrium will be found."
        process = GAMBIT_FOLDER + QUANTAL_RESPONSE_SOLVER
        command_line = [process, ONLY_NASH_OPTION, gambit_file]

    print "Calculating equilibrium for: ", gambit_file, " using ", process

    solver_process = subprocess.Popen(command_line, stdout=subprocess.PIPE)

    nash_equilibrium_strings = []
    while True:
        line = solver_process.stdout.readline()
        if line != '':
            nash_equilibrium_strings.append(line)
        else:
            break

    if debug:
        print "len(nash_equilibrium_strings): ", len(nash_equilibrium_strings)

    start_index = 3

    equilibrium_list = []
    for index, nash_equilibrium in enumerate(nash_equilibrium_strings):
        if debug:
            print "nash_equilibrium: ", nash_equilibrium

        print "Equilibrium ", str(index + 1), " of ", len(nash_equilibrium_strings)

        nash_equilibrium = nash_equilibrium.strip()
        nash_equilibrium = nash_equilibrium[start_index:].split(",")

        team_index = 0
        strategy_index = 0

        equilibrium_profile = defaultdict(defaultdict)
        for probability in nash_equilibrium:

            strategy_name = strategies_catalog[strategy_index].name
            print "Team ", team_index, "-> Strategy: ", strategy_name, " \t\tProbability", probability

            equilibrium_profile[team_index][strategy_name] = probability

            if strategy_index < len(strategies_catalog) - 1:
                strategy_index += 1
            else:
                team_index += 1
                strategy_index = 0

        equilibrium_list.append(equilibrium_profile)

        if is_symmetric_equilibrium(equilibrium_profile):
            print "This is a SYMMETRIC EQUILIBRIUM PROFILE!!"

    return equilibrium_list


def get_equilibrium_as_dict(identifier, profile):
    """
    Returns a map representation of the equilibrium, suitable to be stored in a dataframe.
    :param profile: Equilibrium as dict
    :return: An improved dict
    """

    profile_representation = {}
    for team, strategy_set in profile.iteritems():

        for strategy_name, probability in strategy_set.iteritems():
            profile_representation["TEAM_" + str(team) + "_" + strategy_name] = str(
                round(float(fractions.Fraction(probability)), 2))

    profile_representation["SYMMETRIC"] = is_symmetric_equilibrium(profile)
    profile_representation["IDENTIFIER"] = identifier
    return profile_representation


def is_symmetric_equilibrium(profile):
    """
    Returns true if the equilibrium profile represents a symmetric equilibrium.
    :param profile: Map representing the profile.
    :return:
    """

    strategy_names = set()

    for team, strategy in profile.iteritems():
        for strategy_name, probability in strategy.iteritems():
            strategy_names.add(strategy_name)

    is_symmetric = True

    for strategy_name in strategy_names:
        current_probability = None

        for team, strategy in profile.iteritems():
            if current_probability is None:
                current_probability = strategy[strategy_name]
            elif strategy[strategy_name] != current_probability:
                return False

    return is_symmetric
