"""
This modules provides functions for equilibrium calculation through the Gambit Software.
"""

import subprocess
from string import Template

GAMBIT_FOLDER = "C:\Program Files (x86)\Gambit\\"
QUANTAL_RESPONSE_SOLVER = "gambit-logit.exe"
ONLY_NASH_OPTION = "-e"


def get_strategic_game_format(game_desc, reporter_configuration, strategies_catalog, profile_payoffs):
    """
    Generates the content of a Gambit NFG file.
    :return: Name of the generated file.
    """

    template = 'NFG 1 R "$num_reporters reporters - $num_strategies strategies - $game_desc" { $player_catalog } \n\n ' \
               '{ $actions_per_player \n}\n""\n\n' \
               '{\n$payoff_per_profile\n}\n$profile_ordering'

    nfg_template = Template(template)
    teams = set(['"Team_' + config['name'] + '"' for config in reporter_configuration])
    actions = " ".join(['"' + strategy['name'] + '"' for strategy in strategies_catalog])
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

    file_name = str(num_reporters) + "_players_" + str(num_strategies) + "_strategies_" + game_desc + "_game.nfg"
    with open(file_name, "w") as gambit_file:
        gambit_file.write(file_content)

    return file_name


def calculate_equilibrium(reporter_configuration, strategies_catalog, gambit_file):
    """
    Executes Gambit for equilibrium calculation.
    :param strategies_catalog: Catalog of available strategies.
    :param reporter_configuration: List of reporter configurations.
    :param gambit_file:
    :return:
    """

    print "Calculating equilibrium for: ", gambit_file
    process = GAMBIT_FOLDER + QUANTAL_RESPONSE_SOLVER

    solver_process = subprocess.Popen([process, ONLY_NASH_OPTION, gambit_file], stdout=subprocess.PIPE)

    nash_equilibrium = None
    while True:
        line = solver_process.stdout.readline()
        if line != '':
            nash_equilibrium = line
        else:
            break

    start_index = 3
    last_index = -2

    nash_equilibrium = nash_equilibrium[start_index:last_index].split(",")

    player_index = 0
    strategy_index = 0

    for probability in nash_equilibrium:
        print "Team ", reporter_configuration[player_index]['team'], "-> Strategy: ", \
            strategies_catalog[strategy_index]['name'], " \t\tProbability", probability

        if strategy_index < len(strategies_catalog) - 1:
            strategy_index += 1
        else:
            player_index += 1
            strategy_index = 0
