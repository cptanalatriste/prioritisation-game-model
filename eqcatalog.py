"""
This module contains the equilibrium list used for performance comparison.
"""
import penaltyexp
import simmodel

# Empirical strategy catalog
EMPIRICAL_HONEST = {"name": "empirical_honest",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.05,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.08}

PERSISTENT_DEFLATOR = {"name": "persistent_deflator",
                       simmodel.NON_SEVERE_INFLATED_COLUMN: 0.08,
                       simmodel.SEVERE_DEFLATED_COLUMN: 1.00}

REGULAR_DEFLATOR = {"name": "regular_deflator",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.04,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.58}

EMPIRICAL_INFLATOR = {"name": "empirical_inflator",
                      simmodel.NON_SEVERE_INFLATED_COLUMN: 0.19,
                      simmodel.SEVERE_DEFLATED_COLUMN: 0.02}

OCCASSIONAL_DEFLATOR = {"name": "occasional_deflator",
                        simmodel.NON_SEVERE_INFLATED_COLUMN: 0.06,
                        simmodel.SEVERE_DEFLATED_COLUMN: 0.26}


def generate_single_strategy_profile(player_configuration, strategy_config):
    """
    Returns a strategy profile with a single strategy
    :return: None
    """

    return {reporter['name']: strategy_config for reporter in player_configuration}


def get_heuristic_strategy_catalog():
    """
    The collection of strategies for our game-theoretic model of bug reporting: It includes two heuristic ones and 5
    found in our dataset
    :return: List of strategy configurations.
    """

    return [EMPIRICAL_HONEST, PERSISTENT_DEFLATOR, REGULAR_DEFLATOR, EMPIRICAL_INFLATOR, OCCASSIONAL_DEFLATOR,
            simmodel.HONEST_CONFIG, simmodel.SIMPLE_INFLATE_CONFIG]


def get_unsupervised_prioritization_equilibria(simulation_configuration, input_params):
    """
    Returns the equilibrium list for unsupervised prioritization. Note that this equilibrium doesn't depend on the
    queuing discipline or the dev team factor.

    :param simulation_configuration:
    :param input_params:
    :return:
    """
    desc = "UNSUPERVISED"

    process_configuration = dict(simulation_configuration)
    process_configuration["THROTTLING_ENABLED"] = False
    process_configuration["GATEKEEPER_CONFIG"] = None
    process_configuration["INFLATION_FACTOR"] = None

    equilibrium_profile = generate_single_strategy_profile(input_params.player_configuration,
                                                           simmodel.SIMPLE_INFLATE_CONFIG)

    return [{"desc": desc,
             "simulation_configuration": process_configuration,
             "equilibrium_profiles": [equilibrium_profile]}]


def get_throttling_equilibria(simulation_config, input_params, priority_queue=True, dev_team_factor=1.0):
    """
    Returns the equilibrium profiles for throttling configuration under analysis.
    :param simulation_config:
    :param input_params:
    :return:
    """
    desc_inf001 = "THROTTLING_INF001"

    process_configuration_inf001 = dict(simulation_config)
    process_configuration_inf001["THROTTLING_ENABLED"] = True
    process_configuration_inf001["GATEKEEPER_CONFIG"] = None
    process_configuration_inf001["INFLATION_FACTOR"] = 0.01

    process_configuration_inf001["SUCCESS_RATE"] = 0.95

    if priority_queue:
        equilibrium_profiles_inf001 = [
            generate_single_strategy_profile(input_params.player_configuration, simmodel.SIMPLE_INFLATE_CONFIG)]
    elif not priority_queue and dev_team_factor == 1.0:
        equilibrium_profiles_inf001 = [
            generate_single_strategy_profile(input_params.player_configuration, EMPIRICAL_HONEST)]
    elif not priority_queue and dev_team_factor == 0.5:
        equilibrium_profiles_inf001 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                        {'name': desc_inf001 + "_TSNE1",
                                                                         'strategy_configs': get_heuristic_strategy_catalog(),
                                                                         'probabilities': [0.11, 0.0, 0.0, 0.0, 0.0,
                                                                                           0.89, 0.0]})]

    desc_inf003 = "THROTTLING_INF003"
    process_configuration_inf003 = dict(process_configuration_inf001)
    process_configuration_inf003["INFLATION_FACTOR"] = 0.03

    if priority_queue and dev_team_factor == 0.5:
        equilibrium_profiles_inf003 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                        {'name': desc_inf001 + "_TSNE1",
                                                                         'strategy_configs': get_heuristic_strategy_catalog(),
                                                                         'probabilities': [0.00, 0.0, 0.0, 0.0, 0.11,
                                                                                           0.00, 0.89]}),
                                       generate_single_strategy_profile(input_params.player_configuration,
                                                                        simmodel.HONEST_CONFIG),
                                       generate_single_strategy_profile(input_params.player_configuration,
                                                                        simmodel.SIMPLE_INFLATE_CONFIG)]
    elif priority_queue and dev_team_factor == 1.0:
        equilibrium_profiles_inf003 = [
            generate_single_strategy_profile(input_params.player_configuration, EMPIRICAL_HONEST),
            generate_single_strategy_profile(input_params.player_configuration,
                                             {'name': desc_inf001 + "_TSNE2",
                                              'strategy_configs': get_heuristic_strategy_catalog(),
                                              'probabilities': [0.58, 0.0, 0.0, 0.0, 0.00,
                                                                0.42, 0.00]}),
            generate_single_strategy_profile(input_params.player_configuration,
                                             simmodel.HONEST_CONFIG)]
    elif not priority_queue and dev_team_factor == 0.5:
        equilibrium_profiles_inf003 = [
            generate_single_strategy_profile(input_params.player_configuration, EMPIRICAL_HONEST),
            generate_single_strategy_profile(input_params.player_configuration,
                                             {'name': desc_inf001 + "_TSNE2",
                                              'strategy_configs': get_heuristic_strategy_catalog(),
                                              'probabilities': [0.95, 0.0, 0.0, 0.0, 0.05,
                                                                0.00, 0.00]}),
            generate_single_strategy_profile(input_params.player_configuration,
                                             simmodel.HONEST_CONFIG)]
    elif not priority_queue and dev_team_factor == 1.0:
        equilibrium_profiles_inf003 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                        simmodel.HONEST_CONFIG)]

    # It's the same equilibrium profile in all configurations
    desc_inf005 = "THROTTLING_INF005"
    process_configuration_inf005 = dict(process_configuration_inf001)
    process_configuration_inf005["INFLATION_FACTOR"] = 0.05
    equilibrium_profile_inf005 = generate_single_strategy_profile(input_params.player_configuration,
                                                                  simmodel.HONEST_CONFIG)

    return [{"desc": desc_inf001,
             "simulation_configuration": process_configuration_inf001,
             "equilibrium_profiles": equilibrium_profiles_inf001},
            {"desc": desc_inf003,
             "simulation_configuration": process_configuration_inf003,
             "equilibrium_profiles": equilibrium_profiles_inf003},
            {"desc": desc_inf005,
             "simulation_configuration": process_configuration_inf005,
             "equilibrium_profiles": [equilibrium_profile_inf005]}
            ]


def get_gatekeeper_equilibria(simulation_config, input_params, priority_queue=False, dev_team_factor=1.0):
    """
    Return the equilibrium results for each gatekeeper configuration explored.
    :param simulation_config:
    :param input_params:
    :return:
    """

    desc_succ090 = "GATEKEEPER_SUCC090"
    process_configuration_succ090 = dict(simulation_config)

    process_configuration_succ090["THROTTLING_ENABLED"] = False
    process_configuration_succ090['GATEKEEPER_CONFIG'] = penaltyexp.DEFAULT_GATEKEEPER_CONFIG
    process_configuration_succ090["INFLATION_FACTOR"] = None
    process_configuration_succ090["SUCCESS_RATE"] = 0.9

    equilibrium_profiles_succ090 = None

    if priority_queue:
        equilibrium_profiles_succ090 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         simmodel.SIMPLE_INFLATE_CONFIG)]
    elif not priority_queue and dev_team_factor == 0.5:
        equilibrium_profiles_succ090 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         PERSISTENT_DEFLATOR),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE2",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.83, 0.0, 0.00, 0.0,
                                                                                            0.00, 0.17]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE3",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.14, 0.00, 0.0, 0.00, 0.0,
                                                                                            0.86, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE4",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.09, 0.00, 0.0, 0.00, 0.0,
                                                                                            0.54, 0.37]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE5",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.14, 0.52, 0.0, 0.00, 0.0,
                                                                                            0.34, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE6",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.13, 0.46, 0.0, 0.00, 0.0,
                                                                                            0.33, 0.08]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         simmodel.SIMPLE_INFLATE_CONFIG)]
    elif not priority_queue and dev_team_factor == 1.0:
        equilibrium_profiles_succ090 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE1",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.00, 0.0, 0.14, 0.0,
                                                                                            0.00, 0.86]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ090 + "_TSNE2",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.06, 0.00, 0.0, 0.28, 0.0,
                                                                                            0.00, 0.66]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         simmodel.SIMPLE_INFLATE_CONFIG)
                                        ]

    desc_succ100 = "GATEKEEPER_SUCC100"
    process_configuration_succ100 = dict(process_configuration_succ090)
    process_configuration_succ100["SUCCESS_RATE"] = 1.0

    if priority_queue and dev_team_factor == 0.5:
        equilibrium_profiles_succ100 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE6",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.53, 0.47, 0.00,
                                                                                            0.00, 0.00, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE7",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.31, 0.23, 0.46,
                                                                                            0.00, 0.00, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         EMPIRICAL_INFLATOR)]
    elif priority_queue and dev_team_factor == 1.0:
        equilibrium_profiles_succ100 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE3",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.32, 0.17, 0.00,
                                                                                            0.00, 0.51, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE4",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.40, 0.08, 0.00,
                                                                                            0.00, 0.37, 0.15]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE5",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.00, 0.0, 0.60,
                                                                                            0.40, 0.00, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE7",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.35, 0.0, 0.37,
                                                                                            0.00, 0.28, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE8",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.31, 0.00, 0.35,
                                                                                            0.03, 0.31, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE9",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.25, 0.00, 0.0, 0.42,
                                                                                            0.24, 0.00, 0.09]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE10",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.39, 0.0, 0.30,
                                                                                            0.00, 0.25, 0.06]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         OCCASSIONAL_DEFLATOR),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE15",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.17, 0.08, 0.19,
                                                                                            0.20, 0.36, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         simmodel.SIMPLE_INFLATE_CONFIG)]
    elif not priority_queue and dev_team_factor == 0.5:
        equilibrium_profiles_succ100 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         OCCASSIONAL_DEFLATOR)]
    elif not priority_queue and dev_team_factor == 1.0:
        equilibrium_profiles_succ100 = [generate_single_strategy_profile(input_params.player_configuration,
                                                                         EMPIRICAL_HONEST),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE2",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.41, 0.00, 0.00, 0.00,
                                                                                            0.00, 0.59, 0.00]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE3",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.36, 0.00, 0.00, 0.00,
                                                                                            0.00, 0.41, 0.23]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE4",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.31, 0.00, 0.00, 0.00,
                                                                                            0.09, 0.22, 0.38]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         simmodel.HONEST_CONFIG),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE10",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.00, 0.00, 0.00,
                                                                                            0.00, 0.83, 0.17]}),
                                        generate_single_strategy_profile(input_params.player_configuration,
                                                                         {'name': desc_succ100 + "_TSNE11",
                                                                          'strategy_configs': get_heuristic_strategy_catalog(),
                                                                          'probabilities': [0.00, 0.00, 0.00, 0.00,
                                                                                            0.21, 0.27, 0.52]})]

    # The sub-optimal gatekeeper has the same equilibrium in every scenario
    desc_succ50 = "GATEKEEPER_SUCC50"
    process_configuration_succ050 = dict(process_configuration_succ090)
    process_configuration_succ050["SUCCESS_RATE"] = 0.5

    equilibrium_profile_succ050 = generate_single_strategy_profile(input_params.player_configuration,
                                                                   simmodel.SIMPLE_INFLATE_CONFIG)

    return [{"desc": desc_succ50,
             "simulation_configuration": process_configuration_succ050,
             "equilibrium_profiles": [equilibrium_profile_succ050]},
            {"desc": desc_succ090,
             "simulation_configuration": process_configuration_succ090,
             "equilibrium_profiles": equilibrium_profiles_succ090},
            {"desc": desc_succ100,
             "simulation_configuration": process_configuration_succ100,
             "equilibrium_profiles": equilibrium_profiles_succ100}
            ]
