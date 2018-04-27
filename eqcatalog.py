"""
This module contains the equilibrium list used for performance comparison.
"""
import penaltyexp
import simmodel
import pandas as pd

# Empirical strategy catalog
import simutils

EMPIRICAL_HONEST = {"name": "empirical_honest",
                    "as_string": "EMPIRICAL0_INF5%DEF1%",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.05,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.01}

PERSISTENT_DEFLATOR = {"name": "persistent_deflator",
                       "as_string": "EMPIRICAL1_INF8%DEF100%",
                       simmodel.NON_SEVERE_INFLATED_COLUMN: 0.08,
                       simmodel.SEVERE_DEFLATED_COLUMN: 1.00}

REGULAR_DEFLATOR = {"name": "regular_deflator",
                    "as_string": "EMPIRICAL2_INF4%DEF58%",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.04,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.58}

EMPIRICAL_INFLATOR = {"name": "empirical_inflator",
                      "as_string": "EMPIRICAL3_INF19%DEF2%",
                      simmodel.NON_SEVERE_INFLATED_COLUMN: 0.19,
                      simmodel.SEVERE_DEFLATED_COLUMN: 0.02}

OCCASSIONAL_DEFLATOR = {"name": "occasional_deflator",
                        "as_string": "EMPIRICAL4_INF6%DEF26%",
                        simmodel.NON_SEVERE_INFLATED_COLUMN: 0.06,
                        simmodel.SEVERE_DEFLATED_COLUMN: 0.26}


def get_profiles_from_file(filename, scenario_desc, input_params):
    """
    Extracts a list of profile entries from a file.
    :param filename:
    :return:
    """

    print "Extracting equilibria from: " + filename

    profiles_dataframe = pd.read_csv(filename)
    symmetric_profile_df = profiles_dataframe[profiles_dataframe["SYMMETRIC"] == True]

    result = []
    for index, profile_row in symmetric_profile_df.iterrows():
        strategy_catalog = get_heuristic_strategy_catalog()

        strategy_names = ['TEAM_0_' + strategy['as_string'] for strategy in strategy_catalog]
        probabilities = [profile_row.get(strategy_name) for strategy_name in strategy_names]

        delta = abs(sum(probabilities) - 1.0)
        if delta > simutils.EPSILON:
            raise Exception("The probabilities should sum 1. Probabilities: " + str(
                probabilities) + ". Delta: " + str(delta))

        result.append({'name': scenario_desc + "_TSNE" + str(index),
                       'strategy_configs': strategy_catalog,
                       'probabilities': probabilities})

    return [
        generate_single_strategy_profile(input_params.player_configuration, profile_config) for profile_config in
        result]


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

    if priority_queue and dev_team_factor == 0.5:
        filename_inf001 = "INF1.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf003 = "INF3.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf005 = "INF5.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
    elif priority_queue and dev_team_factor == 1.0:
        filename_inf001 = "INF1.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf003 = "INF3.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf005 = "INF5.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"

    elif not priority_queue and dev_team_factor == 0.5:
        filename_inf001 = "INF1.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf003 = "INF3.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf005 = "INF5.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"

    elif not priority_queue and dev_team_factor == 1.0:
        filename_inf001 = "INF1.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf003 = "INF3.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf005 = "INF5.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"

    equilibrium_profiles_inf001 = get_profiles_from_file("csv/" + filename_inf001, scenario_desc=desc_inf001,
                                                         input_params=input_params)

    desc_inf003 = "THROTTLING_INF003"
    process_configuration_inf003 = dict(process_configuration_inf001)
    process_configuration_inf003["INFLATION_FACTOR"] = 0.03

    equilibrium_profiles_inf003 = get_profiles_from_file("csv/" + filename_inf003, scenario_desc=desc_inf003,
                                                         input_params=input_params)

    desc_inf005 = "THROTTLING_INF005"
    process_configuration_inf005 = dict(process_configuration_inf001)
    process_configuration_inf005["INFLATION_FACTOR"] = 0.05

    equilibrium_profiles_inf005 = get_profiles_from_file("csv/" + filename_inf005, scenario_desc=desc_inf005,
                                                         input_params=input_params)

    return [{"desc": desc_inf001,
             "simulation_configuration": process_configuration_inf001,
             "equilibrium_profiles": equilibrium_profiles_inf001},
            {"desc": desc_inf003,
             "simulation_configuration": process_configuration_inf003,
             "equilibrium_profiles": equilibrium_profiles_inf003},
            {"desc": desc_inf005,
             "simulation_configuration": process_configuration_inf005,
             "equilibrium_profiles": equilibrium_profiles_inf005}]


def get_unsupervised_prioritization_equilibria(simulation_configuration, input_params, priority_queue=False,
                                               dev_team_factor=1.0):
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

    if priority_queue and dev_team_factor == 0.5:
        filename = "ALL__PRIQUEUE_True_DEVFACTOR_0.5vanilla_equilibrium_results.csv"
    elif priority_queue and dev_team_factor == 1.0:
        filename = "ALL__PRIQUEUE_True_DEVFACTOR_1.0vanilla_equilibrium_results.csv"
    elif not priority_queue and dev_team_factor == 0.5:
        filename = "ALL__PRIQUEUE_False_DEVFACTOR_0.5vanilla_equilibrium_results.csv"
    elif not priority_queue and dev_team_factor == 1.0:
        filename = "ALL__PRIQUEUE_False_DEVFACTOR_1.0vanilla_equilibrium_results.csv"

    equilibrium_profiles = get_profiles_from_file("csv/" + filename, scenario_desc=desc,
                                                  input_params=input_params)

    return [{"desc": desc,
             "simulation_configuration": process_configuration,
             "equilibrium_profiles": equilibrium_profiles}]


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

    if priority_queue and dev_team_factor == 0.5:
        filename_succ090 = "GATEKEEPER_SUCCESS0.9_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_succ100 = "GATEKEEPER_SUCCESS1.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_succ050 = "GATEKEEPER_SUCCESS0.5_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"

    elif priority_queue and dev_team_factor == 1.0:
        filename_succ090 = "GATEKEEPER_SUCCESS0.9_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_succ100 = "GATEKEEPER_SUCCESS1.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_succ050 = "GATEKEEPER_SUCCESS0.5_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"

    elif not priority_queue and dev_team_factor == 0.5:
        filename_succ090 = "GATEKEEPER_SUCCESS0.9_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_succ100 = "GATEKEEPER_SUCCESS1.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_succ050 = "GATEKEEPER_SUCCESS0.5_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"

    elif not priority_queue and dev_team_factor == 1.0:
        filename_succ090 = "GATEKEEPER_SUCCESS0.9_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_succ100 = "GATEKEEPER_SUCCESS1.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_succ050 = "GATEKEEPER_SUCCESS0.5_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"

    equilibrium_profiles_succ090 = get_profiles_from_file("csv/" + filename_succ090, scenario_desc=desc_succ090,
                                                          input_params=input_params)

    desc_succ100 = "GATEKEEPER_SUCC100"
    process_configuration_succ100 = dict(process_configuration_succ090)
    process_configuration_succ100["SUCCESS_RATE"] = 1.0

    equilibrium_profiles_succ100 = get_profiles_from_file("csv/" + filename_succ100, scenario_desc=desc_succ100,
                                                          input_params=input_params)

    desc_succ50 = "GATEKEEPER_SUCC50"
    process_configuration_succ050 = dict(process_configuration_succ090)
    process_configuration_succ050["SUCCESS_RATE"] = 0.5

    equilibrium_profiles_succ050 = get_profiles_from_file("csv/" + filename_succ050, scenario_desc=desc_succ50,
                                                          input_params=input_params)
    return [{"desc": desc_succ50,
             "simulation_configuration": process_configuration_succ050,
             "equilibrium_profiles": equilibrium_profiles_succ050},
            {"desc": desc_succ090,
             "simulation_configuration": process_configuration_succ090,
             "equilibrium_profiles": equilibrium_profiles_succ090},
            {"desc": desc_succ100,
             "simulation_configuration": process_configuration_succ100,
             "equilibrium_profiles": equilibrium_profiles_succ100}]
