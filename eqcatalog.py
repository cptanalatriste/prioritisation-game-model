"""
This module contains the equilibrium list used for performance comparison.
"""
import gtconfig
import penaltyexp
import simmodel
import pandas as pd

# Empirical strategy catalog
import simutils

EMPIRICAL_HONEST = {"name": "empirical_honest",
                    "as_string": "EMPIRICAL0_INF5%DEF2%",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.05,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.02}

PERSISTENT_DEFLATOR = {"name": "persistent_deflator",
                       "as_string": "EMPIRICAL4_INF10%DEF100%",
                       simmodel.NON_SEVERE_INFLATED_COLUMN: 0.1,
                       simmodel.SEVERE_DEFLATED_COLUMN: 1.00}

REGULAR_DEFLATOR = {"name": "regular_deflator",
                    "as_string": "EMPIRICAL1_INF3%DEF56%",
                    simmodel.NON_SEVERE_INFLATED_COLUMN: 0.03,
                    simmodel.SEVERE_DEFLATED_COLUMN: 0.56}

EMPIRICAL_INFLATOR = {"name": "empirical_inflator",
                      "as_string": "EMPIRICAL3_INF20%DEF2%",
                      simmodel.NON_SEVERE_INFLATED_COLUMN: 0.2,
                      simmodel.SEVERE_DEFLATED_COLUMN: 0.02}

OCCASSIONAL_DEFLATOR = {"name": "occasional_deflator",
                        "as_string": "EMPIRICAL2_INF4%DEF23%",
                        simmodel.NON_SEVERE_INFLATED_COLUMN: 0.04,
                        simmodel.SEVERE_DEFLATED_COLUMN: 0.23}

logger = gtconfig.get_logger("eq_catalog", "eq_catalog.txt")


def get_profiles_from_file(filename, scenario_desc, input_params):
    """
    Extracts a list of profile entries from a file.
    :param filename:
    :return:
    """

    logger.info("Extracting equilibria from: " + filename)

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
    desc_inf003 = "THROTTLING_INF003"

    process_configuration_inf003 = dict(simulation_config)
    process_configuration_inf003["THROTTLING_ENABLED"] = True
    process_configuration_inf003["GATEKEEPER_CONFIG"] = None
    process_configuration_inf003["INFLATION_FACTOR"] = 0.03
    process_configuration_inf003["SUCCESS_RATE"] = 0.95

    if priority_queue and dev_team_factor == 0.5:
        filename_inf003 = "INF3.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf010 = "INF10.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf020 = "INF20.0_PRIQUEUE_True_DEVFACTOR_0.5_equilibrium_results.csv"
    elif priority_queue and dev_team_factor == 1.0:
        filename_inf003 = "INF3.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf010 = "INF10.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf020 = "INF20.0_PRIQUEUE_True_DEVFACTOR_1.0_equilibrium_results.csv"

    elif not priority_queue and dev_team_factor == 0.5:
        filename_inf003 = "INF3.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf010 = "INF10.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"
        filename_inf020 = "INF20.0_PRIQUEUE_False_DEVFACTOR_0.5_equilibrium_results.csv"

    elif not priority_queue and dev_team_factor == 1.0:
        filename_inf003 = "INF3.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf010 = "INF10.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"
        filename_inf020 = "INF20.0_PRIQUEUE_False_DEVFACTOR_1.0_equilibrium_results.csv"

    equilibrium_profiles_inf003 = get_profiles_from_file("csv/" + filename_inf003, scenario_desc=desc_inf003,
                                                         input_params=input_params)

    desc_inf010 = "THROTTLING_INF010"
    process_configuration_inf010 = dict(process_configuration_inf003)
    process_configuration_inf010["INFLATION_FACTOR"] = 0.10
    equilibrium_profiles_inf010 = get_profiles_from_file("csv/" + filename_inf010, scenario_desc=desc_inf010,
                                                         input_params=input_params)

    desc_inf020 = "THROTTLING_INF020"
    process_configuration_inf020 = dict(process_configuration_inf003)
    process_configuration_inf020["INFLATION_FACTOR"] = 0.20
    equilibrium_profiles_inf020 = get_profiles_from_file("csv/" + filename_inf020, scenario_desc=desc_inf020,
                                                         input_params=input_params)

    return [{"desc": desc_inf003,
             "simulation_configuration": process_configuration_inf003,
             "equilibrium_profiles": equilibrium_profiles_inf003},
            {"desc": desc_inf010,
             "simulation_configuration": process_configuration_inf010,
             "equilibrium_profiles": equilibrium_profiles_inf010},
            {"desc": desc_inf020,
             "simulation_configuration": process_configuration_inf020,
             "equilibrium_profiles": equilibrium_profiles_inf020}]


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
