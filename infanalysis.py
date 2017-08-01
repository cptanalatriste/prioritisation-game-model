"""
This module contains the code for regression analysis for the bug reporting
processes considered
"""

import time
import statsmodels.api as sm
import random
import pandas as pd
import matplotlib.pyplot as plt

import gtconfig
import simdata
import simmodel
import syseval

if gtconfig.is_windows:
    import winsound

logger = gtconfig.get_logger("regression_analysis", "regression_analysis.txt")


def generate_inflated_profile(inflation_rate, empirical_profile):
    """
    Selects a subset of players randomly to adopt the Heuristic Inflator strategy.
    :param inflation_rate: Proportion of players to convert.
    :param empirical_profile: Original profile in the dataset.
    :return: A new strategy profile.
    """
    offender_number = int(len(empirical_profile.keys()) * inflation_rate)
    logger.info("Sampling " + str(offender_number) + " reporters for inflation ...")

    inflated_profile = dict(empirical_profile)

    inflators = random.sample(empirical_profile.keys(), offender_number)

    for inflator in inflators:
        inflated_profile[inflator] = simmodel.SIMPLE_INFLATE_CONFIG

    return inflated_profile


def do_unsupervised_prioritization(simulation_configuration, input_params, simfunction, empirical_profile,
                                   replications_per_rate, step):
    simulation_configuration['REPLICATIONS_PER_PROFILE'] = replications_per_rate
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration["GATEKEEPER_CONFIG"] = None

    regression_data = []

    logger.info("Reporters in population: " + str(len(empirical_profile.keys())))
    for inflation_rate in range(0, 110, step):
        logger.info(
            "Simulating UNSUPERVISED PRIORITIZATION with an HEURISTIC INFLATOR probability of " + str(inflation_rate))

        profile_after_inflation = generate_inflated_profile(inflation_rate / 100.0, empirical_profile)
        syseval.apply_strategy_profile(input_params.player_configuration, profile_after_inflation)

        simulation_output = syseval.run_scenario(simfunction, input_params, simulation_configuration)
        regression_data += [{'inflation_rate': inflation_rate,
                             'severe_completed': severe_completed
                             } for severe_completed in
                            simulation_output.get_completed_per_real_priority(simdata.SEVERE_PRIORITY)]

    dataframe = pd.DataFrame(regression_data)

    severe_completed = dataframe['severe_completed']
    inflation_rates = dataframe['inflation_rate']
    rates_with_constant = sm.add_constant(inflation_rates)

    regression_result = sm.OLS(severe_completed, rates_with_constant).fit()
    logger.info("regression_result.summary(): " + str(regression_result.summary()))

    plt.clf()
    axis = dataframe.plot('inflation_rate', 'severe_completed', style='o')
    plt.ylabel('Heuristic Inflation Probability')
    plt.ylabel('Severe Bugs Fixed')
    plt.title('Severe Bugs Fixed by Inflation Rate')
    sm.graphics.abline_plot(model_results=regression_result, ax=axis)
    plt.savefig("img/" + 'severe_fixes_inf_rate.png')


def main():
    replications_per_rate = 20
    step = 20

    logger.info("Experiment configuration: Replications per Inflation Rate " + str(
        replications_per_rate) + " Offset between rates " + str(step))

    simulation_configuration, simfunction, input_params, empirical_profile = syseval.gather_experiment_inputs()

    do_unsupervised_prioritization(simulation_configuration=simulation_configuration, simfunction=simfunction,
                                   input_params=input_params, empirical_profile=empirical_profile,
                                   replications_per_rate=replications_per_rate, step=step)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    logger.info("Execution time in seconds: " + str(time.time() - start_time))
