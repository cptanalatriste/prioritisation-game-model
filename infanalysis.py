"""
This module contains the code for regression analysis for the bug reporting
processes considered
"""

import time
import logging
import statsmodels.api as sm
import random
import pandas as pd
import matplotlib.pyplot as plt

import gtconfig
import penaltyexp
import simdata
import simmodel
import syseval

if gtconfig.is_windows:
    import winsound

logger = gtconfig.get_logger("regression_analysis", "regression_analysis.txt", level=logging.INFO)

INDEPENDENT_VARIABLE = 'offender_number'
DEPENDENT_VARIABLES = ['severe_time_ratio', 'severe_completed', 'severe_fixed_ratio', 'severe_fixed_ratio_active']


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
    logger.debug("Inflators: " + str(inflators))

    for inflator in inflators:
        inflated_profile[inflator] = simmodel.SIMPLE_INFLATE_CONFIG

    return inflated_profile, offender_number


def get_performance_dataframe(input_params, simfunction, simulation_configuration, empirical_profile, step, desc):
    """
    Produces a dataframe containing performance measure values per several configurations of inflation probability.
    :param input_params: Simulation inputs.
    :param simfunction: Simulation function.
    :param simulation_configuration: Simulation configuration.
    :param empirical_profile: Empirical strategy profile.
    :param step: Offset between inflation probabilities.
    :return: Dataframe instance.
    """
    regression_data = []

    logger.info("Reporters in population: " + str(len(empirical_profile.keys())))
    inflation_rates = range(0, 110, step)

    for inflation_rate in inflation_rates:
        logger.info(
            "Simulating " + desc + " with an HEURISTIC INFLATOR probability of " + str(inflation_rate))

        normalized_rate = inflation_rate / 100.0
        profile_after_inflation, offender_number = generate_inflated_profile(normalized_rate, empirical_profile)
        syseval.apply_strategy_profile(input_params.player_configuration, profile_after_inflation)

        simulation_output = syseval.run_scenario(simfunction, input_params, simulation_configuration)

        simulation_output_file = "csv/" + desc + "_simulaton_results.csv"
        pd.DataFrame(simulation_output.get_consolidated_output(input_params.player_configuration)).to_csv(
            simulation_output_file)

        logger.info("The simulation output was stored at: " + simulation_output_file)

        performance_metrics = zip(simulation_output.get_time_ratio_per_priority(simdata.SEVERE_PRIORITY),
                                  simulation_output.get_completed_per_real_priority(simdata.SEVERE_PRIORITY),
                                  simulation_output.get_fixed_ratio_per_priority(simdata.SEVERE_PRIORITY,
                                                                                 exclude_open=False),
                                  simulation_output.get_fixed_ratio_per_priority(simdata.SEVERE_PRIORITY,
                                                                                 exclude_open=True))

        regression_data += [{'inflation_rate': inflation_rate,
                             'offender_number': offender_number,
                             'normalized_rate': normalized_rate,
                             'severe_time_ratio': severe_time_ratio,
                             'severe_completed': severe_completed,
                             'severe_fixed_ratio': severe_fixed_ratio,
                             'severe_fixed_ratio_active': severe_fixed_ratio_active
                             } for severe_time_ratio, severe_completed, severe_fixed_ratio, severe_fixed_ratio_active in
                            performance_metrics]

    return pd.DataFrame(regression_data)


def perform_regression_analysis(desc, dataframe):
    """
    Performs the regression analysis, logging the output and generating a plot.
    :param desc: Description of the scenario.
    :param dataframe: Dataframe with performance values.
    :param dependent_variable: Name of the dependant variable.
    :param independent_variable: Name of the independent variable.
    :return: None.
    """

    regression_results = {}
    regression_results['dataframe'] = dataframe

    for dependent_variable in DEPENDENT_VARIABLES:
        detailed_desc = desc + '_' + dependent_variable

        dependent_values = dataframe[dependent_variable]
        independent_values = sm.add_constant(dataframe[INDEPENDENT_VARIABLE])

        ols_instance = sm.OLS(dependent_values, independent_values)
        regression_result = ols_instance.fit()
        logger.info(detailed_desc + " -> regression_result.summary(): " + str(regression_result.summary()))

        plt.clf()
        axis = dataframe.plot(INDEPENDENT_VARIABLE, dependent_variable, style='o')
        plt.xlabel(INDEPENDENT_VARIABLE)
        plt.ylabel(dependent_variable)
        plt.title(detailed_desc)
        sm.graphics.abline_plot(model_results=regression_result, ax=axis)

        file_name = "img/" + detailed_desc + '_regression_analysis.png'
        plt.savefig(file_name)
        logger.info("Image stored at " + file_name)

        regression_results[dependent_variable] = ols_instance

    return regression_results


def do_unsupervised_prioritization(simulation_configuration, input_params, simfunction, empirical_profile,
                                   step):
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration["GATEKEEPER_CONFIG"] = None

    desc = "UNSUPERVISED_PRIORITIZATION"
    logger.info("Starting " + desc + " analysis ...")

    dataframe = get_performance_dataframe(input_params=input_params, simfunction=simfunction,
                                          simulation_configuration=simulation_configuration,
                                          empirical_profile=empirical_profile, step=step, desc=desc)

    return perform_regression_analysis(desc=desc, dataframe=dataframe)


def do_throttling(simulation_configuration, input_params, simfunction, empirical_profile, step):
    simulation_configuration["THROTTLING_ENABLED"] = True

    penalty_values = [1, 3, 5]

    # TODO(cgavidia): Remove later
    penalty_values = [5]

    throttling_results = {}
    for penalty in penalty_values:
        simulation_configuration["INFLATION_FACTOR"] = penalty / 100.0

        desc = "THROTTLING_INF00" + str(penalty)
        logger.info("Starting " + desc + " analysis ...")

        dataframe = get_performance_dataframe(input_params=input_params, simfunction=simfunction,
                                              simulation_configuration=simulation_configuration,
                                              empirical_profile=empirical_profile, step=step, desc=desc)

        throttling_results[desc] = perform_regression_analysis(desc=desc, dataframe=dataframe)

    return throttling_results


def do_gatekeeper(simulation_configuration, input_params, simfunction, empirical_profile, step):
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration['GATEKEEPER_CONFIG'] = penaltyexp.DEFAULT_GATEKEEPER_CONFIG

    success_rates = [90, 100]

    # TODO(cgavidia): Remove later
    success_rates = [90]

    gatekeeper_results = {}

    for success_rate in success_rates:
        normalized_success_rate = success_rate / 100.0
        simulation_configuration["SUCCESS_RATE"] = normalized_success_rate
        input_params.catcher_generator.configure(values=[True, False],
                                                 probabilities=[normalized_success_rate, (1 - normalized_success_rate)])

        desc = "GATEKEEPER_SUCC" + str(success_rate)
        logger.info("Starting " + desc + " analysis ...")
        dataframe = get_performance_dataframe(input_params=input_params, simfunction=simfunction,
                                              simulation_configuration=simulation_configuration,
                                              empirical_profile=empirical_profile, step=step, desc=desc)

        gatekeeper_results[desc] = perform_regression_analysis(desc=desc, dataframe=dataframe)

    return gatekeeper_results

def plot_comparison(plot_configs):
    plt.clf()

    for plot_config in plot_configs:
        plt.plot(plot_config['x_values'], plot_config['fitted_values'], plot_config['color'],
                 label=plot_config['legend'])

    plt.legend()
    plt.ylim(1200, 1500)
    plt.xlim(0, 200)
    plt.xlabel(INDEPENDENT_VARIABLE)
    plt.ylabel('Severe Fixes')
    plt.title('Performance Comparison')

    plt.savefig("img/performance_comparison.png")


def main():
    replications_per_rate = 40
    step = 10

    # TODO(cgavidia): Remove later
    # replications_per_rate = 12
    # step = 40

    logger.info("Experiment configuration: Replications per Inflation Rate " + str(
        replications_per_rate) + " Offset between rates " + str(step))

    simulation_configuration, simfunction, input_params, empirical_profile = syseval.gather_experiment_inputs()
    simulation_configuration['REPLICATIONS_PER_PROFILE'] = replications_per_rate

    uo_regression_results = do_unsupervised_prioritization(simulation_configuration=simulation_configuration,
                                                           simfunction=simfunction,
                                                           input_params=input_params,
                                                           empirical_profile=empirical_profile, step=step)

    throt_regression_results = do_throttling(simulation_configuration=simulation_configuration, simfunction=simfunction,
                                             input_params=input_params, empirical_profile=empirical_profile, step=step)

    gate_regression_results = do_gatekeeper(simulation_configuration=simulation_configuration, simfunction=simfunction,
                                            input_params=input_params, empirical_profile=empirical_profile, step=step)

    plot_comparison([{"x_values": uo_regression_results['dataframe'][INDEPENDENT_VARIABLE],
                      "fitted_values": uo_regression_results['severe_completed'].fit().fittedvalues,
                      "color": "red",
                      "legend": "Unsupervised Prioritization"},
                     {"x_values": throt_regression_results['THROTTLING_INF005']['dataframe'][INDEPENDENT_VARIABLE],
                      "fitted_values": throt_regression_results['THROTTLING_INF005'][
                          'severe_completed'].fit().fittedvalues,
                      "color": "blue",
                      "legend": "Throttling with 0.05 penalty"},
                     {"x_values": gate_regression_results['GATEKEEPER_SUCC90']['dataframe'][INDEPENDENT_VARIABLE],
                      "fitted_values": gate_regression_results['GATEKEEPER_SUCC90'][
                          'severe_completed'].fit().fittedvalues,
                      "color": "green",
                      "legend": "Gatekeeper with 10% error rate"}])


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    logger.info("Execution time in seconds: " + str(time.time() - start_time))
