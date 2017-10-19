"""
This module contains the code for regression analysis for the bug reporting
processes considered
"""

import time
import logging
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

import gtconfig
import penaltyexp
import simdata
import syseval

if gtconfig.is_windows:
    import winsound

logger = gtconfig.get_logger("regression_analysis", "regression_analysis.txt", level=logging.INFO)

INDEPENDENT_VARIABLE = 'normalized_value'
DEPENDENT_VARIABLES = ['severe_time_ratio', 'severe_completed', 'severe_fixed_ratio', 'severe_fixed_ratio_active']
DEV_TEAM_RATIO = 1


def apply_gatekeeper_error(independent_variable_value, input_params, empirical_profile, original_team_size,
                           simulation_configuration):
    normalized_success_rate = independent_variable_value / 100.0
    simulation_configuration["SUCCESS_RATE"] = normalized_success_rate
    input_params.catcher_generator.configure(values=[True, False],
                                             probabilities=[normalized_success_rate, (1 - normalized_success_rate)])

    return normalized_success_rate


def apply_inflation_factor(independent_variable_value, input_params, empirical_profile, original_team_size,
                           simulation_configuration):
    normalized_rate = independent_variable_value / 100.0
    profile_after_inflation, offender_number = syseval.generate_inflated_profile(normalized_rate, empirical_profile)
    syseval.apply_strategy_profile(input_params.player_configuration, profile_after_inflation)

    return offender_number


def apply_team_reduction(independent_variable_value, input_params, empirical_profile, original_team_size,
                         simulation_configuration):
    normalized_rate = independent_variable_value / 100.0
    new_team_size = int(original_team_size * normalized_rate)

    input_params.dev_team_size = new_team_size
    return new_team_size


def configure_simulation(independent_variable_value, input_params, empirical_profile, original_team_size,
                         configuration_function, simulation_configuration):
    """
    Adjusts the simulation configuration to the current independent variable value.
    :param independent_variable_value:
    :return:
    """

    return configuration_function(independent_variable_value=independent_variable_value, input_params=input_params,
                                  empirical_profile=empirical_profile, original_team_size=original_team_size,
                                  simulation_configuration=simulation_configuration)


def get_performance_dataframe(input_params, simfunction, simulation_configuration, empirical_profile,
                              original_team_size, step, desc, configuration_function=apply_inflation_factor):
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
    independent_variable_values = range(0, 100, step)

    for independent_variable_value in independent_variable_values:
        logger.info(
            "Simulating " + desc + " with an independent variable  of " + str(independent_variable_value))

        normalized_value = configure_simulation(independent_variable_value, input_params, empirical_profile,
                                                original_team_size,
                                                configuration_function=configuration_function,
                                                simulation_configuration=simulation_configuration)

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

        regression_data += [{'independent_variable_value': independent_variable_value,
                             'normalized_value': normalized_value,
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
                                   original_team_size,
                                   step):
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration["GATEKEEPER_CONFIG"] = None

    desc = "UNSUPERVISED_PRIORITIZATION"
    logger.info("Starting " + desc + " analysis ...")

    dataframe = get_performance_dataframe(input_params=input_params, simfunction=simfunction,
                                          simulation_configuration=simulation_configuration,
                                          empirical_profile=empirical_profile, original_team_size=original_team_size,
                                          step=step, desc=desc)

    return perform_regression_analysis(desc=desc, dataframe=dataframe)


def do_gatekeeper(simulation_configuration, input_params, simfunction, empirical_profile, original_team_size, step):
    simulation_configuration["THROTTLING_ENABLED"] = False
    simulation_configuration['GATEKEEPER_CONFIG'] = penaltyexp.DEFAULT_GATEKEEPER_CONFIG

    queue_configurations = [True, False]
    dev_team_factors = [0.5, 1.0]

    gatekeeper_results = {}

    original_team_size = input_params.dev_team_size
    logger.info("Original team size: " + str(original_team_size))

    for queue_configuration in queue_configurations:

        for dev_team_factor in dev_team_factors:
            logger.info("Using dev team factor " + str(dev_team_factor))
            input_params.dev_team_size = int(original_team_size * dev_team_factor)

            desc = "GATEKEEPER_PRIQUEUE_" + str(queue_configuration) + "_DEV_FACTOR_" + str(dev_team_factor)
            logger.info("Starting " + desc + " analysis ...")

            simulation_configuration["PRIORITY_QUEUE"] = queue_configuration
            logger.info("Using Priority Queue? " + str(queue_configuration))

            dataframe = get_performance_dataframe(input_params=input_params, simfunction=simfunction,
                                                  simulation_configuration=simulation_configuration,
                                                  empirical_profile=empirical_profile,
                                                  original_team_size=original_team_size, step=step, desc=desc,
                                                  configuration_function=apply_gatekeeper_error)

            gatekeeper_results[desc] = perform_regression_analysis(desc=desc, dataframe=dataframe)

    return gatekeeper_results


def do_throttling(simulation_configuration, input_params, simfunction, empirical_profile, original_team_size, step):
    simulation_configuration["THROTTLING_ENABLED"] = True

    penalty_values = [1, 3, 5]

    # TODO(cgavidia): Remove later
    penalty_values = [3]

    throttling_results = {}
    for penalty in penalty_values:
        simulation_configuration["INFLATION_FACTOR"] = penalty / 100.0

        desc = "THROTTLING_INF00" + str(penalty)
        logger.info("Starting " + desc + " analysis ...")

        dataframe = get_performance_dataframe(input_params=input_params, simfunction=simfunction,
                                              simulation_configuration=simulation_configuration,
                                              empirical_profile=empirical_profile,
                                              original_team_size=original_team_size, step=step, desc=desc,
                                              configuration_function=apply_gatekeeper_error)

        throttling_results[desc] = perform_regression_analysis(desc=desc, dataframe=dataframe)

    return throttling_results


def plot_comparison(plot_configs, y_min, y_max, desc):
    """
    Plots several regression lines for the sake of comparison.
    :param plot_configs: Each plot parameters
    :param y_min: Y axis minimum value
    :param y_max: Y axis maximum value
    :param desc: Variable under analysis.
    :return: None
    """
    plt.clf()

    for plot_config in plot_configs:
        plt.plot(plot_config['x_values'], plot_config['fitted_values'], plot_config['color'],
                 label=plot_config['legend'])

    plt.legend()
    plt.ylim(y_min, y_max)
    plt.xlim(0, 175)
    plt.xlabel(INDEPENDENT_VARIABLE)
    plt.ylabel(desc)
    plt.title('Performance Comparison: ' + desc)

    file_name = "img/" + desc + "_performance_comparison.png"
    plt.savefig(file_name)
    logger.info("Performance comparison plot was stored in " + file_name)


def compare_regression_results(uo_regression_results, throt_regression_results, gate_regression_results):
    for performance_metric in DEPENDENT_VARIABLES:

        y_min = 0.0
        y_max = 1.0

        if performance_metric == "severe_completed":
            y_min = 800
            y_max = 1200
        elif performance_metric == "severe_time_ratio":
            y_min = 0.05
            y_max = 0.1

        plot_comparison(plot_configs=[{"x_values": uo_regression_results['dataframe'][INDEPENDENT_VARIABLE],
                                       "fitted_values": uo_regression_results[performance_metric].fit().fittedvalues,
                                       "color": "red",
                                       "legend": "Unsupervised Prioritization"},
                                      {"x_values": throt_regression_results['THROTTLING_INF005']['dataframe'][
                                          INDEPENDENT_VARIABLE],
                                       "fitted_values": throt_regression_results['THROTTLING_INF005'][
                                           performance_metric].fit().fittedvalues,
                                       "color": "blue",
                                       "legend": "Throttling with 0.05 penalty"},
                                      {"x_values": gate_regression_results['GATEKEEPER_SUCC90']['dataframe'][
                                          INDEPENDENT_VARIABLE],
                                       "fitted_values": gate_regression_results['GATEKEEPER_SUCC90'][
                                           performance_metric].fit().fittedvalues,
                                       "color": "green",
                                       "legend": "Gatekeeper with 10% error rate"}], desc=performance_metric,
                        y_min=y_min, y_max=y_max)


def main():
    replications_per_rate = 120
    step = 20

    logger.info("Experiment configuration: Replications per Inflation Rate " + str(
        replications_per_rate) + " Offset between rates " + str(step))

    simulation_configuration, simfunction, input_params, empirical_profile = syseval.gather_experiment_inputs()

    original_team_size = input_params.dev_team_size

    input_params.dev_team_size = int(input_params.dev_team_size * DEV_TEAM_RATIO)

    logger.info("The original dev team size is " + str(original_team_size) + " . The size under analysis is " + str(
        input_params.dev_team_size))

    simulation_configuration['REPLICATIONS_PER_PROFILE'] = replications_per_rate

    gate_regression_results = do_gatekeeper(simulation_configuration=simulation_configuration, simfunction=simfunction,
                                            input_params=input_params, empirical_profile=empirical_profile,
                                            original_team_size=original_team_size, step=step)

    # So far, we are only concerned in the regression analysis for the Gatekeeper process
    # uo_regression_results = do_unsupervised_prioritization(simulation_configuration=simulation_configuration,
    #                                                        simfunction=simfunction,
    #                                                        input_params=input_params,
    #                                                        empirical_profile=empirical_profile,
    #                                                        original_team_size=original_team_size, step=step)
    #
    # throt_regression_results = do_throttling(simulation_configuration=simulation_configuration, simfunction=simfunction,
    #                                          input_params=input_params, empirical_profile=empirical_profile,
    #                                          original_team_size=original_team_size, step=step)
    #
    # compare_regression_results(uo_regression_results=uo_regression_results,
    #                            throt_regression_results=throt_regression_results,
    #                            gate_regression_results=gate_regression_results)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    logger.info("Execution time in seconds: " + str(time.time() - start_time))
