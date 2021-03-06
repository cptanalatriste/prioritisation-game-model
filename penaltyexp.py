"""
This modules performs the experiments for finding the optimal value of the inflation
penalty
"""
import logging

import pandas as pd

import time
from fractions import Fraction

import simdata
import simdriver
import simmodel

import payoffgetter
import gtutils
import simutils
import gtconfig

if gtconfig.is_windows:
    import winsound

DEFAULT_GATEKEEPER_CONFIG = {'review_time_gen': simutils.ConstantGenerator(name="review_time_gen", value=20.0 / 60.0),
                             'capacity': 2}

logger = gtconfig.get_logger("config_experiments", "config_experiments.txt", level=logging.INFO)


def get_profile_for_plotting(equilibrium_list):
    """
    List of symmetric equilibrium

    :param equilibrium_list:
    :return: Representative profile.
    """
    selected_profile = None

    sample_team = 0
    strategy_of_interest = simmodel.SIMPLE_INFLATE_STRATEGY

    for profile in equilibrium_list:

        if selected_profile is None:
            selected_profile = profile

        else:
            print "profile[sample_team][strategy_of_interest] ", profile[sample_team][strategy_of_interest]
            print "selected_profile[sample_team][strategy_of_interest] ", selected_profile[sample_team][
                strategy_of_interest]

            if Fraction(profile[sample_team][strategy_of_interest]) > Fraction(
                    selected_profile[sample_team][strategy_of_interest]):
                selected_profile = profile

    return profile


def simulate_and_obtain_equilibria(input_params, game_configuration, prefix="", file_name=None, priority_queue=False,
                                   dev_team_factor=1.0):
    """
    Given a game configuration, it computes the heuristic payoff matrix and calculates the symmetric Nash Equilibrium
    :param input_params: Simulation parameters.
    :param game_configuration: Game configuration.
    :param prefix: Prefix for the generated file.
    :return: A list of equilibria, including the symmetric ones.
    """
    equilibrium_list = payoffgetter.run_simulation(strategy_maps=input_params.strategy_maps,
                                                   strategies_catalog=input_params.strategies_catalog,
                                                   player_configuration=input_params.player_configuration,
                                                   dev_team_size=input_params.dev_team_size,
                                                   resolution_time_gen=input_params.resolution_time_gen,
                                                   teams=input_params.teams,
                                                   ignored_gen=input_params.ignored_gen,
                                                   reporter_gen=input_params.reporter_gen,
                                                   target_fixes=input_params.target_fixes,
                                                   batch_size_gen=input_params.batch_size_gen,
                                                   interarrival_time_gen=input_params.interarrival_time_gen,
                                                   priority_generator=input_params.priority_generator,
                                                   catcher_generator=input_params.catcher_generator,
                                                   priority_queue=priority_queue,
                                                   dev_team_factor=dev_team_factor,
                                                   game_configuration=game_configuration)

    symmetric_equilibrium = [profile for profile in equilibrium_list if gtutils.is_symmetric_equilibrium(profile)]
    logger.info("Symmetric Equilibria: " + str(len(symmetric_equilibrium)))

    if file_name is None:
        file_name = "csv/" + prefix + "_equilibrium_results.csv"
    pd.DataFrame(
        [gtutils.get_equilibrium_as_dict(identifier=prefix, profile=profile) for profile in equilibrium_list]).to_csv(
        file_name)
    logger.info("Equilibrium results stored in " + str(file_name))

    return equilibrium_list, symmetric_equilibrium


def do_penalty_experiments(input_params, game_configuration, priority_queue=False, dev_team_factor=1.0):
    """
    Executes the simulation model using different settings for the penalty factor, and calculates the equilibrium under
    each of this conditions.

    :param input_params: Simulation inputs.
    :param game_configuration: Game parameters.
    :return: None.
    """
    game_configuration['THROTTLING_ENABLED'] = True
    game_configuration["SUCCESS_RATE"] = 0.95
    input_params.catcher_generator.configure(values=[True, False], probabilities=[game_configuration["SUCCESS_RATE"], (
            1 - game_configuration["SUCCESS_RATE"])])

    experiment_results = []

    inflation_factors = gtconfig.inflation_factors
    for raw_inflation in inflation_factors:
        game_configuration['INFLATION_FACTOR'] = raw_inflation

        print "Current inflation factor: ", game_configuration['INFLATION_FACTOR']

        prefix = "INF" + str(game_configuration['INFLATION_FACTOR'] * 100) + "_PRIQUEUE_" + str(
            priority_queue) + "_DEVFACTOR_" + str(dev_team_factor)
        equilibrium_list, symmetric_equilibrium = simulate_and_obtain_equilibria(input_params, game_configuration,
                                                                                 prefix=prefix,
                                                                                 priority_queue=priority_queue,
                                                                                 dev_team_factor=dev_team_factor)

        profile_for_plotting = get_profile_for_plotting(symmetric_equilibrium)
        sample_team = 0
        inflation_at_equilibrium = float(Fraction(profile_for_plotting[sample_team][simmodel.SIMPLE_INFLATE_STRATEGY]))

        results = {"total_equilibrium": len(equilibrium_list),
                   "symmetric equilibrium": len(symmetric_equilibrium),
                   "inflation_factor": game_configuration['INFLATION_FACTOR'],
                   "inflation_at_equilibrium": inflation_at_equilibrium,
                   "priority_queue": priority_queue,
                   "dev_team_factor": dev_team_factor}

        logger.info("results: " + str(results))

        experiment_results.append(results)

    dataframe = pd.DataFrame(experiment_results)
    project_prefix = "ALL"

    if game_configuration['PROJECT_FILTER'] is not None and len(game_configuration['PROJECT_FILTER']) > 0:
        project_prefix = "_".join(game_configuration['PROJECT_FILTER'])

    filename = "csv/" + project_prefix + "_" + prefix + "_penalty_experiment_results.csv"
    dataframe.to_csv(filename, index=False)
    logger.info("Penalty experiment results stored in " + filename)


def do_gatekeeper_experiments(input_params, game_configuration, priority_queue=False, dev_team_factor=1.0):
    """
    Performs the Gatekeeper game with several levels of success rate for inflation detection.
    :param input_params: Simulation inputs.
    :param game_configuration: Game parameters.
    :return: None
    """

    success_rates = gtconfig.success_rates
    for success_rate in success_rates:
        game_configuration['SUCCESS_RATE'] = success_rate

        input_params.catcher_generator.configure(values=[True, False], probabilities=[success_rate, (1 - success_rate)])

        prefix = "GATEKEEPER_SUCCESS" + str(game_configuration['SUCCESS_RATE']) + "_PRIQUEUE_" + str(
            priority_queue) + "_DEVFACTOR_" + str(dev_team_factor)
        simulate_and_obtain_equilibria(input_params, game_configuration, prefix=prefix, priority_queue=priority_queue,
                                       dev_team_factor=dev_team_factor)


def analyse_project(project_list, enhanced_dataframe, valid_projects, replications_per_profile=1000,
                    use_empirical=False, use_heuristic=True, priority_queue=False, dev_team_factor=1.0):
    """

    :param project_list:
    :param enhanced_dataframe:
    :param valid_projects:
    :param replications_per_profile: The default value of 1000 is a recommendation from Software Process Dynamics by R. Madachy
    :param use_empirical:
    :return:
    """
    logger.info("Analyzing " + str(valid_projects) + " with " + str(
        replications_per_profile) + " replications and use_empirical=" +
                str(use_empirical) + " priority_queue=" + str(priority_queue) + " dev_team_factor=" + str(
        dev_team_factor))

    game_configuration = dict(payoffgetter.DEFAULT_CONFIGURATION)
    game_configuration['PROJECT_FILTER'] = project_list
    game_configuration[
        'REPLICATIONS_PER_PROFILE'] = replications_per_profile

    game_configuration['HEURISTIC_STRATEGIES'] = use_heuristic
    game_configuration['EMPIRICAL_STRATEGIES'] = use_empirical

    do_gatekeeper = gtconfig.do_gatekeeper
    do_throttling = gtconfig.do_throttling

    input_params = payoffgetter.prepare_simulation_inputs(enhanced_dataframe, valid_projects,
                                                          game_configuration, priority_queue=priority_queue)

    if do_throttling:
        logger.info("Starting Throttling penalty experiments...")
        game_configuration['THROTTLING_ENABLED'] = True
        do_penalty_experiments(input_params, game_configuration, priority_queue=priority_queue,
                               dev_team_factor=dev_team_factor)

    if do_gatekeeper:
        print "Starting gatekeeper analysis ..."

        game_configuration['THROTTLING_ENABLED'] = False
        game_configuration['GATEKEEPER_CONFIG'] = DEFAULT_GATEKEEPER_CONFIG

        do_gatekeeper_experiments(input_params, game_configuration, priority_queue=priority_queue,
                                  dev_team_factor=dev_team_factor)


def main():
    """
    Initial execution point
    :return:
    """
    logger.info("Loading information from " + simdata.ALL_ISSUES_CSV)
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    logger.info("Adding calculated fields...")
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    valid_projects = simdriver.get_valid_projects(enhanced_dataframe=enhanced_dataframe,
                                                  exclude_self_fix=gtconfig.exclude_self_fix)

    per_project = False
    consolidated = True

    replications_per_profile = gtconfig.replications_per_profile

    for priority_queue in gtconfig.priority_queues:
        for dev_team_factor in gtconfig.dev_team_factors:

            logger.info("GAME CONFIGURATION: Priority Queue " + str(priority_queue) + " Dev Team Factor: " + str(
                dev_team_factor))

            if per_project:
                logger.info("Running per-project analysis ...")
                for project in valid_projects:
                    analyse_project([project], enhanced_dataframe, valid_projects,
                                    replications_per_profile=replications_per_profile,
                                    use_empirical=gtconfig.use_empirical_strategies,
                                    use_heuristic=gtconfig.use_heuristic_strategies,
                                    priority_queue=priority_queue,
                                    dev_team_factor=dev_team_factor)

            if consolidated:
                analyse_project(None, enhanced_dataframe, valid_projects,
                                replications_per_profile=replications_per_profile,
                                use_empirical=gtconfig.use_empirical_strategies,
                                use_heuristic=gtconfig.use_heuristic_strategies,
                                priority_queue=priority_queue,
                                dev_team_factor=dev_team_factor)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows and gtconfig.beep:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
