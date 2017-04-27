"""
This modules performs the experiments for finding the optimal value of the inflation
penalty
"""

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


def simulate_and_obtain_equilibria(input_params, game_configuration, prefix="", file_name=None):
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
                                                   game_configuration=game_configuration)

    symmetric_equilibrium = [profile for profile in equilibrium_list if gtutils.is_symmetric_equilibrium(profile)]
    print "Symmetric Equilibria: ", len(symmetric_equilibrium)

    if file_name is None:
        file_name = "csv/" + prefix + "_equilibrium_results.csv"
    pd.DataFrame(
        [gtutils.get_equilibrium_as_dict(identifier=prefix, profile=profile) for profile in equilibrium_list]).to_csv(
        file_name)
    print "Equilibrium results stored in ", file_name

    return equilibrium_list, symmetric_equilibrium


def do_penalty_experiments(input_params, game_configuration):
    """
    Executes the simulation model using different settings for the penalty factor, and calculates the equilibrium under
    each of this conditions.

    :param input_params: Simulation inputs.
    :param game_configuration: Game parameters.
    :return: None.
    """
    game_configuration['THROTTLING_ENABLED'] = True

    experiment_results = []

    inflation_factors = gtconfig.inflation_factors
    for raw_inflation in inflation_factors:
        game_configuration['INFLATION_FACTOR'] = raw_inflation

        print "Current inflation factor: ", game_configuration['INFLATION_FACTOR']
        equilibrium_list, symmetric_equilibrium = simulate_and_obtain_equilibria(input_params, game_configuration,
                                                                                 prefix="INF" + str(game_configuration[
                                                                                                        'INFLATION_FACTOR'] * 100))

        profile_for_plotting = get_profile_for_plotting(symmetric_equilibrium)
        sample_team = 0
        inflation_at_equilibrium = float(Fraction(profile_for_plotting[sample_team][simmodel.SIMPLE_INFLATE_STRATEGY]))

        results = {"total_equilibrium": len(equilibrium_list),
                   "symmetric equilibrium": len(symmetric_equilibrium),
                   "inflation_factor": game_configuration['INFLATION_FACTOR'],
                   "inflation_at_equilibrium": inflation_at_equilibrium}

        print "results ", results

        experiment_results.append(results)

    dataframe = pd.DataFrame(experiment_results)
    prefix = "ALL"

    if game_configuration['PROJECT_FILTER'] is not None and len(game_configuration['PROJECT_FILTER']) > 0:
        prefix = "_".join(game_configuration['PROJECT_FILTER'])

    filename = "csv/" + prefix + "_penalty_experiment_results.csv"
    dataframe.to_csv(filename, index=False)
    print "Penalty experiment results stored in ", filename


def do_gatekeeper_experiments(input_params, game_configuration):
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
        simulate_and_obtain_equilibria(input_params, game_configuration,
                                       prefix="GATEKEEPER_SUCCESS" + str(game_configuration['SUCCESS_RATE']))


def analyse_project(project_list, enhanced_dataframe, valid_projects, replications_per_profile=1000,
                    use_empirical=False, use_heuristic=True):
    """

    :param project_list:
    :param enhanced_dataframe:
    :param valid_projects:
    :param replications_per_profile: The default value of 1000 is a recommendation from Software Process Dynamics by R. Madachy
    :param use_empirical:
    :return:
    """
    print "Analyzing ", valid_projects, " with ", replications_per_profile, " replications and use_empirical=", use_empirical

    game_configuration = dict(payoffgetter.DEFAULT_CONFIGURATION)
    game_configuration['PROJECT_FILTER'] = project_list
    game_configuration[
        'REPLICATIONS_PER_PROFILE'] = replications_per_profile

    game_configuration['HEURISTIC_STRATEGIES'] = use_heuristic
    game_configuration['EMPIRICAL_STRATEGIES'] = use_empirical

    # TODO(cgavidia): Only for testing
    do_gatekeeper = gtconfig.do_gatekeeper
    do_throttling = gtconfig.do_throttling

    input_params = payoffgetter.prepare_simulation_inputs(enhanced_dataframe, valid_projects,
                                                          game_configuration)

    if do_throttling:
        print "Starting Throtling penalty experiments..."
        game_configuration['THROTTLING_ENABLED'] = True
        do_penalty_experiments(input_params, game_configuration)

    if do_gatekeeper:
        print "Starting gatekeeper analysis ..."

        gatekeepers = 2
        review_time_minutes = 20.0
        review_time_gen = simutils.ConstantGenerator(name="review_time_gen", value=review_time_minutes / 60.0)

        game_configuration['THROTTLING_ENABLED'] = False
        game_configuration['GATEKEEPER_CONFIG'] = {'review_time_gen': review_time_gen,
                                                   'capacity': gatekeepers}

        do_gatekeeper_experiments(input_params, game_configuration)


def main():
    """
    Initial execution point
    :return:
    """
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    valid_projects = simdriver.get_valid_projects(enhanced_dataframe)

    per_project = False
    consolidated = True

    replications_per_profile = gtconfig.replications_per_profile

    if per_project:
        print "Running per-project analysis ..."
        for project in valid_projects:
            analyse_project([project], enhanced_dataframe, valid_projects,
                            replications_per_profile=replications_per_profile,
                            use_empirical=gtconfig.use_empirical_strategies,
                            use_heuristic=gtconfig.use_heuristic_strategies)

    if consolidated:
        analyse_project(None, enhanced_dataframe, valid_projects, replications_per_profile=replications_per_profile,
                        use_empirical=gtconfig.use_empirical_strategies, use_heuristic=gtconfig.use_heuristic_strategies)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
