"""
This modules performs the experiments for finding the optimal value of the inflation
penalty
"""

import pandas as pd

import winsound
import time
from fractions import Fraction

import simdata
import simdriver
import simmodel

import payoffgetter
import gtutils


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


def simulate_and_obtain_equilibria(input_params, game_configuration):
    equilibrium_list = payoffgetter.run_simulation(strategy_maps=input_params.strategy_maps,
                                                   strategies_catalog=input_params.strategies_catalog,
                                                   player_configuration=input_params.player_configuration,
                                                   dev_team_size=input_params.dev_team_size,
                                                   bugs_by_priority=input_params.bugs_by_priority,
                                                   resolution_time_gen=input_params.resolution_time_gen,
                                                   dev_team_bandwith=input_params.dev_team_bandwith,
                                                   teams=input_params.teams,
                                                   game_configuration=game_configuration)

    symmetric_equilibrium = [profile for profile in equilibrium_list if gtutils.is_symmetric_equilibrium(profile)]
    print "Symmetric Equilibria: ", len(symmetric_equilibrium)

    return equilibrium_list, symmetric_equilibrium


def do_penalty_experiments(input_params, game_configuration):
    game_configuration['THROTTLING_ENABLED'] = True

    experiment_results = []
    for inflation_factor in range(0, 5):
        game_configuration['INFLATION_FACTOR'] = float(inflation_factor) / 20

        print "Current inflation factor: ", game_configuration['INFLATION_FACTOR']
        equilibrium_list, symmetric_equilibrium = simulate_and_obtain_equilibria(input_params, game_configuration)

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
    dataframe.to_csv("csv/" + "_".join(game_configuration['PROJECT_FILTER']) + "_penalty_experiment_results.csv",
                     index=False)


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

    input_params = payoffgetter.prepare_simulation_inputs(enhanced_dataframe, valid_projects,
                                                          game_configuration)
    # print "Starting AS-IS Game Analysis ..."
    # game_configuration['THROTTLING_ENABLED'] = False
    #
    # simulate_and_obtain_equilibria(input_params, game_configuration)

    print "Starting Throtling penalty experiments..."
    game_configuration['THROTTLING_ENABLED'] = True
    do_penalty_experiments(input_params, game_configuration)

    # inflation_factor = 0.1
    # print "Starting Throttling Game Analysis with an Inflation Factor of ", inflation_factor
    # game_configuration['THROTTLING_ENABLED'] = True
    # game_configuration['INFLATION_FACTOR'] = inflation_factor
    # simulate_and_obtain_equilibria(input_params, game_configuration)
    #
    # print "Starting gatekeeper analysis ..."
    # game_configuration['THROTTLING_ENABLED'] = False
    # game_configuration['GATEKEEPER_CONFIG'] = {'review_time': 8,
    #                                            'capacity': 1}
    #
    # simulate_and_obtain_equilibria(input_params, game_configuration)


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

    for project in valid_projects:
        analyse_project([project], enhanced_dataframe, valid_projects, replications_per_profile=200,
                        use_empirical=True)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
