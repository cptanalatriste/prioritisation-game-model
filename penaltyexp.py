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

    experiment_results = []

    game_configuration = payoffgetter.DEFAULT_CONFIGURATION
    game_configuration['REDUCING_FACTOR'] = 0.15
    game_configuration['REPLICATIONS_PER_PROFILE'] = 1000
    game_configuration['THROTTLING_ENABLED'] = True

    input_params = payoffgetter.prepare_simulation_inputs(enhanced_dataframe, valid_projects,
                                                          game_configuration)

    for inflation_factor in range(0, 5):
        game_configuration['INFLATION_FACTOR'] = float(inflation_factor) / 20

        print "Current inflation factor: ", game_configuration['INFLATION_FACTOR']

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
    dataframe.to_csv("csv/penalty_experiment_results.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
