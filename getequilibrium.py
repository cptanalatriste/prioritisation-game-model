"""
This module exposes the equilibrium analysis for bug reporting with 
command-line argument support.

It is intended to be used on the unsupervised prioritization experiments.
"""

import time
import sys
import pandas as pd

import simdata
import simdriver
import payoffgetter
import gtconfig
import penaltyexp

if gtconfig.is_windows:
    import winsound

TWINS_REDUCTION = False


def get_base_configuration():
    """
    Produce the configuration parameters that are common to all experiment executions.
    :return: 
    """

    gtconfig.report_stream_batching = False
    gtconfig.simple_reporting_model = True

    game_configuration = dict(payoffgetter.DEFAULT_CONFIGURATION)
    game_configuration['PROJECT_FILTER'] = None
    game_configuration['REPLICATIONS_PER_PROFILE'] = gtconfig.replications_per_profile
    game_configuration['HEURISTIC_STRATEGIES'] = True
    game_configuration['EMPIRICAL_STRATEGIES'] = False
    game_configuration['THROTTLING_ENABLED'] = False
    game_configuration['GATEKEEPER_CONFIG'] = None
    game_configuration['SUCCESS_RATE'] = None
    game_configuration['TWINS_REDUCTION'] = False
    game_configuration['PLAYER_CRITERIA'] = 'TOP_FROM_TEAMS'
    game_configuration['ENABLE_RECYCLING'] = False
    game_configuration['ALL_EQUILIBRIA'] = False

    return game_configuration


def main(parameter_list, game_configuration):
    testers, developers, target_bugs, file_name = parameter_list
    print "testers: ", testers, "developers: ", developers, "target_bugs: ", target_bugs, " file_name: ", file_name

    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)
    all_valid_projects = simdriver.get_valid_projects(enhanced_dataframe)

    if not TWINS_REDUCTION:
        game_configuration["NUMBER_OF_TEAMS"] = int(testers)

    input_params = payoffgetter.prepare_simulation_inputs(enhanced_dataframe=enhanced_dataframe,
                                                          all_project_keys=all_valid_projects,
                                                          game_configuration=game_configuration)

    input_params.dev_team_size = int(developers)
    input_params.target_fixes = int(target_bugs)

    penaltyexp.simulate_and_obtain_equilibria(input_params, game_configuration, file_name=file_name)


if __name__ == "__main__":
    start_time = time.time()
    try:
        game_configuration = get_base_configuration()
        main(sys.argv[1:], game_configuration)

    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
