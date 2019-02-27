"""
This module contains the functions to consolidate simulation output information for game theoretic analysis purposes.
"""

import simdata
import pandas as pd

import scipy.stats as st
import logging

import gtconfig

logger = gtconfig.get_logger("sim_data_analysis", "sim_data_analysis.txt", level=logging.INFO)


def get_payoff_score(reporter_info, score_map, priority_based=True):
    """
    Calculates the payoff per user after an specific run.

    :param priority_based: If True, each defect solved has a specific score based on the priority.
    :param severe_completed: Severe bugs resolved.
    :param non_severe_completed: Non Severe bugs resolved.
    :param normal_completed: Normal bugs resolved.
    :return: Payoff score.
    """
    reported_severe_fixed = reporter_info['reported_severe_fixed']
    reported_nonsevere_fixed = reporter_info['reported_nonsevere_fixed']

    if priority_based:

        reported_severe_fixed = reporter_info['reported_severe_fixed']
        reported_nonsevere_fixed = reporter_info['reported_nonsevere_fixed']

        score = reported_severe_fixed * score_map[simdata.SEVERE_PRIORITY] + reported_nonsevere_fixed * score_map[
            simdata.NON_SEVERE_PRIORITY]
    else:
        score = reported_severe_fixed + reported_nonsevere_fixed

    return score


def consolidate_payoff_results(period, reporter_configuration, simulation_output, score_map, priority_based):
    """
    Gather per-run metrics according to a simulation result.
    :param resolved_per_reporter: Resolved issues per priority, including a priority detail.
    :param period: Description of the period.
    :param reporter_configuration: List of reporter configuration.
    :param completed_per_reporter: List containing completed reports per reporter per run.
    :param bugs_per_reporter: List containing found reports per reporter per priority per run.
    :param reports_per_reporter: ist containing reported (sic) reports per reporter per priority per run.
    :return: Consolidated metrics in a list.
    """

    simulation_results = simulation_output.get_consolidated_output(reporter_configuration)

    logger.info(
        "Payoff function parameters: Priority-based " + str(priority_based) + " Severe Score: " + str(
            score_map[simdata.SEVERE_PRIORITY]) + " Non-Severe Score " + str(score_map[simdata.NON_SEVERE_PRIORITY]))

    for reporter_info in simulation_results:
        reporter_info["period"] = period
        payoff_score = get_payoff_score(reporter_info=reporter_info, score_map=score_map, priority_based=priority_based)
        reporter_info["payoff_score"] = payoff_score

    return simulation_results


def get_team_metrics(file_prefix, game_period, teams, overall_dataframes, number_of_teams):
    """
    Analizes the performance of the team based on fixed issues, according to a scenario description.

    :param teams: Number of teams in the game.
    :param file_prefix: Strategy profile descripcion.
    :param game_period: Game period description.
    :param overall_dataframe: Dataframe with run information.
    :return: List of outputs per team
    """
    runs = overall_dataframes[0]['run'].unique()

    consolidated_result = []

    logger.info("Dataframes under analysis: " + str(len(overall_dataframes)) + ". Number of runs: " + str(
        len(runs)) + " Number of teams: " + str(teams))

    for run in runs:

        team_results = {}
        for team in range(teams):

            for index, overall_dataframe in enumerate(overall_dataframes):

                period_reports = overall_dataframe[overall_dataframe['period'] == game_period]
                reports_in_run = period_reports[period_reports['run'] == run]

                team_run_reports = reports_in_run[reports_in_run['reporter_team'] == team]
                if len(team_run_reports.index) > 0:
                    team_resolved = team_run_reports['reported_completed'].sum()
                    team_reported = team_run_reports['reported'].sum()
                    team_score = team_run_reports['payoff_score'].sum()

                    team_results[team] = {"team_resolved": team_resolved,
                                          "team_reported": team_reported,
                                          "team_score": team_score}

        simulation_result = {"run": run}

        for team_index in range(number_of_teams):
            team_prefix = "team_" + str(team_index + 1) + "_"

            simulation_result[team_prefix + "results"] = team_results[team_index]['team_resolved']
            simulation_result[team_prefix + "reports"] = team_results[team_index]['team_reported']
            simulation_result[team_prefix + "score"] = team_results[team_index]['team_score']

        consolidated_result.append(simulation_result)

    consolidated_dataframe = pd.DataFrame(consolidated_result)
    consolidated_dataframe.to_csv("csv/" + file_prefix + "_consolidated_result.csv", index=False)

    team_averages = []

    for team_index in range(number_of_teams):
        score_column = "team_" + str(team_index + 1) + "_score"

        mean = consolidated_dataframe[score_column].mean()
        team_averages.append(int(mean))

        # This is the procedure found -and validated- on Chapter 2 of Introduction to Discrete Event Simulation by
        # Theodore Allen
        sem = st.sem(consolidated_dataframe[score_column])
        df = consolidated_dataframe[score_column].count() - 1
        alpha = 0.95

        interval = st.t.interval(alpha=alpha, df=df, loc=mean, scale=sem)
        print file_prefix, ": Confidence Interval Analysis for Team ", team_index, " mean=", mean, " sem=", sem, " df=", df, " alpha=", \
            alpha, " interval=", interval

        logger.info(file_prefix + ": Confidence Interval Analysis for Team " + str(team_index) + " mean=" + str(
            mean) + " sem=" + str(sem) + " df=" + str(df) + " alpha=", \
                    str(alpha) + " interval=" + str(interval))

    return [str(team_avg) for team_avg in team_averages]
