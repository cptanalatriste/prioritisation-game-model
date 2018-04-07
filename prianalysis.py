"""
This modules analyses priority behaviour in our dataset
"""
import logging
import math
import winsound

import pandas as pd
import numpy as np
from scipy import stats

import gtconfig
import time

import simdata
import simdriver

logger = gtconfig.get_logger("priority_analysis", "priority_analysis.txt", level=logging.INFO)


def get_effect_size(one_sample, other_sample):
    """
    Return's the Cohen's d statistic
    :return:
    """

    one_sample_count, other_sample_count = one_sample.count() - 1, other_sample.count() - 1

    pooled_std = math.sqrt((one_sample_count * one_sample.std() ** 2 + other_sample_count * other_sample.std() ** 2) / (
            one_sample_count + other_sample_count - 2))

    effect_size = (one_sample.mean() - other_sample.mean()) / pooled_std

    return effect_size


def main():
    logger.info("Starting priority analysis ...")

    logger.info("Loading information from " + simdata.ALL_ISSUES_CSV)
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    logger.info("Adding calculated fields...")
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    valid_projects = simdriver.get_valid_projects(enhanced_dataframe, threshold=simdriver.VALID_THRESHOLD)

    _, training_issues, _, _ = simdriver.split_bug_dataset(enhanced_dataframe, test_size=simdriver.TEST_SIZE,
                                                           valid_projects=valid_projects)

    priority_sample = training_issues[simdata.SIMPLE_PRIORITY_COLUMN]
    counts_per_priority = priority_sample.value_counts()
    logger.info("Simplified Priorities in Training Range: \n " + str(counts_per_priority))

    all_resolved_issues = simdata.filter_resolved(training_issues, only_with_commits=True,
                                                  only_valid_resolution=True)

    samples_per_priority = {}

    for priority in priority_sample.unique():
        if not np.isnan(priority):
            priority_resolved = all_resolved_issues[all_resolved_issues[simdata.SIMPLE_PRIORITY_COLUMN] == priority]

            resolution_time_sample = priority_resolved[simdata.RESOLUTION_TIME_COLUMN].dropna()

            desc = "Priority_" + str(priority)
            logger.info("Resolution times in Training Range for " + desc + ": \n" +
                        str(resolution_time_sample.describe()))

            samples_per_priority[priority] = resolution_time_sample

    t_statistic, p_value = stats.ttest_ind(samples_per_priority[simdata.NON_SEVERE_PRIORITY],
                                           samples_per_priority[simdata.SEVERE_PRIORITY],
                                           equal_var=False)

    logger.info("Welch t-test result: t_statistic " + str(t_statistic) + " p_value " + str(p_value))
    logger.info("Effect size (Cohen's d): " + str(get_effect_size(samples_per_priority[simdata.NON_SEVERE_PRIORITY],
                                                                  samples_per_priority[simdata.SEVERE_PRIORITY])))

    threshold = 0.05

    if p_value > threshold:
        logger.info("We CANNOT REJECT the null hypothesis of identical average scores")
    else:
        logger.info("We REJECT the null hypothesis of equal averages")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
