"""
This modules analyses priority behaviour in our dataset
"""
import logging
import winsound

import gtconfig
import time

import pandas as pd

import simdata
import simdriver

logger = gtconfig.get_logger("priority_analysis", "priority_analysis.txt", level=logging.INFO)


def main():
    logger.info("Starting priority analysis ...")

    logger.info("Loading information from " + simdata.ALL_ISSUES_CSV)
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    logger.info("Adding calculated fields...")
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    valid_projects = simdriver.get_valid_projects(enhanced_dataframe, threshold=simdriver.VALID_THRESHOLD)

    _, training_dataset, _, _ = simdriver.split_bug_dataset(enhanced_dataframe, test_size=simdriver.TEST_SIZE,
                                                            valid_projects=valid_projects)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    finally:
        if gtconfig.is_windows:
            winsound.Beep(2500, 1000)

    print "Execution time in seconds: ", (time.time() - start_time)
