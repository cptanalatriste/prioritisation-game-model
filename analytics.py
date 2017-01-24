"""
This model produces insights from the data. Not related to any simulation task.
"""

import simdata
import datetime
import pandas as pd


def get_bucket_definition(reporting_start, reporting_end, bucket_days):
    """
    Divides a date range in buckets.

    :param reporting_start: Start of the date range.
    :param reporting_end: End of the date range.
    :param bucket_days: Size of the bucket in days.
    :return: Two lists: One for bucket start and the other for bucket end.
    """
    bucket_start = reporting_start
    bucket_end = reporting_start

    bucket_start_list = []
    bucket_end_list = []
    while bucket_end <= reporting_end:
        bucket_start_list.append(bucket_start)

        bucket_end = bucket_start + datetime.timedelta(days=bucket_days)
        bucket_end_list.append(bucket_end)

        bucket_start = bucket_end

    return bucket_start_list, bucket_end_list


def get_project_history(bucket_start_list, bucket_end_list, project_issues):
    """
    According to a bucket definition, it returns the number of issues that got reported on each specific buket
    :param bucket_start_list: List containing the start dates per bucket.
    :param bucket_end_list: List containing the end dates per bucket.
    :param project_issues: List of project issues.
    :return:
    """
    return [len(simdata.filter_by_create_date(project_issues, bucket_start, bucket_end, True)) for
            bucket_start, bucket_end in
            zip(bucket_start_list, bucket_end_list)]


def run_project_analysis(project_keys, issues_in_range):
    """
    Gathers project-related metrics

    :param project_keys: List of keys of the projects to analyse.
    :param issues_in_range:  Dataframe with the issues.
    :return: None
    """

    bucket_days = 30
    bucket_start_list, bucket_end_list = get_bucket_definition(issues_in_range[simdata.CREATED_DATE_COLUMN].min(),
                                                               issues_in_range[simdata.CREATED_DATE_COLUMN].max(),
                                                               bucket_days)

    history_dataframe = pd.DataFrame({'bucket_start_list': bucket_start_list,
                                      'bucket_end_list': bucket_end_list})

    for project_key in project_keys:
        project_issues = simdata.filter_by_project(issues_in_range, [project_key])

        issues = len(project_issues.index)
        reporting_start = project_issues[simdata.CREATED_DATE_COLUMN].min()
        reporting_end = project_issues[simdata.CREATED_DATE_COLUMN].max()

        print "Project ", project_key, ": Issues ", issues, " Reporting Start: ", reporting_start, " Reporting End: ", \
            reporting_end

        history_dataframe[project_key] = get_project_history(bucket_start_list, bucket_end_list, project_issues)

    print "Total issues: ", len(issues_in_range)

    history_file = "csv/" + "_".join(project_keys) + "_project_report_history.csv"
    print "Saving reporting history to ", history_file
    history_dataframe.to_csv(history_file)
