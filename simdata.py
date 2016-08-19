"""
This module contains the data providing logic for the simulation random variate streams.
"""
import dateutil.parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz

ALL_ISSUES_CSV = "C:\Users\Carlos G. Gavidia\git\github-data-miner\UNFILTERED\Release_Counter_UNFILTERED.csv"

PRIORITY_CHANGER_COLUMN = "Priority Changer"
CREATED_DATE_COLUMN = 'Parsed Created Date'
RESOLUTION_DATE_COLUMN = 'Parsed Resolution Date'
PERIOD_COLUMN = 'Month'
BATCH_COLUMN = 'Batch'
RESOLUTION_TIME_COLUMN = 'Resolution Time'
SIMPLE_PRIORITY_COLUMN = 'Simplified Priority'

REPORTER_COLUMN = 'Reported By'
RESOLVER_COLUMN = 'JIRA Resolved By'

TIME_FACTOR = 60.0 * 60.0
VALID_RESOLUTION_VALUES = ['Done', 'Implemented', 'Fixed']

SEVERE_PRIORITY = 3
NORMAL_PRIORITY = 2
NON_SEVERE_PRIORITY = 1
BATCH_SIZE = 50


def launch_histogram(data_points, config=None):
    """
    Launches an histogram of the data points passed as parameter.
    :param data_points: List of data points
    :return: None
    """
    histogram, bin_edges = np.histogram(data_points, bins="auto")

    file_name = None

    plt.clf()
    if config:
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        file_name = config['file_name']

    plt.bar(bin_edges[:-1], histogram, width=(bin_edges[1] - bin_edges[0]))
    plt.xlim(min(bin_edges), max(bin_edges))

    if file_name:
        plt.savefig("img/" + file_name, bbox_inches='tight')
    else:
        plt.show()


def get_resolution_time(report_series):
    """
    Calculates the fix effort in the units defined by time factor. It is defined as the days between the resolution and the "In Progress" status
    change by the resolver.

    :param report_series: Bug report as Series.
    :return: Fix effort in days.
    """

    first_contact_str = report_series['Creation Date']
    resolution_date_str = report_series['JIRA Resolved Date']

    if isinstance(first_contact_str, basestring) and isinstance(resolution_date_str, basestring):
        first_contact = dateutil.parser.parse(first_contact_str)
        resolution_date = dateutil.parser.parse(resolution_date_str)

        if first_contact < resolution_date:
            return (resolution_date - first_contact).total_seconds() / TIME_FACTOR

    return None


def parse_create_date(report_series):
    """
    Transforms the create field in a series that is a date string to a datetime instance.
    :param report_series: The series with the Bug Report info.
    :return: The date as a datetime instance.
    """
    date_string = report_series['Creation Date']
    return dateutil.parser.parse(date_string)


def parse_resolution_date(report_series):
    """
    Transforms the resolution date field in a series that is a date string to a datetime instance.
    :param report_series: The series with the Bug Report info.
    :return: The date as a datetime instance.
    """
    date_string = report_series['JIRA Resolved Date']

    if isinstance(date_string, basestring):
        resolution_date = dateutil.parser.parse(date_string)
        return resolution_date

    return None


def week_of_month(dt):
    """ Returns the week of the month for the specified date.

    From: http://stackoverflow.com/questions/3806473/python-week-number-of-the-month
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    # return int(np.ceil(adjusted_dom / 14.0))
    return 1 if dt.day <= 15 else 2


def period_identifier(report_series):
    """
    Generates a period identifier based on report information.
    :param report_series: Bug report information.
    :return: Period identifier.
    """

    index_value = report_series.name
    batch_size = BATCH_SIZE
    batch_identifier = int(index_value) / int(batch_size)
    return batch_identifier


def date_as_string(report_series):
    """
    Returns a string representation of the created
    :param report_series:
    :return:
    """

    parsed_date = parse_create_date(report_series)
    period_identifier = str(parsed_date.month)

    if len(period_identifier) == 1:
        period_identifier = "0" + period_identifier

    return str(parsed_date.year) + "-" + str(period_identifier)


def filter_by_reporter(bug_reports, reporters):
    """
    From a bug dataframe, filters out based on the reporter name
    :param bug_reports: Bug dataframe.
    :param reporters: Reporter name list.
    :return: Filtered dataframe.
    """
    reporter_filter = bug_reports[REPORTER_COLUMN].isin(reporters)
    return bug_reports.loc[reporter_filter]


def include_batch_information(bug_reports):
    """
    Includes the column for grouping bug reports in batches.
    :param bug_reports:
    :return: Dataframe with a batch column
    """
    with_refreshed_index = bug_reports.sort(columns=[CREATED_DATE_COLUMN], ascending=[1])
    with_refreshed_index = with_refreshed_index.reset_index()
    with_refreshed_index[BATCH_COLUMN] = with_refreshed_index.apply(period_identifier, axis=1)

    return with_refreshed_index


def enhace_report_dataframe(bug_reports):
    """
    Adds additional series to the original report dataframe.
    :param bug_reports: Original dataframe.
    :return: Improved dataframe.
    """
    bug_reports[CREATED_DATE_COLUMN] = bug_reports.apply(parse_create_date, axis=1)
    bug_reports[RESOLUTION_DATE_COLUMN] = bug_reports.apply(parse_resolution_date, axis=1)

    bug_reports[PERIOD_COLUMN] = bug_reports.apply(date_as_string, axis=1)

    bug_reports[RESOLUTION_TIME_COLUMN] = bug_reports.apply(get_resolution_time, axis=1)

    simplified_priorities = {"Blocker": SEVERE_PRIORITY,
                             "Critical": SEVERE_PRIORITY,
                             "Major": NON_SEVERE_PRIORITY,
                             "Minor": NON_SEVERE_PRIORITY,
                             "Trivial": NON_SEVERE_PRIORITY}
    bug_reports[SIMPLE_PRIORITY_COLUMN] = bug_reports['Priority'].replace(simplified_priorities)
    return bug_reports


def filter_by_create_date(bug_reports, start_date, end_date):
    """
    Filters a bug report dataframe according to a range for creation date.
    :param bug_reports: Bug report dataframe.
    :param start_date:  Start date.
    :param end_date: End date.
    :return: Filtered dataframe.
    """
    return filter_by_date_range(CREATED_DATE_COLUMN, bug_reports, start_date, end_date)


def filter_by_date_range(column_name, bug_reports, start_date, end_date):
    """
    Filters by a column and a specific date range.
    :param column_name: Column name.
    :param bug_reports: Bug dataframe.
    :param start_date: Range start.
    :param end_date: Range end.
    :return: Filtered dataframe.
    """
    date_filter = (bug_reports[column_name] <= end_date) & (
        bug_reports[column_name] >= start_date)

    issues_for_analysis = bug_reports[date_filter]
    return issues_for_analysis


def exclude_self_fixes(bug_reports):
    """
    Removes from the datasource the bug that were reported and fixed by the same person.
    :param bug_reports: List of bug reports
    :return: Bug reports without self-fixes.
    """
    third_party_resolver_filter = (~bug_reports[RESOLVER_COLUMN].isnull()) & \
                                  (bug_reports[REPORTER_COLUMN] == bug_reports[RESOLVER_COLUMN])

    clean_bug_reports = bug_reports[~third_party_resolver_filter]
    return clean_bug_reports


def filter_resolved(bug_reports, only_with_commits=True, only_valid_resolution=True):
    """
    Return the issues that are Closed/Resolved with a valid resolution and with commits in Git.
    :param bug_reports: Original dataframe
    :return: Only resolved issues.
    """
    resolved_issues = bug_reports[bug_reports['Status'].isin(['Closed', 'Resolved'])]

    if only_valid_resolution:
        resolved_issues = resolved_issues[resolved_issues['Resolution'].isin(VALID_RESOLUTION_VALUES)]

    if only_with_commits:
        resolved_issues = resolved_issues[resolved_issues['Commits'] > 0]

    return resolved_issues


def filter_by_project(bug_reports, project_keys):
    """
    From a bug report dataframe, it filters the information by project
    :param bug_reports: Bug report dataframe.
    :param project_key: Project key.
    :return: Bug reports for the project.
    """

    project_filter = bug_reports['Project Key'].isin(project_keys)
    project_bug_reports = bug_reports[project_filter]
    return project_bug_reports


def get_modified_priority_bugs(bug_reports):
    """
    Returns the bug reports whose priority was corrected by a non-reporter.

    :param bug_reports: Bug Reports.
    :return: Bug reports with a corrected priority.
    """

    third_party_changer = (~bug_reports[PRIORITY_CHANGER_COLUMN].isnull()) & \
                          (bug_reports[REPORTER_COLUMN] != bug_reports[PRIORITY_CHANGER_COLUMN])

    issues_validated_priority = bug_reports.loc[third_party_changer]
    return issues_validated_priority


def get_distance_in_hours(distance):
    """
    Transforms a time delta to date
    :param param: Time delta.
    :return: Time delta equivalent in days.
    """
    if isinstance(distance, datetime.timedelta):
        time = distance.total_seconds() / TIME_FACTOR
    else:
        time = distance / np.timedelta64(1, 's') / TIME_FACTOR

    return time


def get_interarrival_times(arrival_times, period_start):
    """
    Given a list of report dates, it returns the list corresponding to the interrival times.
    :param arrival_times: List of arrival times.
    :return: List of inter-arrival times.
    """
    interarrival_times = []

    for position, created_date in enumerate(arrival_times):
        if position > 0:
            distance = created_date - arrival_times[position - 1]
            interarrival_times.append(get_distance_in_hours(distance))
        else:
            if isinstance(created_date, np.datetime64):
                created_date = datetime.datetime.utcfromtimestamp(created_date.tolist() / 1e9)
                created_date = pytz.utc.localize(created_date)

            distance = get_distance_in_hours(created_date - period_start)

            if distance > 0:
                interarrival_times.append(distance)

    return pd.Series(data=interarrival_times)


def get_report_batches(bug_reports, window_size=1):
    """
    Return a list of bug report batches, according to a batch size.
    :param bug_reports: Bug report dataframe.
    :param window_size: Size of the window that represents a batch. In DAYS
    :return: List containing batch start and batch count.
    """

    report_dates = bug_reports[CREATED_DATE_COLUMN]
    report_dates = report_dates.order()

    batches = []
    for position, created_date in enumerate(report_dates.values):
        if len(batches) == 0:
            batches.append({"batch_head": created_date,
                            "batch_count": 1})
        else:
            last_batch_head = batches[-1]["batch_head"]
            distance = created_date - last_batch_head

            if hasattr(distance, 'days'):
                distance_in_days = distance.days
            else:
                distance_in_days = distance.astype('timedelta64[D]')
                distance_in_days = distance_in_days / np.timedelta64(1, 'D')

            if distance_in_days <= window_size:
                batches[-1]["batch_count"] += 1
            else:
                batches.append({"batch_head": created_date,
                                "batch_count": 1})

    return batches
