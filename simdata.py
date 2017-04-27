"""
This module contains the data providing logic for the simulation random variate streams.
"""
import dateutil.parser
import pandas as pd
import numpy as np
import datetime
import pytz
import gtconfig

import matplotlib

if not gtconfig.is_windows:
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

GIT_HOME = gtconfig.git_home
ALL_ISSUES_CSV = gtconfig.all_issues_csv

PROJECT_KEY_COUMN = "Project Key"
PRIORITY_CHANGER_COLUMN = "Priority Changer"
CREATED_DATE_COLUMN = 'Parsed Created Date'
RESOLUTION_DATE_COLUMN = 'Parsed Resolution Date'
PRIORITY_CHANGE_TIME_COLUMN = 'Priority Change Time'
PERIOD_COLUMN = 'Month'
ISSUE_KEY_COLUMN = 'Issue Key'
BATCH_COLUMN = 'Batch'
RESOLVED_IN_BATCH_COLUMN = 'Resolved in Batch'
RESOLUTION_TIME_COLUMN = 'Resolution Time'
STATUS_COLUMN = 'Status'

SIMPLE_PRIORITY_COLUMN = 'Simplified Priority'
ORIGINAL_SIMPLE_PRIORITY_COLUMN = 'Original Simplified Priority'
NEW_SIMPLE_PRIORITY_COLUMN = 'New Simplified Priority'

REPORTER_COLUMN = 'Reported By'
RESOLVER_COLUMN = 'JIRA Resolved By'

TIME_FACTOR = 60.0 * 60.0
VALID_RESOLUTION_VALUES = ['Done', 'Implemented', 'Fixed']
RESOLUTION_STATUS = ['Closed', 'Resolved']

SEVERE_PRIORITY = 3
NORMAL_PRIORITY = 2
NON_SEVERE_PRIORITY = 1
SIMPLIFIED_PRIORITIES = {"Blocker": SEVERE_PRIORITY,
                         "Critical": SEVERE_PRIORITY,
                         "Major": NON_SEVERE_PRIORITY,
                         "Minor": NON_SEVERE_PRIORITY,
                         "Trivial": NON_SEVERE_PRIORITY}
SUPPORTED_PRIORITIES = [NON_SEVERE_PRIORITY, SEVERE_PRIORITY]

BATCH_SIZE = 20


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
    Calculates the fix effort in the units defined by time factor. It is defined as the hours between the resolution and
     the "In Progress" status change by the resolver.

    :param report_series: Bug report as Series.
    :return: Fix effort in hours.
    """

    first_contact_str = report_series['Creation Date']
    resolution_date_str = report_series['JIRA Resolved Date']

    if isinstance(first_contact_str, basestring) and isinstance(resolution_date_str, basestring):
        first_contact = dateutil.parser.parse(first_contact_str)
        resolution_date = dateutil.parser.parse(resolution_date_str)

        if first_contact < resolution_date:
            return (resolution_date - first_contact).total_seconds() / TIME_FACTOR

    return None


def get_priority_change_time(report_series):
    """
    Calculates the time for priority change in the units defined by time factor. It is defined as the hours between the
    report creation and the priority change.

    :param report_series: Bug report as Series.
    :return: Fix effort in hours.
    """
    create_date_str = report_series['Creation Date']
    priority_change_str = report_series['Priority Change Date']

    if isinstance(create_date_str, basestring) and isinstance(priority_change_str, basestring):
        create_date = dateutil.parser.parse(create_date_str)
        priority_change_date = dateutil.parser.parse(priority_change_str)

        if create_date < priority_change_date:
            return (priority_change_date - create_date).total_seconds() / TIME_FACTOR

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


def period_identifier(report_series, batch_size=BATCH_SIZE):
    """
    Generates a period identifier based on report information.
    :param report_series: Bug report information.
    :return: Period identifier.
    """

    index_value = report_series.name

    if batch_size > 0:
        batch_identifier = int(index_value) / int(batch_size)
    else:
        batch_identifier = 1

    return batch_identifier


def include_batch_information(bug_reports, target_fixes=20, only_with_commits=True, only_valid_resolution=True):
    """
    Includes the column for grouping bug reports in batches.
    :param bug_reports:
    :return: Dataframe with a batch column
    """

    print "Starting batch assignment for ", len(bug_reports.index), " bug reports ..."
    with_refreshed_index = bug_reports.sort_values(by=[CREATED_DATE_COLUMN], ascending=[1])
    with_refreshed_index = with_refreshed_index.reset_index()

    current_batch = 0
    current_batch_start = None
    batches = []
    resolved_in_batch = []
    batch_starts = []

    report_counter = 0

    for _, report_series in with_refreshed_index.iterrows():

        if current_batch_start is None:
            current_batch_start = report_series[CREATED_DATE_COLUMN]
            batch_starts.append(current_batch_start)

        batches.append(current_batch)
        report_counter += 1
        current_creation_date = report_series[CREATED_DATE_COLUMN]

        previous_reports = with_refreshed_index[
            (with_refreshed_index[CREATED_DATE_COLUMN] >= current_batch_start) &
            (with_refreshed_index[CREATED_DATE_COLUMN] <= current_creation_date) &
            (with_refreshed_index[RESOLUTION_DATE_COLUMN] <= current_creation_date)]

        current_fixes = filter_resolved(previous_reports, only_with_commits, only_valid_resolution)

        if len(current_fixes.index) >= target_fixes:
            current_batch += 1
            report_counter = 0
            current_batch_start = None

    print "The bug reports where grouped in ", current_batch + 1, " batches."
    print "Starting resoluton in batch status calculation ..."

    with_refreshed_index[BATCH_COLUMN] = pd.Series(batches, index=with_refreshed_index.index)

    previous_batch = 0
    batch_resolved_count = 0
    for _, report_series in with_refreshed_index.iterrows():

        current_batch = report_series[BATCH_COLUMN]
        if previous_batch != current_batch:
            batch_resolved_count = 0

        batch_start = batch_starts[current_batch]
        batch_reports = with_refreshed_index[with_refreshed_index[BATCH_COLUMN] == current_batch]
        batch_end = max(batch_reports[CREATED_DATE_COLUMN].dropna().values)

        resolved = False
        if (resolved_definition(report_series, only_with_commits, only_valid_resolution)) and \
                (batch_resolved_count < target_fixes) and \
                (report_series[RESOLUTION_DATE_COLUMN] is not None) and \
                (batch_start <= report_series[RESOLUTION_DATE_COLUMN] <= batch_end):
            resolved = True
            batch_resolved_count += 1

        resolved_in_batch.append(resolved)
        previous_batch = current_batch

    with_refreshed_index[RESOLVED_IN_BATCH_COLUMN] = pd.Series(resolved_in_batch, index=with_refreshed_index.index)
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
    bug_reports[PRIORITY_CHANGE_TIME_COLUMN] = bug_reports.apply(get_priority_change_time, axis=1)

    bug_reports[SIMPLE_PRIORITY_COLUMN] = bug_reports['Priority'].replace(SIMPLIFIED_PRIORITIES)
    bug_reports[ORIGINAL_SIMPLE_PRIORITY_COLUMN] = bug_reports['Original Priority']
    bug_reports[NEW_SIMPLE_PRIORITY_COLUMN] = bug_reports['New Priority'].replace(SIMPLIFIED_PRIORITIES)

    # bug_reports[SIMPLE_PRIORITY_COLUMN] = bug_reports[SIMPLE_PRIORITY_COLUMN].fillna(NON_SEVERE_PRIORITY)
    # bug_reports[ORIGINAL_SIMPLE_PRIORITY_COLUMN] = bug_reports[ORIGINAL_SIMPLE_PRIORITY_COLUMN].fillna(
    #     NON_SEVERE_PRIORITY)
    # bug_reports[NEW_SIMPLE_PRIORITY_COLUMN] = bug_reports[NEW_SIMPLE_PRIORITY_COLUMN].fillna(NON_SEVERE_PRIORITY)

    return bug_reports


def filter_by_create_date(bug_reports, start_date, end_date, is_bucket=False):
    """
    Filters a bug report dataframe according to a range for creation date.
    :param bug_reports: Bug report dataframe.
    :param start_date:  Start date.
    :param end_date: End date.
    :return: Filtered dataframe.
    """
    return filter_by_date_range(CREATED_DATE_COLUMN, bug_reports, start_date, end_date, is_bucket)


def filter_by_date_range(column_name, bug_reports, start_date, end_date, is_bucket=False):
    """
    Filters by a column and a specific date range.
    :param is_bucket: If its true, the inequality regardind the end date is < instead of <=
    :param column_name: Column name.
    :param bug_reports: Bug dataframe.
    :param start_date: Range start.
    :param end_date: Range end.
    :return: Filtered dataframe.
    """
    date_filter = (bug_reports[column_name] <= end_date) & (
        bug_reports[column_name] >= start_date)

    if is_bucket:
        date_filter = (bug_reports[column_name] < end_date) & (
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


def resolved_definition(bug_report, only_with_commits=True, only_valid_resolution=True):
    """
    Given a bug report series, returns True if it is considered resolved.
    :param bug_report: Bug Report Series.
    :param only_with_commits: True if it should have commits related to it.
    :param only_valid_resolution: True if the resolution value implies development effort.
    :return:
    """
    is_resolved = bug_report[STATUS_COLUMN] in RESOLUTION_STATUS

    if only_valid_resolution:
        is_resolved = is_resolved and bug_report['Resolution'] in VALID_RESOLUTION_VALUES

    if only_with_commits:
        is_resolved = is_resolved and bug_report['Commits'] > 0

    return is_resolved


def filter_resolved(bug_reports, only_with_commits=True, only_valid_resolution=True):
    """
    Return the issues that are Closed/Resolved with a valid resolution and with commits in Git.
    :param bug_reports: Original dataframe
    :return: Only resolved issues.
    """

    resolved_issues = bug_reports[
        bug_reports.apply(lambda report: resolved_definition(report, only_with_commits, only_valid_resolution), axis=1)]

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
    report_dates = report_dates.sort_values()

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
