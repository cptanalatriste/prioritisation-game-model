"""
This is a module for detecting default priority overuse.
"""

import pandas as pd
import numpy as np

import simdata
import simdriver
import gtconfig

import matplotlib

if not gtconfig.is_windows:
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def get_default_usage_data(enhanced_dataframe):
    """
    Returns a dataframe contaning the non-default usage per project.

    The reports considered are from non-drive-by reporters and excludes self-fixed reports.

    :param enhanced_dataframe: Bug Report dataframe.
    :return: Project dataframe.
    """

    project_lists = enhanced_dataframe["Project Key"].unique()
    project_dist = []

    for project_key in project_lists:
        valid_reports = simdriver.get_valid_reports([project_key], enhanced_dataframe)

        reporters_config = simdriver.get_reporter_configuration(valid_reports)
        engaged_testers = [reporter_config['name'] for reporter_config in reporters_config]
        valid_reports = simdata.filter_by_reporter(valid_reports, engaged_testers)

        total_reports = len(valid_reports.index)
        non_default_reports = valid_reports[valid_reports['Priority'] != "Major"]
        total_non_default = len(non_default_reports.index)

        non_default_ratio = None
        if total_reports > 0:
            non_default_ratio = float(total_non_default) / total_reports

        project_dist.append({'project_key': project_key,
                             'total_reports': total_reports,
                             'total_non_default': total_non_default,
                             'non_default_ratio': non_default_ratio})

    project_dataframe = pd.DataFrame(project_dist)
    return project_dataframe


def main():
    print "Loading information from ", simdata.ALL_ISSUES_CSV
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "Adding calculated fields..."
    enhanced_dataframe = simdata.enhace_report_dataframe(all_issues)

    project_dataframe = get_default_usage_data(enhanced_dataframe)
    project_dataframe = project_dataframe.sort(columns="non_default_ratio")

    non_default_series = project_dataframe['non_default_ratio']
    config = {'title': 'Non-Default Priority Usage',
              'xlabel': 'Non-Default Percentge',
              'ylabel': 'Number of Projects',
              'file_name': 'default_abuse.png'}
    simdata.launch_histogram(non_default_series, config=config)

    print "project_dataframe \n", project_dataframe

    plt.clf()
    sorted_values = non_default_series.sort_values()
    sorted_values[len(sorted_values)] = sorted_values.iloc[-1]

    cdf_values = np.linspace(0.0, 1.0, len(sorted_values))
    cdf_series = pd.Series(cdf_values, index=sorted_values)

    cdf_series.plot(drawstyle='steps')
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel('CDF')
    plt.savefig("img/" + 'cdf_default_abuse.png')


if __name__ == "__main__":
    main()
