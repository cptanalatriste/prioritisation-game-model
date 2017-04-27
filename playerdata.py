"""
This modules contain some data analysis do detect players actions and strategies.
"""

import simdata
import pandas as pd
import numpy as np
import gtconfig

import matplotlib

if not gtconfig.is_windows:
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

SIMPLE_ORIG_PRIORITY_COLUMN = 'Simplified Original Priority'


def get_inflation_behaviour(inflation_percentage, real_priority, corrected_issues):
    """
    Classifies the inflation behavior. The threshold were build thorugh the visual inspection of the histogram of the
    inflation levels of top-reporters for top-projects.

    :param corrected_issues: List of real_priority with priority corrections.
    :param real_priority: Real priority under analysis.
    :param inflation_percentage: Inflation percentage.
    :return: Behaviour category.
    """

    prioritity_equivalence = {"Severe": 3,
                              "Regular": 2,
                              "Non-Severe": 1}

    if inflation_percentage < 0.1:
        inflation_impact = "Low"
        return inflation_impact
    elif inflation_percentage < 0.4:
        inflation_impact = "Moderate"
    else:
        inflation_impact = "High"

    target_priority = corrected_issues[SIMPLE_ORIG_PRIORITY_COLUMN].mode().iloc[0]
    inflation_degree_num = prioritity_equivalence[target_priority] - prioritity_equivalence[real_priority]

    if inflation_degree_num == 1:
        inflation_degree = "Inflation"
    elif inflation_degree_num > 1:
        inflation_degree = "Hyperinflation"
    else:
        inflation_degree = "Deflation"

    return inflation_impact + "-" + inflation_degree


if __name__ == "__main__":
    print "Starting analysis ..."
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    all_issues = simdata.enhace_report_dataframe(all_issues)
    simplified_priorities = {"Blocker": "Severe",
                             "Critical": "Severe",
                             "Major": "Regular",
                             "Minor": "Non-Severe",
                             "Trivial": "Non-Severe"}
    simple_priority_column = 'Simplified Priority'
    all_issues[simple_priority_column] = all_issues['Priority'].replace(simplified_priorities)

    all_issues[SIMPLE_ORIG_PRIORITY_COLUMN] = all_issues['Original Priority'].replace(simplified_priorities)

    print "Unfiltered issues: ", len(all_issues.index)
    simple_priorities = all_issues[simple_priority_column].value_counts()
    print "Priority distribution: \n", simple_priorities

    issues_validated_priority = simdata.get_modified_priority_bugs(all_issues)
    issues_by_project = issues_validated_priority['Project Key'].value_counts()

    print "Project counts:  \n", issues_by_project
    top_projects = issues_by_project.iloc[:3]

    print "Top-3 projects with validated priorities: \n", top_projects

    inflation_catalog = []
    for project_key, _ in top_projects.iteritems():
        project_changed_issues = simdata.filter_by_project(issues_validated_priority, [project_key])
        print "Validated priorities for project ", project_key, ": ", len(project_changed_issues.index)

        creation_dates = project_changed_issues[simdata.CREATED_DATE_COLUMN]
        min_creation_date = creation_dates.min()
        max_creation_date = creation_dates.max()
        print "Validated priorities creation range: ", min_creation_date, " - ", max_creation_date

        reported_by_column = 'Reported By'
        reporters = project_changed_issues[reported_by_column].value_counts()
        top_reporters = reporters.iloc[:5]

        print "Reporters with Priorities adjustments: \n", top_reporters

        for reporter, _ in top_reporters.iteritems():
            reporter_filter = all_issues[reported_by_column] == reporter
            issues_for_analysis = simdata.filter_by_create_date(all_issues, min_creation_date, max_creation_date)
            issues_for_analysis = issues_for_analysis[reporter_filter]
            total_issues = len(issues_for_analysis.index)
            print "Issues for analysis for ", reporter, ": ", total_issues

            for simplified_priority, _ in simple_priorities.iteritems():
                issues_per_priority = issues_for_analysis[
                    issues_for_analysis[simple_priority_column] == simplified_priority]
                total_per_priority = len(issues_per_priority.index)

                corrected_filter = (~issues_per_priority[simdata.PRIORITY_CHANGER_COLUMN].isnull()) & \
                                   (issues_per_priority[simdata.PRIORITY_CHANGER_COLUMN] != reporter)
                corrected_issues = issues_per_priority[corrected_filter]

                correction_report = " [ "
                for reported_priority, count in corrected_issues[
                    SIMPLE_ORIG_PRIORITY_COLUMN].value_counts().iteritems():
                    count_percentage = 0.0 if count == 0 else float(count) / total_per_priority
                    correction_report += reported_priority + ": " + str(count) + " (" + str(
                        count_percentage) + " %) \t "

                correction_report += " ] "

                total_corrections = len(corrected_issues.index)
                correction_percent = float(total_corrections) / total_per_priority if total_per_priority != 0 else 0.0

                inflation_catalog.append(correction_percent)

                print simplified_priority, " issues: ", total_per_priority, " (", float(
                    total_per_priority) / total_issues, " %) with corrections: ", total_corrections, " (", correction_percent, \
                    " % ", get_inflation_behaviour(correction_percent,
                                                   simplified_priority,
                                                   corrected_issues), " ) DETAIL: ", correction_report

    show_inflation_hist = False

    if show_inflation_hist:
        histogram, bin_edges = np.histogram(inflation_catalog, bins="auto")
        plt.bar(bin_edges[:-1], histogram, width=(bin_edges[1] - bin_edges[0]))
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.show()
