"""
This modules contain some data analysis do detect players actions and strategies.
"""

import siminput
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    inflation_impact = ""
    if inflation_percentage < 0.1:
        inflation_impact = "Low"
        return inflation_impact
    elif inflation_percentage < 0.4:
        inflation_impact = "Moderate"
    else:
        inflation_impact = "High"

    target_priority = corrected_issues[SIMPLE_ORIG_PRIORITY_COLUMN].mode().iloc[0]
    inflation_degree_num = prioritity_equivalence[target_priority] - prioritity_equivalence[real_priority]

    inflation_degree = ""
    if inflation_degree_num == 1:
        inflation_degree = "Inflation"
    elif inflation_degree_num > 1:
        inflation_degree = "Hyperinflation"
    else:
        inflation_degree = "Deflation"

    return inflation_impact + "-" + inflation_degree


if __name__ == "__main__":
    print "Starting analysis ..."
    all_issues = pd.read_csv(siminput.ALL_ISSUES_CSV)
    created_date_column = 'Parsed Created Date'
    all_issues[created_date_column] = all_issues.apply(siminput.parse_create_date, axis=1)

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

    priority_changer_column = "Priority Changer"
    third_party_changer = (~all_issues[priority_changer_column].isnull()) & \
                          (all_issues['Reported By'] != all_issues[priority_changer_column])
    print "Priority changed by non-reporter: \n", third_party_changer.value_counts()

    issues_validated_priority = all_issues.loc[third_party_changer]
    issues_by_project = issues_validated_priority['Project Key'].value_counts()

    print "Project counts:  \n", issues_by_project
    top_projects = issues_by_project.iloc[:3]

    print "Top-3 projects with validated priorities: \n", top_projects

    inflation_catalog = []
    for project_key, _ in top_projects.iteritems():
        project_filter = issues_validated_priority['Project Key'] == project_key

        project_changed_issues = issues_validated_priority[project_filter]
        print "Validated priorities for project ", project_key, ": ", len(project_changed_issues.index)

        creation_dates = project_changed_issues[created_date_column]
        min_creation_date = creation_dates.min()
        max_creation_date = creation_dates.max()
        print "Validated priorities creation range: ", min_creation_date, " - ", max_creation_date

        reported_by_column = 'Reported By'
        reporters = project_changed_issues[reported_by_column].value_counts()
        top_reporters = reporters.iloc[:5]

        print "Reporters with Priorities adjustments: \n", top_reporters

        for reporter, _ in top_reporters.iteritems():
            reporter_filter = all_issues[reported_by_column] == reporter
            date_filter = (all_issues[created_date_column] <= max_creation_date) & (
                all_issues[created_date_column] >= min_creation_date)

            issues_for_analysis = all_issues[reporter_filter & date_filter]
            total_issues = len(issues_for_analysis.index)
            print "Issues for analysis for ", reporter, ": ", total_issues

            for simplified_priority, _ in simple_priorities.iteritems():
                issues_per_priority = issues_for_analysis[
                    issues_for_analysis[simple_priority_column] == simplified_priority]
                total_per_priority = len(issues_per_priority.index)

                corrected_filter = (~issues_per_priority[priority_changer_column].isnull()) & \
                                   (issues_per_priority[priority_changer_column] != reporter)
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
