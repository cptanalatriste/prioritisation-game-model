from collections import Counter

import pandas as pd
import simdata

import sqlite3

JIRA_DB = \
    "E:\OneDrive\phd2\jira_db\issue_repository.db"


def get_issue_type(issue_key):
    connection = sqlite3.connect(JIRA_DB)
    sql_query = "Select t.name from Issue i, IssueType t where i.key='{key}' and i.issueTypeId = t.id".format(
        key=issue_key)

    cursor = connection.cursor()
    cursor.execute(sql_query)
    issue_type = cursor.fetchone()
    connection.close()

    return issue_type[0]


def main():
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)
    print "total_issues: ", len(all_issues)

    issue_types = []

    for _, issue_row in all_issues.iterrows():
        issue_key = issue_row[simdata.ISSUE_KEY_COLUMN]
        issue_type = get_issue_type(issue_key)

        # print "issue_type", issue_type
        issue_types.append(issue_type)

    type_counter = Counter(issue_types)

    print "type_counter", type_counter
    print "Bug percetage", type_counter["Bug"] / float(len(all_issues)) * 100


if __name__ == "__main__":
    main()
