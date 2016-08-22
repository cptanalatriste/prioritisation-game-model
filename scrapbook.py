import simdata
import pandas as pd


def main():
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)
    all_issues = simdata.enhace_report_dataframe(all_issues)

    print "len(all_issues.index) ", len(all_issues.index)

    with_validated_priority = simdata.get_modified_priority_bugs(all_issues)
    print "len(with_validated_priority.index)", len(with_validated_priority.index)

    simplified_priorities = with_validated_priority[
        with_validated_priority[simdata.ORIGINAL_SIMPLE_PRIORITY_COLUMN] == with_validated_priority[
            simdata.NEW_SIMPLE_PRIORITY_COLUMN]]
    print "len(simplified_priorities.index) ", len(simplified_priorities.index)

    projects = with_validated_priority['Project Key'].unique()
    print "projects ", projects, "len(projects.index) ", len(projects)

    users = with_validated_priority[simdata.REPORTER_COLUMN].unique()
    print "users ", users, "len(projects.index) ", len(users)

    print "with_validated_priority.columns", with_validated_priority.columns

    six_projects = simdata.filter_by_project(with_validated_priority,
                                             ['OFBIZ', 'CASSANDRA', 'CLOUDSTACK', 'MAHOUT', 'ISIS', 'SPARK'])
    print "len(six_projects.index)", len(six_projects.index)


if __name__ == "__main__":
    main()
