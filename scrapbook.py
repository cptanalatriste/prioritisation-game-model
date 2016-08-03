import simdata
import pandas as pd


def main():
    all_issues = pd.read_csv(simdata.ALL_ISSUES_CSV)

    print "len(all_issues.index) ", len(all_issues.index)

    with_validated_priority = simdata.get_modified_priority_bugs(all_issues)
    print "len(with_validated_priority.index)", len(with_validated_priority.index)

    six_projects = simdata.filter_by_project(with_validated_priority,
                                             ['OFBIZ', 'CASSANDRA', 'CLOUDSTACK', 'MAHOUT', 'ISIS', 'SPARK'])
    print "len(six_projects.index)", len(six_projects.index)


if __name__ == "__main__":
    main()
