"""
This module is a discrete event simulation model for the bug reporting process
"""

from SimPy.Simulation import *
from random import expovariate, seed


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def start_reporting(self, report_number, mean_arrival_time, mean_fix_effort, developer_resource):
        """
        Activates a number of bug reports according to an interarrival time.
        :param report_number: Number of bugs to report.
        :param mean_arrival_time: Time between reports.
        :param developer_resource: The Development Team resource.
        :return: None
        """
        for key in range(report_number):
            bug_report = BugReport(name="Report-" + str(key))
            activate(bug_report,
                     bug_report.arrive(mean_fix_effort=mean_fix_effort, developer_resource=developer_resource))

            yield hold, self, self.get_interarrival_time(mean_arrival_time)

    def get_interarrival_time(self, mean_arrival_time):
        """
        Returns the time to wait after producing another bug report.
        :param mean_arrival_time: Mean interarrival time.
        :return: Time to hold before next report.
        """
        return expovariate(1.0 / mean_arrival_time)


class BugReport(Process):
    """
    A project member whose main responsibility is bug reporting.
    """

    def arrive(self, mean_fix_effort, developer_resource):
        """
        The Process Execution Method for the Bug Reported process.
        :param bug_effort: Effort required for the reported defect. In days
        :return:
        """
        arrival_time = now()
        print arrival_time, ": Report ", self.name, " arrived."

        yield request, self, developer_resource
        waiting_time = now() - arrival_time
        print now(), ": Report ", self.name, " ready for fixing after ", waiting_time, " waiting."

        yield hold, self, self.get_fix_effort(mean_fix_effort)
        yield release, self, developer_resource
        print now(), ": Report ", self.name, " got fixed. "

    def get_fix_effort(self, mean_fix_effort):
        """
        Return the time required for a bug to be fixed.
        :param mean_fix_effort: Mean fix effort.
        :return: Effort required to fix this bug.
        """
        return expovariate(1.0 / mean_fix_effort)


def main():
    """
    Initial execution point.
    :return: None
    """
    max_time = 400.0

    report_number = 5
    mean_arrival_time = 10.0
    mean_fix_effort = 12.0

    random_seed = 12345
    seed(random_seed)

    developer_resource = Resource(name="dev_team", unitName="developer")
    initialize()
    bug_reporter = BugReportSource(name="a_tester")
    activate(bug_reporter,
             bug_reporter.start_reporting(report_number=report_number, mean_arrival_time=mean_arrival_time,
                                          mean_fix_effort=mean_fix_effort,
                                          developer_resource=developer_resource), at=0.0)

    simulate(until=max_time)


if __name__ == "__main__":
    main()
