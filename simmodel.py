"""
This module is a discrete event simulation model for the bug reporting process
"""

from SimPy.Simulation import *
from SimPy.SimPlot import *
from random import expovariate, seed

CRITICAL_PRIORITY = 100


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def start_reporting(self, report_number, mean_arrival_time, mean_fix_effort, developer_resource, wait_monitor):
        """
        Activates a number of bug reports according to an interarrival time.
        :param report_number: Number of bugs to report.
        :param mean_arrival_time: Time between reports.
        :param developer_resource: The Development Team resource.
        :return: None
        """
        for key in range(report_number):
            bug_report = BugReport(name="Report-" + str(key))
            priority = self.get_priority()

            activate(bug_report,
                     bug_report.arrive(mean_fix_effort=mean_fix_effort, priority=priority,
                                       developer_resource=developer_resource,
                                       wait_monitor=wait_monitor))

            yield hold, self, self.get_interarrival_time(mean_arrival_time)

    def get_interarrival_time(self, mean_arrival_time):
        """
        Returns the time to wait after producing another bug report.
        :param mean_arrival_time: Mean interarrival time.
        :return: Time to hold before next report.
        """
        # TODO This probability distribution and its parameters should come from the data.
        return expovariate(1.0 / mean_arrival_time)

    def get_priority(self):
        """
        Returns the priority of the produced bug report.
        :return:
        """
        # TODO The reported priority depends on the tester strategy. And probably from the ground-truth priority.
        return 0


class BugReport(Process):
    """
    A project member whose main responsibility is bug reporting.
    """

    def arrive(self, mean_fix_effort, developer_resource, wait_monitor, priority):
        """
        The Process Execution Method for the Bug Reported process.
        :param bug_effort: Effort required for the reported defect. In days
        :return:
        """
        arrival_time = now()
        pending_bugs = len(developer_resource.waitQ)
        print arrival_time, ": Report ", self.name, " arrived. Pending bugs: ", pending_bugs

        yield request, self, developer_resource, priority
        waiting_time = now() - arrival_time
        wait_monitor.observe(waiting_time)

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
        # TODO This probability distribution and its parameters should come from the data.
        return expovariate(1.0 / mean_fix_effort)


def run_model(random_seed, team_capacity, report_number, mean_arrival_time, mean_fix_effort, max_time):
    start_time = 0.0
    seed(random_seed)

    # The Resource is non-preemptable. It won't interrupt ongoing fixes.
    developer_resource = Resource(capacity=team_capacity, name="dev_team", unitName="developer", qType=PriorityQ,
                                  preemptable=False, monitored=True)
    wait_monitor = Monitor()

    initialize()
    bug_reporter = BugReportSource(name="a_tester")
    activate(bug_reporter,
             bug_reporter.start_reporting(report_number=report_number, mean_arrival_time=mean_arrival_time,
                                          mean_fix_effort=mean_fix_effort,
                                          developer_resource=developer_resource,
                                          wait_monitor=wait_monitor), at=start_time)

    # TODO: Only for testing purposes.
    critical_bug = BugReport(name="Report-CRITICAL")
    activate(critical_bug,
             critical_bug.arrive(mean_fix_effort=mean_fix_effort, developer_resource=developer_resource,
                                 wait_monitor=wait_monitor,
                                 priority=CRITICAL_PRIORITY), at=100.0)

    simulate(until=max_time)

    print "Wait Monitor Avg: ", developer_resource.waitMon.timeAverage(), \
        " Active Monitor Avg: ", developer_resource.actMon.timeAverage()

    plt = SimPlot()
    plt.plotStep(developer_resource.waitMon, color="red", width=2)
    plt.mainloop()

    return wait_monitor


def main():
    """
    Initial execution point.
    :return: None
    """
    max_time = 400.0

    report_number = 20
    mean_arrival_time = 10
    mean_fix_effort = 12.0
    team_capacity = 1

    # random_seeds = [393939, 31555999, 777999555, 319999771]
    random_seeds = [393939]
    for a_seed in random_seeds:
        wait_monitor = run_model(a_seed, team_capacity, report_number, mean_arrival_time, mean_fix_effort, max_time)
        print "Average wait for ", wait_monitor.count(), " completitions is ", wait_monitor.mean()

        wait_histogram = wait_monitor.histogram(low=0.0, high=200.0, nbins=20)
        # plt = SimPlot()
        # plt.plotHistogram(wait_histogram, xlab="Total Bug Time (days)", title="Total Bug Time", color="red", width=2)
        # plt.mainloop()


if __name__ == "__main__":
    main()
