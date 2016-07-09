"""
This module is a discrete event simulation model for the bug reporting process
"""

from SimPy.Simulation import *
from SimPy.SimPlot import *
from random import expovariate, seed

CRITICAL_PRIORITY = 100


class TestingContext:
    """
    This class will produce the characteristics of the discovered defects.
    """

    def __init__(self, mean_fix_effort):
        self.mean_fix_effort = mean_fix_effort

    def get_fix_effort(self):
        """
        Return the time required for a bug to be fixed.
        :return: Effort required to fix a bug.
        """
        # TODO This probability distribution and its parameters should come from the data.
        return expovariate(1.0 / self.mean_fix_effort)

    def get_priority(self):
        """
        Returns the priority of the produced bug report.
        :return:
        """
        # TODO The reported priority depends on the tester strategy. And probably from the ground-truth priority.
        return 0


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def __init__(self, name, reports_produced, mean_arrival_time, testing_context):
        Process.__init__(self)
        self.name = name
        self.reports_produced = reports_produced

        # TODO We are still not sure what are the parameters of this probability distributions.
        self.mean_arrival_time = mean_arrival_time
        self.testing_context = testing_context

    def start_reporting(self, developer_resource, wait_monitor):
        """
        Activates a number of bug reports according to an inter-arrival time.
        :param wait_monitor: Monitor for the waiting time.
        :param developer_resource: The Development Team resource.
        :return: None
        """
        for key in range(self.reports_produced):
            bug_report = BugReport(name="Report-" + str(key), fix_effort=self.testing_context.get_fix_effort(),
                                   priority=self.testing_context.get_priority())

            activate(bug_report,
                     bug_report.arrive(developer_resource=developer_resource,
                                       wait_monitor=wait_monitor))

            yield hold, self, self.get_interarrival_time()

    def get_interarrival_time(self):
        """
        Returns the time to wait after producing another bug report.
        :return: Time to hold before next report.
        """
        # TODO This probability distribution and its parameters should come from the data.
        return expovariate(1.0 / self.mean_arrival_time)


class BugReport(Process):
    """
    A project member whose main responsibility is bug reporting.
    """

    def __init__(self, name, fix_effort, priority):
        Process.__init__(self)
        self.name = name
        self.fix_effort = fix_effort
        self.priority = priority

    def arrive(self, developer_resource, wait_monitor):
        """
        The Process Execution Method for the Bug Reported process.
        :param bug_effort: Effort required for the reported defect. In days
        :return:
        """
        arrival_time = now()
        pending_bugs = len(developer_resource.waitQ)
        print arrival_time, ": Report ", self.name, " arrived. Effort: ", self.fix_effort, "Pending bugs: ", pending_bugs

        yield request, self, developer_resource, self.priority
        waiting_time = now() - arrival_time
        wait_monitor.observe(waiting_time)

        print now(), ": Report ", self.name, " ready for fixing after ", waiting_time, " waiting."

        yield hold, self, self.fix_effort
        yield release, self, developer_resource
        print now(), ": Report ", self.name, " got fixed. "


def run_model(random_seed, team_capacity, report_number, mean_arrival_time, mean_fix_effort, max_time):
    start_time = 0.0
    seed(random_seed)

    # The Resource is non-preemptable. It won't interrupt ongoing fixes.
    developer_resource = Resource(capacity=team_capacity, name="dev_team", unitName="developer", qType=PriorityQ,
                                  preemptable=False, monitored=True)
    wait_monitor = Monitor()

    initialize()
    testing_context = TestingContext(mean_fix_effort=mean_fix_effort)
    bug_reporter = BugReportSource(name="a_tester", mean_arrival_time=mean_arrival_time,
                                   testing_context=testing_context, reports_produced=report_number)
    activate(bug_reporter,
             bug_reporter.start_reporting(developer_resource=developer_resource,
                                          wait_monitor=wait_monitor), at=start_time)

    # TODO: Only for testing purposes.
    critical_bug = BugReport(name="Report-CRITICAL", fix_effort=testing_context.get_fix_effort(),
                             priority=CRITICAL_PRIORITY)
    activate(critical_bug,
             critical_bug.arrive(developer_resource=developer_resource,
                                 wait_monitor=wait_monitor), at=100.0)

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

    #TODO The team capacity should also come from a probability distribution.
    team_capacity = 1

    # random_seeds = [393939, 31555999, 777999555, 319999771]
    random_seeds = [393939]
    for a_seed in random_seeds:
        wait_monitor = run_model(random_seed=a_seed, team_capacity=team_capacity, report_number=report_number,
                                 mean_arrival_time=mean_arrival_time, mean_fix_effort=mean_fix_effort,
                                 max_time=max_time)
        print "Average wait for ", wait_monitor.count(), " completitions is ", wait_monitor.mean()

        wait_histogram = wait_monitor.histogram(low=0.0, high=200.0, nbins=20)
        # plt = SimPlot()
        # plt.plotHistogram(wait_histogram, xlab="Total Bug Time (days)", title="Total Bug Time", color="red", width=2)
        # plt.mainloop()


if __name__ == "__main__":
    main()
