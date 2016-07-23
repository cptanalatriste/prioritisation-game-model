"""
This module is a discrete event simulation model for the bug reporting process
"""

from SimPy.Simulation import *
from SimPy.SimPlot import *

CRITICAL_PRIORITY = 100


class TestingContext:
    """
    This class will produce the characteristics of the discovered defects.
    """

    def __init__(self, resolution_time_gen, priority_gen):
        self.resolution_time_gen = resolution_time_gen
        self.priority_gen = priority_gen

    def get_fix_effort(self):
        """
        Return the time required for a bug to be fixed.
        :return: Effort required to fix a bug.
        """
        return self.resolution_time_gen.generate()

    def get_priority(self):
        """
        Returns the priority of the produced bug report.
        :return: The priority of the generate bug.
        """
        return self.priority_gen.generate()


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def __init__(self, name, reports_produced, interarrival_time_gen, testing_context):
        Process.__init__(self)
        self.name = name
        self.reports_produced = reports_produced

        self.interarrival_time_gen = interarrival_time_gen
        self.testing_context = testing_context

    def start_reporting(self, developer_resource, resol_time_monitor):
        """
        Activates a number of bug reports according to an inter-arrival time.
        :param resol_time_monitor: Monitor for the resolution time.
        :param developer_resource: The Development Team resource.
        :return: None
        """
        for key in range(self.reports_produced):
            bug_report = BugReport(name="Report-" + str(key), reporter=self.name,
                                   fix_effort=self.testing_context.get_fix_effort(),
                                   priority=self.testing_context.get_priority())

            activate(bug_report,
                     bug_report.arrive(developer_resource=developer_resource,
                                       resol_time_monitor=resol_time_monitor))

            interarrival_time = self.get_interarrival_time()

            yield hold, self, interarrival_time

    def get_interarrival_time(self):
        """
        Returns the time to wait after producing another bug report.
        :return: Time to hold before next report.
        """
        return self.interarrival_time_gen.generate()


class BugReport(Process):
    """
    A project member whose main responsibility is bug reporting.
    """

    def __init__(self, name, reporter, fix_effort, priority):
        Process.__init__(self)
        self.name = name
        self.reporter = reporter
        self.fix_effort = fix_effort
        self.priority = priority

    def arrive(self, developer_resource, resol_time_monitor):
        """
        The Process Execution Method for the Bug Reported process.
        :return:
        """
        arrival_time = now()
        pending_bugs = len(developer_resource.waitQ)
        # print arrival_time, ": Report ", self.name, " arrived. Effort: ", self.fix_effort, " Priority: ", self.priority,\
        #     "Pending bugs: ", pending_bugs

        yield request, self, developer_resource, self.priority

        # print now(), ": Report ", self.name, " ready for fixing after ", waiting_time, " waiting."

        yield hold, self, self.fix_effort
        yield release, self, developer_resource

        resol_time = now() - arrival_time
        resol_time_monitor.observe(resol_time)
        # print now(), ": Report ", self.name, " got fixed. "


def run_model(team_capacity, report_number, interarrival_time_gen, resolution_time_gen, priority_gen, max_time):
    start_time = 0.0

    # The Resource is non-preemptable. It won't interrupt ongoing fixes.
    developer_resource = Resource(capacity=team_capacity, name="dev_team", unitName="developer", qType=PriorityQ,
                                  preemptable=False, monitored=True)
    resol_time_monitor = Monitor()

    initialize()
    testing_context = TestingContext(resolution_time_gen=resolution_time_gen, priority_gen=priority_gen)
    bug_reporter = BugReportSource(name="a_tester", interarrival_time_gen=interarrival_time_gen,
                                   testing_context=testing_context, reports_produced=report_number)
    activate(bug_reporter,
             bug_reporter.start_reporting(developer_resource=developer_resource,
                                          resol_time_monitor=resol_time_monitor), at=start_time)

    simulate(until=max_time)

    # plt = SimPlot()
    # plt.plotStep(developer_resource.waitMon, color="red", width=2)
    # plt.mainloop()

    return resol_time_monitor


def main():
    """
    Initial execution point.
    :return: None
    """
    max_time = 400.0

    report_number = 20
    mean_arrival_time = 10
    mean_fix_effort = 12.0

    # TODO The team capacity should also come from a probability distribution.
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
