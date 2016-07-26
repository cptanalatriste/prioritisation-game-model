"""
This module is a discrete event simulation model for the bug reporting process
"""

from SimPy.Simulation import *
from SimPy.SimPlot import *

import numpy as np

CRITICAL_PRIORITY = 100

NOT_INFLATE_STRATEGY = 'NOT_INFLATE'
INFLATE_STRATEGY = 'INFLATE'


class TestingContext:
    """
    This class will produce the characteristics of the discovered defects.
    """

    def __init__(self, resolution_time_gen, priority_gen, bug_level):
        self.resolution_time_gen = resolution_time_gen
        self.priority_gen = priority_gen
        self.bug_level = bug_level

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
        return self.priority_gen.generate()[0]


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def __init__(self, reporter_config=None, testing_context=None):
        Process.__init__(self)
        self.name = reporter_config['name']
        self.interarrival_time_gen = reporter_config['interarrival_time_gen']
        self.batch_size_gen = reporter_config['batch_size_gen']
        self.strategy = reporter_config['strategy']
        self.testing_context = testing_context

    def start_reporting(self, developer_resource, resol_time_monitor):
        """
        Activates a number of bug reports according to an inter-arrival time.
        :param resol_time_monitor: Monitor for the resolution time.
        :param developer_resource: The Development Team resource.
        :return: None
        """
        bug_level = self.testing_context.bug_level

        batch_size = self.get_batch_size()
        while bug_level.amount >= batch_size:
            yield get, self, bug_level, batch_size

            report_key = "Report-" + str(bug_level.amount)
            bug_report = BugReport(name=report_key, reporter=self.name,
                                   fix_effort=self.testing_context.get_fix_effort(),
                                   priority=self.get_report_priority())

            activate(bug_report,
                     bug_report.arrive(developer_resource=developer_resource,
                                       resol_time_monitor=resol_time_monitor))

            interarrival_time = self.get_interarrival_time()
            yield hold, self, interarrival_time

            batch_size = self.get_batch_size()

    def get_report_priority(self):
        real_priority = self.testing_context.get_priority()
        priority_for_report = real_priority
        return priority_for_report

    def get_interarrival_time(self):
        """
        Returns the time to wait after producing another bug report.
        :return: Time to hold before next report.
        """
        return self.interarrival_time_gen.generate()

    def get_batch_size(self):
        """
        Returns the number of bugs to be contained in the batch report.
        :return: Number of bug reports.
        """
        return int(np.asscalar(self.batch_size_gen.generate()[0]))


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

    def arrive(self, developer_resource, resol_time_monitor, debug=False):
        """
        The Process Execution Method for the Bug Reported process.
        :return:
        """
        arrival_time = now()

        if debug:
            pending_bugs = len(developer_resource.waitQ)
            print arrival_time, ": Report ", self.name, " arrived. Effort: ", self.fix_effort, " Priority: ", self.priority, \
                "Pending bugs: ", pending_bugs

        yield request, self, developer_resource, self.priority

        if debug:
            print now(), ": Report ", self.name, " ready for fixing. "

        yield hold, self, self.fix_effort
        yield release, self, developer_resource

        resol_time = now() - arrival_time
        resol_time_monitor.observe(resol_time)

        if debug:
            print now(), ": Report ", self.name, " got fixed after ", resol_time, " of reporting."


def run_model(team_capacity, report_number, reporters_config, resolution_time_gen, priority_gen, max_time):
    """
    Triggers the simulation, according to the provided parameters.
    :param team_capacity: Number of bug resolvers.
    :param report_number: Total number of defects.
    :param reporters_config: Configuration of the bug reporters.
    :param resolution_time_gen: Variate generator for resolution time.
    :param priority_gen: Variate generator for priorities.
    :param max_time: Simulation time.
    :return: Monitor for each bug reporter.
    """
    start_time = 0.0

    # The Resource is non-preemptable. It won't interrupt ongoing fixes.
    developer_resource = Resource(capacity=team_capacity, name="dev_team", unitName="developer", qType=PriorityQ,
                                  preemptable=False, monitored=True)
    bug_level = Level(capacity=sys.maxint, initialBuffered=report_number, monitored=True)

    initialize()
    testing_context = TestingContext(resolution_time_gen=resolution_time_gen, priority_gen=priority_gen,
                                     bug_level=bug_level)

    monitors = {}
    for reporter_config in reporters_config:
        resol_time_monitor = Monitor()
        bug_reporter = BugReportSource(reporter_config=reporter_config,
                                       testing_context=testing_context)
        activate(bug_reporter,
                 bug_reporter.start_reporting(developer_resource=developer_resource,
                                              resol_time_monitor=resol_time_monitor), at=start_time)
        monitors[reporter_config['name']] = resol_time_monitor

    simulate(until=max_time)
    return monitors


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
