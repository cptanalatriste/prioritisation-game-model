"""
This module is a discrete event simulation model for the bug reporting process
"""

from SimPy.Simulation import *
from SimPy.SimPlot import *

import numpy as np

import simdata
import simutils

CRITICAL_PRIORITY = 100

NOT_INFLATE_STRATEGY = 'NOT_INFLATE'
INFLATE_STRATEGY = 'INFLATE'
SIMPLE_INFLATE_STRATEGY = 'SIMPLE_INFLATE'

INITIAL_REPUTATION = 10


class TestingContext:
    """
    This class will produce the characteristics of the discovered defects.
    """

    def __init__(self, resolution_time_gen, priority_gen, bug_level, default_review_time, throttling, devtime_level,
                 quota_system):
        self.resolution_time_gen = resolution_time_gen
        self.priority_gen = priority_gen
        self.bug_level = bug_level
        self.devtime_level = devtime_level
        self.default_review_time = default_review_time
        self.throttling = throttling
        self.last_report_time = None
        self.quota_system = quota_system

        self.priority_monitors = {simdata.NON_SEVERE_PRIORITY: {'completed': Monitor(),
                                                                'reported': 0},
                                  simdata.NORMAL_PRIORITY: {'completed': Monitor(),
                                                            'reported': 0},
                                  simdata.SEVERE_PRIORITY: {'completed': Monitor(),
                                                            'reported': 0}}
        self.med_resolution_time = self.get_med_resolution_time()

    def stop_simulation(self, current_time):
        """
        Returns True if the simulation must be stopped.
        :param current_time: Current simulation time.
        :return: True if we should stop. False otherwise.
        """

        if self.last_report_time:
            elapsed_time = (current_time - self.last_report_time)

        if self.last_report_time is not None and elapsed_time >= self.med_resolution_time:
            return True

        return False

    def get_fix_effort(self, report_priority):
        """
        Return the time required for a bug to be fixed.
        :return: Effort required to fix a bug.
        """

        generator = self.resolution_time_gen[int(report_priority)]

        fix_effort = 0.0
        if generator is not None:
            fix_effort = generator.generate().item()

        return fix_effort

    def get_priority(self):
        """
        Returns the priority of the produced bug report.
        :return: The priority of the generate bug.
        """
        return self.priority_gen.generate()[0]

    def get_review_time(self):
        """
        Returns the review time required by the gatekeeper to correct the priority.
        :return:
        """
        return self.default_review_time

    def get_med_resolution_time(self):
        """
        Returns the median of all the resolution times, of all priorities.
        :return: Median resolution time.
        """
        all_resolution_times = []

        for priority in [simdata.NON_SEVERE_PRIORITY, simdata.NORMAL_PRIORITY, simdata.SEVERE_PRIORITY]:
            generator = self.resolution_time_gen[priority]
            if generator is not None:
                all_resolution_times.extend(generator.observations)

        return np.median(all_resolution_times)


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def __init__(self, reporter_config=None, testing_context=None):
        Process.__init__(self, reporter_config['name'])
        self.interarrival_time_gen = reporter_config['interarrival_time_gen']
        self.batch_size_gen = reporter_config['batch_size_gen']
        self.testing_context = testing_context
        self.reporter_reputation = INITIAL_REPUTATION

        self.inflation_gen = None
        strategy_key = 'strategy'
        if strategy_key in reporter_config:
            self.strategy = reporter_config[strategy_key]
            if self.strategy == INFLATE_STRATEGY:
                self.inflation_gen = simutils.DiscreteEmpiricalDistribution(values=[True, False],
                                                                            probabilities=[0.5, 0.5])

        self.priority_counters = {simdata.NON_SEVERE_PRIORITY: 0,
                                  simdata.NORMAL_PRIORITY: 0,
                                  simdata.SEVERE_PRIORITY: 0}
        self.report_counters = {simdata.NON_SEVERE_PRIORITY: 0,
                                simdata.NORMAL_PRIORITY: 0,
                                simdata.SEVERE_PRIORITY: 0}

    def start_reporting(self, developer_resource, gatekeeper_resource, reporter_monitor):
        """
        Activates a number of bug reports according to an inter-arrival time.
        :param resol_time_monitor: Monitor for the resolution time.
        :param developer_resource: The Development Team resource.
        :return: None
        """
        interarrival_time = self.get_interarrival_time()
        yield hold, self, interarrival_time

        bug_level = self.testing_context.bug_level

        batch_size = self.get_batch_size()
        while bug_level.amount >= batch_size:
            yield get, self, bug_level, batch_size

            if self.testing_context.last_report_time is None and bug_level.amount <= 0:
                self.testing_context.last_report_time = now()

            for index in range(batch_size):
                report_key = self.name + "-batch-" + str(bug_level.amount) + "-bug" + str(index)
                real_priority, report_priority = self.get_report_priority()
                fix_effort = self.testing_context.get_fix_effort(real_priority)
                review_time = self.testing_context.get_review_time()
                throttling = self.testing_context.throttling

                bug_report = BugReport(name=report_key, reporter=self,
                                       fix_effort=fix_effort,
                                       report_priority=report_priority,
                                       real_priority=real_priority,
                                       review_time=review_time,
                                       throttling=throttling)

                reported_priority_monitor = self.testing_context.priority_monitors[report_priority]['completed']

                devtime_level = self.testing_context.devtime_level
                activate(bug_report,
                         bug_report.arrive(developer_resource=developer_resource,
                                           gatekeeper_resource=gatekeeper_resource,
                                           resolution_monitors=[reporter_monitor, reported_priority_monitor],
                                           devtime_level=devtime_level, quota_system=self.testing_context.quota_system))

                self.priority_counters[real_priority] += 1

                self.report_counters[report_priority] += 1
                self.testing_context.priority_monitors[report_priority]['reported'] += 1

            interarrival_time = self.get_interarrival_time()
            yield hold, self, interarrival_time

            batch_size = self.get_batch_size()

        self.testing_context.last_report_time = now()

    def get_report_priority(self):
        real_priority = self.testing_context.get_priority()
        priority_for_report = real_priority

        if real_priority < simdata.SEVERE_PRIORITY:
            if self.inflation_gen is not None and self.inflation_gen.generate():
                priority_for_report += 1

            if self.strategy == SIMPLE_INFLATE_STRATEGY:
                priority_for_report += 2

        return real_priority, priority_for_report

    def get_interarrival_time(self):
        """
        Returns the time to wait after producing another bug report.
        :return: Time to hold before next report.
        """

        # This was put in place to avoid negative inter-arrival times.
        interarrival_time = -1
        while interarrival_time < 0:
            interarrival_time = self.interarrival_time_gen.generate()
        return interarrival_time

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

    def __init__(self, name, reporter, fix_effort, report_priority, real_priority, review_time, throttling):
        Process.__init__(self, name)
        self.reporter = reporter
        self.fix_effort = fix_effort
        self.report_priority = report_priority
        self.real_priority = real_priority
        self.review_time = review_time
        self.throttling = throttling

    def arrive(self, developer_resource, gatekeeper_resource, resolution_monitors, devtime_level, quota_system,
               debug=False):
        """
        The Process Execution Method for the Bug Reported process.
        :return:
        """
        arrival_time = now()

        inflation = False
        if self.report_priority != self.real_priority:
            inflation = True

        if debug:
            pending_bugs = len(developer_resource.waitQ)
            print arrival_time, ": Report ", self.name, " arrived. Effort: ", self.fix_effort, " Reported Priority: ", \
                self.report_priority, "Pending bugs: ", pending_bugs

        if gatekeeper_resource:
            # print "Requesting gatekeeper ", now(), "priority ", self.reporter.reporter_reputation

            yield request, self, gatekeeper_resource, self.reporter.reporter_reputation
            yield hold, self, self.review_time

            # print "Gatekeeper finished review", now()
            if inflation and self.throttling:
                self.reporter.reporter_reputation -= 1
                # print "Inflation detected! Applying throttling ", self.reporter.reporter_reputation

            self.report_priority = self.real_priority
            yield release, self, gatekeeper_resource
            # print "Gatekeeper released!", now()

        # print "Requesting developer resource: ", self.report_priority
        yield request, self, developer_resource, int(self.report_priority)

        if debug:
            print now(), ": Report ", self.name, " ready for fixing. "

        if quota_system:
            # TODO(cgavidia): Improve penalty calculation.
            penalty = 4000
            quota_manager = devtime_level

            donation = float(penalty) / (len(quota_manager.keys()) - 1)
            devtime_level = quota_manager[self.reporter.name]

            if inflation:
                # print "Inflation detected!"

                yield get, self, devtime_level, penalty
                # print " penalty ", penalty, " appplied to ", self.reporter.name

                for reporter, quota in quota_manager.iteritems():
                    if reporter != self.reporter.name:
                        yield put, self, quota, donation
                        # print " donation ", donation, " appplied to ", reporter

        if devtime_level.amount <= 0:
            if debug:
                print "No more developer time available."

            yield release, self, developer_resource
        elif self.fix_effort <= devtime_level.amount:
            yield get, self, devtime_level, self.fix_effort
            yield hold, self, self.fix_effort
            yield release, self, developer_resource

            resol_time = now() - arrival_time

            for monitor in resolution_monitors:
                monitor.observe(resol_time)

            if debug:
                print now(), ": Report ", self.name, " got fixed after ", resol_time, " of reporting. Fix effort: ", \
                    self.fix_effort, " Available Dev Time: ", devtime_level.amount
        else:
            # yield get, self, devtime_level, devtime_level.amount
            # yield hold, self, devtime_level.amount
            yield release, self, developer_resource


class SimulationController(Process):
    """
    Controls the execution of the simulation.
    """

    def __init__(self, name="", testing_context=None):
        Process.__init__(self, name)
        self.testing_context = testing_context

    def control(self):
        while True:
            yield hold, self, 1

            if self.testing_context.stop_simulation(now()):
                # print "Simulation ended at ", now(), ". Time after last report: ", self.testing_context.med_resolution_time
                stopSimulation()


def run_model(team_capacity, report_number, reporters_config, resolution_time_gen, priority_gen, max_time,
              dev_team_bandwith,
              gatekeeper_config=False,
              quota_system=False):
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

    gatekeeper_resource = None
    default_review_time = None
    throttling = False
    if gatekeeper_config:
        default_review_time = gatekeeper_config['review_time']
        throttling = gatekeeper_config['throttling']
        gatekeeper_resource = Resource(capacity=gatekeeper_config['capacity'], name="gatekeeper_team",
                                       unitName="gatekeeper", qType=PriorityQ,
                                       preemptable=False)

    # The Resource is non-preemptable. It won't interrupt ongoing fixes.
    preemptable = False

    developer_resource = Resource(capacity=team_capacity, name="dev_team", unitName="developer", qType=PriorityQ,
                                  preemptable=preemptable)

    bug_level = Level(capacity=sys.maxint, initialBuffered=report_number, monitored=True)
    devtime_level = Level(capacity=sys.maxint, initialBuffered=dev_team_bandwith, monitored=True)

    if quota_system:

        quota_per_dev = dev_team_bandwith / len(reporters_config)
        devtime_level = {config['name']: Level(capacity=sys.maxint, initialBuffered=quota_per_dev, monitored=True) for
                         config in reporters_config}

    initialize()
    testing_context = TestingContext(resolution_time_gen=resolution_time_gen, priority_gen=priority_gen,
                                     bug_level=bug_level, default_review_time=default_review_time,
                                     throttling=throttling, devtime_level=devtime_level, quota_system=quota_system)

    controller = SimulationController(testing_context=testing_context)
    activate(controller, controller.control())

    reporter_monitors = {}
    for reporter_config in reporters_config:
        reporter_monitor = Monitor()
        bug_reporter = BugReportSource(reporter_config=reporter_config,
                                       testing_context=testing_context)

        activate(bug_reporter,
                 bug_reporter.start_reporting(developer_resource=developer_resource,
                                              gatekeeper_resource=gatekeeper_resource,
                                              reporter_monitor=reporter_monitor), at=start_time)
        reporter_monitors[reporter_config['name']] = {"resolved_monitor": reporter_monitor,
                                                      "priority_counters": bug_reporter.priority_counters,
                                                      "report_counters": bug_reporter.report_counters}

    simulation_result = simulate(until=max_time)
    # print "simulation_result ", simulation_result
    return reporter_monitors, testing_context.priority_monitors


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
