"""
This module is a discrete event simulation model for the bug reporting process
"""
from collections import defaultdict

from SimPy.Simulation import *
from SimPy.SimPlot import *

import numpy as np

import simdata
import simutils

SIMPLE_INFLATE_STRATEGY = 'SIMPLEINFLATE'
HONEST_STRATEGY = 'HONEST'


class EmpiricalInflationStrategy:
    """
    Empirical Strategy for a two-level priority hierarchy
    """

    def __init__(self, strategy_config):
        self.name = strategy_config['name']
        self.strategy_config = strategy_config

    def priority_to_report(self, original_priority):
        result = original_priority

        correct = np.random.uniform()
        if original_priority == simdata.NON_SEVERE_PRIORITY and correct <= self.strategy_config[
            simutils.NONSEVERE_CORRECTION_COLUMN]:
            inflate = np.random.uniform()
            if inflate <= self.strategy_config[simutils.NON_SEVERE_INFLATED_COLUMN]:
                result = simdata.SEVERE_PRIORITY

        elif original_priority == simdata.SEVERE_PRIORITY and correct <= self.strategy_config[
            simutils.SEVERE_CORRECTION_COLUMN]:
            deflate = np.random.uniform()
            if deflate <= self.strategy_config[simutils.SEVERE_DEFLATED_COLUMN]:
                result = simdata.NON_SEVERE_PRIORITY

        return result

    def __str__(self):
        return self.name


class TestingContext:
    """
    This class will produce the characteristics of the discovered defects.
    """

    def __init__(self, resolution_time_gen, ignored_gen, reporter_gen, default_review_time, priority_generator,
                 quota_system,
                 quota_per_dev,
                 review_time_gen,
                 target_fixes,
                 views_to_discard,
                 inflation_factor=1):
        """
        Configures the context of the simulation.

        :param resolution_time_gen: Generators for resolution time, depending on their priority.
        :param bugs_by_priority: Number of defects present on the system, per priority.
        :param default_review_time: Time the gatekeeper uses for assesing the bug true priority.
        :param devtime_level: Level containing the number of developer time hours available for bug fixing.
        :param quota_system: If true, the number of developer hours available will be distributed among testers, penalizing inflators.
        """

        self.resolution_time_gen = resolution_time_gen
        self.ignored_gen = ignored_gen
        self.reporter_gen = reporter_gen
        self.priority_generator = priority_generator
        self.default_review_time = default_review_time
        self.review_time_gen = review_time_gen
        self.first_report_time = None
        self.last_report_time = None
        self.target_fixes = target_fixes
        self.quota_per_dev = quota_per_dev
        self.quota_system = quota_system
        self.views_to_discard = views_to_discard

        self.priority_monitors = {simdata.NON_SEVERE_PRIORITY: {'completed': Monitor(),
                                                                'reported': 0},
                                  simdata.NORMAL_PRIORITY: {'completed': Monitor(),
                                                            'reported': 0},
                                  simdata.SEVERE_PRIORITY: {'completed': Monitor(),
                                                            'reported': 0}}

        self.bug_counter = 0

        if self.quota_per_dev and self.quota_system:
            self.inflation_penalty = self.get_default_inflation_penalty() * inflation_factor

    def catch_bug(self):
        """
        Removes an item from the bug catalog.
        :return: The removed item.
        """

        priority = self.priority_generator.generate()[0]
        reporter = self.reporter_gen.generate()
        index = self.bug_counter

        bug_config = {'report_key': 'Priotity_' + str(priority) + "_Index_" + str(index),
                      'real_priority': priority,
                      'fix_effort': self.get_fix_effort(priority),
                      'reporter': reporter}

        self.bug_counter += 1
        return bug_config

    def get_reporting_time(self):
        """
        Returns the time spent on reporting activities,
        :return: Reporting time in hours.
        """

        if self.first_report_time is not None and self.last_report_time is not None:
            return self.last_report_time - self.first_report_time

        return 0

    def stop_simulation(self):
        """
        Returns True if the simulation must be stopped.
        :param current_time: Current simulation time.
        :return: True if we should stop. False otherwise.
        """

        total_fixed = sum([monitors['completed'].count() for priority, monitors in self.priority_monitors.iteritems()])

        # print "total_fixed: ", total_fixed

        if total_fixed < self.target_fixes:
            return False

        return True

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

    def ignore(self, reported_priority):
        """
        Returns if the reported removed from the queue should be fixed.
        :param reported_priority: Priority in the report.
        :return: True if it should be ignored, false otherwise.
        """
        generator = self.ignored_gen[int(reported_priority)]
        generator_output = generator.generate()
        return generator_output

    def discard(self, report):
        """
        Returns True if an ignored report should not be returned to the reports queue.
        :param report:  Report to evaluate.
        :return: True to discard, false otherwise.
        """
        return report.views_counter >= self.views_to_discard

    def get_review_time(self):
        """
        Returns the review time required by the gatekeeper to correct the priority.
        :return:
        """

        review_time = self.default_review_time

        if self.review_time_gen is not None:
            review_time = self.review_time_gen.generate().item()

        return review_time

    def get_default_inflation_penalty(self):
        """
        Returns the discount from a tester's quota if inflation is detected.
        :return: Inflation penalty.
        """
        return self.quota_per_dev


class BugReportSource(Process):
    """
    Represents a Tester, who generates Bug Reports.
    """

    def __init__(self, interarrival_time_gen=None, batch_size_gen=None, reporters_config=None, testing_context=None):
        Process.__init__(self, "Reporting_Process")
        self.interarrival_time_gen = interarrival_time_gen
        self.batch_size_gen = batch_size_gen
        self.reporters_config = reporters_config
        self.testing_context = testing_context

    def get_reporter_strategy(self, reporter_name):
        """
        Gets the strategy associated to a specific report
        :param reporter_name: Reporter name
        :return: Associated strategy.
        """
        strategy_key = 'strategy'

        for config in self.reporters_config:

            if config['name'] == reporter_name and strategy_key in config:
                return config[strategy_key]

        return None

    def start_reporting(self, developer_resource, gatekeeper_resource, reporter_monitors):
        """
        Activates a number of bug reports according to an inter-arrival time.
        :param gatekeeper_resource: The Gatekeeping Team Resource.
        :param reporter_monitor: Monitor for reporters.
        :param developer_resource: The Development Team resource.
        :return: None
        """

        interarrival_time = self.get_interarrival_time()
        yield hold, self, interarrival_time

        while True:
            batch_size = self.get_batch_size()
            arrival_time = now()
            self.testing_context.last_report_time = arrival_time

            for index in range(batch_size):
                bug_info = self.testing_context.catch_bug()
                report_key = bug_info['report_key']
                real_priority = bug_info['real_priority']
                fix_effort = bug_info['fix_effort']
                reporter = bug_info['reporter']

                report_priority = self.get_report_priority(reporter_name=reporter, real_priority=real_priority)
                review_time = self.testing_context.get_review_time()

                bug_report = BugReport(name=report_key, reporter=reporter,
                                       fix_effort=fix_effort,
                                       report_priority=report_priority,
                                       real_priority=real_priority,
                                       review_time=review_time,
                                       arrival_time=arrival_time)

                reported_priority_monitor = self.testing_context.priority_monitors[report_priority]['completed']

                reporter_metrics = reporter_monitors[reporter]
                reporter_monitor = reporter_metrics['resolved_monitor']
                resolved_counters = reporter_metrics['resolved_counters']

                inflation_penalty = self.get_inflation_penalty()

                if self.testing_context.first_report_time is None:
                    self.testing_context.first_report_time = arrival_time

                activate(bug_report,
                         bug_report.arrive(developer_resource=developer_resource,
                                           gatekeeper_resource=gatekeeper_resource,
                                           resolution_monitors=[reporter_monitor, reported_priority_monitor,
                                                                resolved_counters],
                                           testing_context=self.testing_context,
                                           inflation_penalty=inflation_penalty))

                reporter_metrics['priority_counters'][real_priority] += 1
                reporter_metrics['report_counters'][report_priority] += 1
                self.testing_context.priority_monitors[report_priority]['reported'] += 1

            interarrival_time = self.get_interarrival_time()
            yield hold, self, abs(interarrival_time)

        self.testing_context.last_report_time = arrival_time

    def get_report_priority(self, reporter_name, real_priority):
        """
        Returns a priority to include in the report, according to the tester strategy.
        :param real_priority: Ground-truth priority of the bug.
        :return: Priority to report.
        """

        reporter_strategy = self.get_reporter_strategy(reporter_name)

        if reporter_strategy is None:
            return real_priority

        if reporter_strategy is not None and isinstance(reporter_strategy, EmpiricalInflationStrategy):
            return reporter_strategy.priority_to_report(real_priority)

        priority_for_report = real_priority
        return priority_for_report

    def get_inflation_penalty(self):
        """
        Returns the amount of the developer quota that will be taken in case of inflation.
        :param devtime_level: List of tester's quotas.
        :return: Penalty for inflation, None if it is not the case.
        """
        inflation_penalty = None

        # TODO(cgavidia): Temporary comment. Will fix later.
        # if self.testing_context.quota_system:
        #     default_inflation_penalty = self.testing_context.inflation_penalty
        #
        #     inflation_penalty = 0
        #
        #     developer_quota = devtime_level[self.name]
        #
        #     if developer_quota.amount >= default_inflation_penalty:
        #         inflation_penalty = default_inflation_penalty
        #     elif 0 <= developer_quota.amount < default_inflation_penalty:
        #         inflation_penalty = developer_quota.amount

        return inflation_penalty

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

    def __init__(self, name, reporter, fix_effort, report_priority, real_priority, review_time, arrival_time):
        Process.__init__(self, name)
        self.reporter = reporter
        self.fix_effort = fix_effort
        self.report_priority = report_priority
        self.real_priority = real_priority
        self.review_time = review_time
        self.arrival_time = arrival_time
        self.views_counter = 0

    def get_priority_for_queue(self):
        return int(self.arrival_time)

    def arrive(self, developer_resource, gatekeeper_resource, resolution_monitors, testing_context, inflation_penalty,
               debug=False):
        """
        The Process Execution Method for the Bug Reported process.

        :param developer_resource: Resource representing the development team.
        :param gatekeeper_resource: Resource representing the Bug Gatekeeper.
        :param resolution_monitors: List of monitors for resolved bugs.
        :param devtime_level: Level containing the developer time hours available.
        :param inflation_penalty: The number of dev times hours an inflator gets penalized, in case there's a quota system in place.
        :param debug: True to have detailed log messaged.
        :return: None.
        """

        arrival_time = now()

        false_report = False
        if self.report_priority != self.real_priority:
            false_report = True

        # This section relates to the gatekeeper logic.
        if gatekeeper_resource:
            yield request, self, gatekeeper_resource
            yield hold, self, abs(self.review_time)

            self.report_priority = self.real_priority
            yield release, self, gatekeeper_resource

        yield request, self, developer_resource, self.get_priority_for_queue()

        # This sections applies the quota system process. The Bug Report is holding a Developer resource.
        if inflation_penalty is not None:
            # TODO(cgavidia): Temporary value. Replace later.
            quota_manager = None

            donation = float(inflation_penalty) / (len(quota_manager.keys()) - 1)
            devtime_level = quota_manager[self.reporter]

            if false_report:

                if inflation_penalty > 0:
                    if debug:
                        print "Penalty to be applied to", self.reporter, " : Removing ", inflation_penalty, \
                            " from existing quota of ", devtime_level.amount

                    yield get, self, devtime_level, abs(inflation_penalty)

                    for reporter, quota in quota_manager.iteritems():
                        if reporter != self.reporter:
                            if debug:
                                print "Adding ", donation, " to reporter ", reporter, " quota. Previous value: ", quota.amount
                            yield put, self, quota, donation

                # Priority gets corrected and the report is re-entered on the queue
                self.report_priority = self.real_priority
                yield release, self, developer_resource
                yield request, self, developer_resource, self.get_priority_for_queue()

        if testing_context.ignore(self.report_priority):
            self.views_counter += 1
            yield release, self, developer_resource

            if not testing_context.discard(self):
                yield request, self, developer_resource, self.get_priority_for_queue()
            else:
                return

        # Finally, here we're using our development time budget.
        yield hold, self, abs(self.fix_effort)
        yield release, self, developer_resource

        resol_time = now() - arrival_time

        for monitor in resolution_monitors:
            if isinstance(monitor, dict):
                monitor[self.real_priority] += 1
            else:
                monitor.observe(resol_time)

            if debug:
                print now(), ": Report ", self.name, "by reporter ", self.reporter.name, " got fixed after ", resol_time, \
                    " of reporting. Fix effort: ", self.fix_effort, " Available Dev Time: ", devtime_level.amount


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

            if self.testing_context.stop_simulation():
                stopSimulation()


def start_priority_counter():
    """
    Returns a priority-based counter.
    :return: A map, with an entry per priority.
    """
    return {simdata.NON_SEVERE_PRIORITY: 0,
            simdata.NORMAL_PRIORITY: 0,
            simdata.SEVERE_PRIORITY: 0}


def run_model(team_capacity, reporters_config, resolution_time_gen, ignored_gen, reporter_gen, max_time,
              priority_generator=None,
              target_fixes=None,
              dev_size_generator=None,
              gatekeeper_config=False,
              interarrival_time_gen=None,
              batch_size_gen=None,
              views_to_discard=0,
              quota_system=False, inflation_factor=1, debug=False):
    """
    Triggers the simulation, according to the provided parameters.
    :param debug: Enable to got debug information.
    :param gatekeeper_config: Configuration parameters for the Gatekeeper.
    :param quota_system: If true, the developer time is divided among all the bug reporters in the simulation.
                            If inflation happens, the offender gets penalized.
    :param dev_team_bandwith: Number of hours available for bug fixing tasks.
    :param team_capacity: Number of bug resolvers.
    :param bugs_by_priority: Total number of defects per priority.
    :param reporters_config: Configuration of the bug reporters.
    :param resolution_time_gen: Variate generator for resolution time.
    :param max_time: Simulation time.
    :return: Monitor for each bug reporter.
    """
    start_time = 0.0

    gatekeeper_resource = None
    default_review_time = None
    if gatekeeper_config:
        default_review_time = gatekeeper_config['review_time']
        gatekeeper_resource = Resource(capacity=gatekeeper_config['capacity'], name="gatekeeper_team",
                                       unitName="gatekeeper", qType=PriorityQ,
                                       preemptable=False)

    # The Resource is non-preemptable. It won't interrupt ongoing fixes.
    preemptable = False

    if dev_size_generator is not None:
        # We are ensuring a minimum capacity of one developer.
        team_capacity = max(1, dev_size_generator.generate()[0])

    # print "team_capacity: ", team_capacity
    developer_resource = Resource(capacity=team_capacity, name="dev_team", unitName="developer", qType=PriorityQ,
                                  preemptable=preemptable)
    quota_per_dev = None
    if quota_system:
        # TODO(cgavidia): Temporary value
        dev_team_bandwith = 0
        quota_per_dev = dev_team_bandwith / len(reporters_config)
        devtime_level = {config['name']: Level(capacity=sys.maxint, initialBuffered=quota_per_dev, monitored=True) for
                         config in reporters_config}

    initialize()

    # TODO(cgavidia): Temporary value
    review_time_gen = None
    testing_context = TestingContext(resolution_time_gen=resolution_time_gen,
                                     ignored_gen=ignored_gen,
                                     reporter_gen=reporter_gen,
                                     priority_generator=priority_generator, default_review_time=default_review_time,
                                     quota_system=quota_system,
                                     quota_per_dev=quota_per_dev,
                                     inflation_factor=inflation_factor,
                                     review_time_gen=review_time_gen,
                                     views_to_discard=views_to_discard,
                                     target_fixes=target_fixes)

    controller = SimulationController(testing_context=testing_context)
    activate(controller, controller.control())

    reporter_monitors = {}
    for reporter_config in reporters_config:
        reporter_monitor = Monitor()

        reporter_monitors[reporter_config['name']] = {"resolved_monitor": reporter_monitor,
                                                      "priority_counters": start_priority_counter(),
                                                      "report_counters": start_priority_counter(),
                                                      "resolved_counters": start_priority_counter()}

    bug_reporter = BugReportSource(reporters_config=reporters_config,
                                   testing_context=testing_context,
                                   interarrival_time_gen=interarrival_time_gen,
                                   batch_size_gen=batch_size_gen)

    activate(bug_reporter,
             bug_reporter.start_reporting(developer_resource=developer_resource,
                                          gatekeeper_resource=gatekeeper_resource,
                                          reporter_monitors=reporter_monitors), at=start_time)

    if debug:
        print "testing_context.inflation_penalty ", testing_context.inflation_penalty, "len(reporters_config) ", len(
            reporters_config)

    simulation_result = simulate(until=max_time)
    return reporter_monitors, testing_context.priority_monitors, testing_context.get_reporting_time()


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
