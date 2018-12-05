"""
Utility types for supporting the simulation.
"""
from collections import defaultdict

import sys

import math

import time

import logging
from pathos.multiprocessing import ProcessingPool as Pool

import scipy.interpolate as interpolate
import numpy as np

import pandas as pd

from scipy.stats import uniform
from scipy.stats import rv_discrete

from sklearn.cluster import KMeans

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import progressbar
import simdata
import simmodel
import gtconfig

import matplotlib

if not gtconfig.is_windows:
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

MINIMUM_OBSERVATIONS = 3
EPSILON = 0.001

REPORTER_COLUMNS = [simmodel.NON_SEVERE_INFLATED_COLUMN, simmodel.SEVERE_DEFLATED_COLUMN]

logger = gtconfig.get_logger("simulation_utils", "simulation_utils.txt", level=logging.INFO)


class SimulationConfig:
    """
    Contains the parameters necessary for the simulation execution.
    """

    def __init__(self, team_capacity, reporters_config, resolution_time_gen, batch_size_gen, interarrival_time_gen,
                 max_time=sys.maxint, quota_system=False, inflation_factor=None, priority_generator=None,
                 ignored_gen=None, reporter_gen=None, target_fixes=None, dev_time_budget=None,
                 dev_size_generator=None, gatekeeper_config=None, catcher_generator=None, bug_stream=None,
                 replication_id=None, priority_queue=False, views_to_discard=0):
        self.team_capacity = team_capacity
        self.reporters_config = reporters_config
        self.resolution_time_gen = resolution_time_gen
        self.batch_size_gen = batch_size_gen
        self.interarrival_time_gen = interarrival_time_gen
        self.max_time = max_time
        self.ignored_gen = ignored_gen
        self.reporter_gen = reporter_gen
        self.priority_generator = priority_generator
        self.catcher_generator = catcher_generator
        self.target_fixes = target_fixes
        self.dev_time_budget = dev_time_budget
        self.dev_size_generator = dev_size_generator
        self.gatekeeper_config = gatekeeper_config
        self.quota_system = quota_system
        self.inflation_factor = inflation_factor
        self.bug_stream = bug_stream

        self.replication_id = replication_id
        self.priority_queue = priority_queue
        self.views_to_discard = views_to_discard

    def __str__(self):
        gatekeeper_params = "NONE"
        if self.gatekeeper_config is not None:
            gatekeeper_params = " Capacity " + str(self.gatekeeper_config['capacity']) + " Time Generator " + str(
                self.gatekeeper_config['review_time_gen'])

        return "Target fixes: " + str(
            self.target_fixes) + " Dev Time Budget " + str(self.dev_time_budget) + " .Throttling enabled: " + str(
            self.quota_system) + " . Inflation penalty: " + str(self.inflation_factor) + " Developers in team: " + str(
            self.team_capacity) + " Success probabilities: " + str(
            self.catcher_generator) + " Gatekeeper Config: " + (
                   gatekeeper_params) + " Max Reporter Probability: " + str(max(
            self.reporter_gen.probabilities)) + " Min Reporter Probability " + str(min(
            self.reporter_gen.probabilities)) + " Priority distribution: " + str(
            self.priority_generator) + " Severe Ignore Probabilities: " + str(self.ignored_gen[
                                                                                  simdata.SEVERE_PRIORITY].probabilities) + " Non-Severe Ignore Probabilities: " + str(
            self.ignored_gen[
                simdata.NON_SEVERE_PRIORITY].probabilities) + " Priority Queue: " + str(self.priority_queue)


class MixedEmpiricalInflationStrategy:
    """
    This class represents a mixed empirical strategy: It randomly selects an instance of EmpiricalInflationStrategy
    according to probability distribution.
    """

    def __init__(self, mixed_strategy_config):
        if len(mixed_strategy_config['strategy_configs']) != len(mixed_strategy_config['probabilities']):
            raise Exception(
                "The number of configurations and probabilities does not match for strategy " + mixed_strategy_config[
                    'name'])

        delta = abs(sum(mixed_strategy_config['probabilities']) - 1.0)
        if delta > EPSILON:
            raise Exception("The probabilities in mixed strategy " + mixed_strategy_config[
                'name'] + " should sum 1. Probabilities: " + str(
                mixed_strategy_config['probabilities']) + ". Delta: " + str(delta))

        self.name = mixed_strategy_config['name']
        self.strategy_config = mixed_strategy_config

        self.strategy_selector = DiscreteEmpiricalDistribution(name="strategy_selector",
                                                               values=mixed_strategy_config['strategy_configs'],
                                                               probabilities=mixed_strategy_config['probabilities'])

        self.current_strategy = None

    def configure(self):
        self.current_strategy = EmpiricalInflationStrategy(strategy_config=self.strategy_selector.generate())

    def priority_to_report(self, original_priority):
        if self.current_strategy is None:
            self.configure()

        return self.current_strategy.priority_to_report(original_priority)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class EmpiricalInflationStrategy:
    """
    Empirical Strategy for a two-level priority hierarchy
    """

    def __init__(self, strategy_config):
        logger.debug("strategy_config: ", strategy_config)

        self.name = strategy_config['name']
        self.strategy_config = strategy_config

        self.inflation_prob = strategy_config[simmodel.NON_SEVERE_INFLATED_COLUMN]
        self.deflation_prob = strategy_config[simmodel.SEVERE_DEFLATED_COLUMN]

        self.inflation_generator = None
        self.deflation_generator = None

    def configure(self):
        self.inflation_generator = DiscreteEmpiricalDistribution(name="inflation_generator",
                                                                 values=[True, False],
                                                                 probabilities=[self.inflation_prob,
                                                                                (1 - self.inflation_prob)])

        self.deflation_generator = DiscreteEmpiricalDistribution(name="deflation_generator",
                                                                 values=[True, False],
                                                                 probabilities=[self.deflation_prob,
                                                                                (1 - self.deflation_prob)])

    def priority_to_report(self, original_priority):

        if self.inflation_generator is None or self.deflation_generator is None:
            self.configure()

        result = original_priority

        if original_priority == simdata.NON_SEVERE_PRIORITY:
            if self.inflation_generator.generate():
                result = simdata.SEVERE_PRIORITY
        elif original_priority == simdata.SEVERE_PRIORITY:
            if self.deflation_generator.generate():
                result = simdata.NON_SEVERE_PRIORITY

        return result

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class SimulationMetrics:
    """
    Consolidates the information generated by several simulation replications, in lists. Each list item corresponds to a
    simulation replication.
    """

    def __init__(self):
        self.completed_per_reporter = []
        # This is according to the priority contained in the report.
        self.reports_per_reporter = []
        self.completed_per_priority = []
        self.reported_per_priotity = []  # TODO(cgavidia): Correct this awful typo

        # This is according to the real priority of the bug.
        self.bugs_per_reporter = []
        self.resolved_per_reporter = []
        self.time_per_priority = []
        self.active_per_priority = []

        self.reporting_times = []

    def append_results(self, simulation_metrics):
        """
        Merges the current information with the metrics from other simulation runs.
        :param simulation_metrics:
        :return:
        """
        self.completed_per_reporter += simulation_metrics.completed_per_reporter
        self.completed_per_priority += simulation_metrics.completed_per_priority
        self.bugs_per_reporter += simulation_metrics.bugs_per_reporter
        self.reports_per_reporter += simulation_metrics.reports_per_reporter
        self.resolved_per_reporter += simulation_metrics.resolved_per_reporter
        self.reported_per_priotity += simulation_metrics.reported_per_priotity
        self.reporting_times += simulation_metrics.reporting_times
        self.time_per_priority += simulation_metrics.time_per_priority
        self.active_per_priority += simulation_metrics.active_per_priority

    def process_simulation_output(self, reporter_monitors, priority_monitors, reporting_time):
        """
        After simulation finishes, it collects the outputs.
        :param reporter_monitors:
        :param priority_monitors:
        :param reporting_time:
        :return:
        """
        result_per_reporter = {reporter_name: reporter_info['resolved_monitor'].count() for reporter_name, reporter_info
                               in
                               reporter_monitors.iteritems()}
        self.completed_per_reporter.append(result_per_reporter)

        result_per_priority = {priority: monitors[simmodel.METRIC_BUGS_FIXED].count() for priority, monitors in
                               priority_monitors.iteritems()}
        self.completed_per_priority.append(result_per_priority)

        reports_per_priority = {priority: monitors[simmodel.METRIC_BUGS_REPORTED] for priority, monitors in
                                priority_monitors.iteritems()}
        self.reported_per_priotity.append(reports_per_priority)

        time_per_priority = {priority: monitors[simmodel.METRIC_TIME_INVESTED] for priority, monitors in
                             priority_monitors.iteritems()}
        self.time_per_priority.append(time_per_priority)

        active_per_priority = {priority: monitors[simmodel.METRIC_BUGS_ACTIVE] for priority, monitors in
                               priority_monitors.iteritems()}
        self.active_per_priority.append(active_per_priority)

        self.bugs_per_reporter.append(gather_reporter_statistics(reporter_monitors, 'priority_counters'))
        self.reports_per_reporter.append(gather_reporter_statistics(reporter_monitors, 'report_counters'))
        self.resolved_per_reporter.append(gather_reporter_statistics(reporter_monitors, 'resolved_counters'))

        self.reporting_times.append(reporting_time)

    def get_consolidated_output(self, reporter_configuration):
        """
        Returns a summary view of the performance of reporters in all the simulations.
        :param reporter_configuration: List of reporters.
        :return: A list of dictionaries, with performance metrics.
        """

        if len(self.completed_per_reporter) != len(self.bugs_per_reporter):
            raise Exception("The output of the simulation doesn't match!")

        simulation_results = []

        for run in range(len(self.completed_per_reporter)):

            for reporter_config in reporter_configuration:
                reporter_name = reporter_config['name']

                reporter_team = None
                if 'team' in reporter_config.keys():
                    reporter_team = reporter_config['team']

                reporter_strategy = reporter_config[simmodel.STRATEGY_KEY].name

                consolidate_result = {"run": run,
                                      "reporter_name": reporter_name,
                                      "reporter_team": reporter_team,
                                      "reporter_strategy": reporter_strategy}

                reporter_performance = self.get_reporter_performance(run, reporter_name)

                consolidate_result.update(reporter_performance)
                simulation_results.append(consolidate_result)

        return simulation_results

    def get_reporter_performance(self, run, reporter_name):
        """
        Returns how the reporter did in an specific simulation run.
        :param run: Run identifier
        :param reporter_name: Reporter name.
        :return: A dict containing all the metrics.
        """

        run_resolved = self.completed_per_reporter[run]
        run_found = self.bugs_per_reporter[run]
        run_reported = self.reports_per_reporter[run]
        run_resolved_priority = self.resolved_per_reporter[run]

        reported_completed = run_resolved[reporter_name]

        severe_completed = run_resolved_priority[reporter_name][simdata.SEVERE_PRIORITY]
        non_severe_completed = run_resolved_priority[reporter_name][simdata.NON_SEVERE_PRIORITY]
        normal_completed = run_resolved_priority[reporter_name][simdata.NORMAL_PRIORITY]

        severe_found = run_found[reporter_name][simdata.SEVERE_PRIORITY]
        non_severe_found = run_found[reporter_name][simdata.NON_SEVERE_PRIORITY]
        normal_found = run_found[reporter_name][simdata.NORMAL_PRIORITY]

        severe_reported = run_reported[reporter_name][simdata.SEVERE_PRIORITY]
        non_severe_reported = run_reported[reporter_name][simdata.NON_SEVERE_PRIORITY]
        normal_reported = run_reported[reporter_name][simdata.NORMAL_PRIORITY]

        return {'reported_completed': reported_completed,
                'severe_completed': severe_completed,
                'non_severe_completed': non_severe_completed,
                'normal_completed': normal_completed,
                'severe_found': severe_found,
                'non_severe_found': non_severe_found,
                'normal_found': normal_found,
                'severe_reported': severe_reported,
                'non_severe_reported': non_severe_reported,
                'normal_reported': normal_reported,
                "reported": severe_reported + non_severe_reported + normal_reported}

    def get_completed_per_priority(self, priority):
        """
        Get the number of reports that were completed according to the REPORTED priority
        :param priority: Priority
        :return: List containing the replication values.
        """
        return [report[priority] for report in self.completed_per_priority]

    def get_reported_per_priority(self, priority):
        """
        Get the number of reports that were REPORTED (not real) according to a priority
        :param priority: Priority
        :return: List containing the replication values.
        """
        return [report[priority] for report in self.reported_per_priotity]

    def get_active_by_real_priority(self, priority):
        """
        Get the number of reports that were active, that means removed from the queue, during the simulation.
        :param priority: REAL priority of the bug
        :return: List containing replication values.
        """
        return [report[priority] for report in self.active_per_priority]

    def get_completed_per_real_priority(self, priority):
        """
        Get the number of reports that were completed according to the REAL priority
        :param priority: Priority
        :return: List containing the replication values.
        """

        return SimulationMetrics.consolidate_per_priority(priority, self.resolved_per_reporter)

    def get_reported_by_real_priority(self, priority):
        """
        Get the number of reports that were REPORTED according to the REAL priority
        :param priority: Priority
        :return: List containing the replication values.
        """

        return SimulationMetrics.consolidate_per_priority(priority, self.bugs_per_reporter)

    def get_time_per_priority(self, priority):
        """
        Get the time spent on bugs according to a priority
        :param priority: Priority
        :return: List containing the replication values.
        """
        return [report[priority] for report in self.time_per_priority]

    def get_time_ratio_per_priority(self, priority):
        """
        Get the time spent as a ratio on bugs according to a priority
        :param priority: Priority
        :return: List containing the replication values.
        """
        priority_times = self.get_time_per_priority(priority)
        total_times = [sum(report.values()) for report in self.time_per_priority]

        return [priority_time / float(total_time) if total_time > 0 else 0.0 for priority_time, total_time in
                zip(priority_times, total_times)]

    def get_fixed_ratio_per_priority(self, priority, exclude_open=False):
        """
        Get the fixes as a ratio on bugs according to a priority
        :param priority: Priority
        :return: List containing the replication values.
        """
        fixed = self.get_completed_per_real_priority(priority)

        if not exclude_open:
            total = self.get_reported_by_real_priority(priority)
        else:
            total = self.get_active_by_real_priority(priority)

        return [fixed_bugs / float(reported_bugs) if reported_bugs > 0 else 0.0 for fixed_bugs, reported_bugs in
                zip(fixed, total)]

    def get_completed_per_reporter(self, reporter_name):
        """
        Get the number of reports that were completed according to a reporter
        :param priority: Priority
        :return: List containing the replication values.
        """
        return [report[reporter_name] for report in self.completed_per_reporter]

    def get_total_resolved(self, reporters_config):
        """
        Gets the total number of fixes, according to the reporter information.
        :param reporters_config: List of reporters
        :return: List containing the replication values.
        """
        return SimulationMetrics.consolidate_reporter_metrics(reporters_config, self.completed_per_reporter)

    def get_total_reported(self, reporters_config):
        """
        Gets the total number of reports, according to the reporter information.
        :param reporters_config: List of reporters
        :return: List containing the replication values.
        """

        reports_list = [{reporter_name: sum(reports[reporter_name].values()) for
                         reporter_name in reports.keys()} for
                        reports in
                        self.reports_per_reporter]

        return SimulationMetrics.consolidate_reporter_metrics(reporters_config, reports_list)

    @staticmethod
    def consolidate_reporter_metrics(reporters_config, metrics):
        """
        Accumulates reporter information.
        :param reporters_config: Reporter catalog.
        :param metrics: Reporter metrics
        :return:
        """
        results = []
        for report in metrics:
            total = 0
            for reporter_config in reporters_config:
                total += report[reporter_config['name']]
            results.append(total)

        return results

    @staticmethod
    def consolidate_per_priority(priority, metrics):
        """
        Accumulates reporter information according to a priority.
        :param priority: Priority
        :return: List containing the replication values.
        """

        results = []
        for run_stats in metrics:
            total = 0

            for reporter, reporter_stats in run_stats.iteritems():
                total += reporter_stats[priority]

            results.append(total)

        return results


class ContinuousEmpiricalDistribution:
    def __init__(self, observations=None, distribution=None, parameters=None):
        self.distribution = None
        self.parameters = None

        if observations is not None and distribution is None:
            if len(observations) < MINIMUM_OBSERVATIONS:
                raise ValueError("Only " + str(len(observations)) + " samples were provided.")

            self.inverse_cdf = get_inverse_cdf(observations)

        if distribution is not None and parameters is not None:
            self.distribution = distribution
            self.parameters = parameters

    def generate_from_scipy(self):
        """
        A theoretical distribution fitted from the data. Generates samples from the fitted distributions.
        (From: Discrete Event Simulation by G. Fishman, Chapter 10)

        :return:
        """
        parameter_tuple = self.parameters

        if len(parameter_tuple) == 2:
            loc = parameter_tuple[0]
            scale = parameter_tuple[1]
            rand_variate = self.distribution.rvs(loc=loc, scale=scale, size=1)[0]

        elif len(parameter_tuple) == 3:
            shape = parameter_tuple[0]
            loc = parameter_tuple[1]
            scale = parameter_tuple[2]

            rand_variate = self.distribution.rvs(shape, loc=loc, scale=scale, size=1)[0]

        return rand_variate

    def generate(self, rand_uniform=None):
        """
        Samples from this empirical distribution using the Inverse Transform method.

        :return:Random variate
        """

        if rand_uniform is None:
            rand_uniform = uniform.rvs(size=1)[0]

        if self.distribution is not None and self.parameters is not None:
            return self.generate_from_scipy()

        rand_variate = self.inverse_cdf(rand_uniform)
        return rand_variate


class ConstantGenerator:
    """
    A Generator that produces the same value all the time.
    """

    def __init__(self, name="", value=None):
        self.name = name
        self.value = value

    def generate(self):
        return self.value

    def __str__(self):
        return str(self.value) + " (Constant)"


class DiscreteEmpiricalDistribution:
    def __init__(self, name="", observations=None, values=None, probabilities=None, inverse_cdf=False):
        self.inverse_cdf = None
        self.disc_distribution = None
        self.name = name

        if observations is not None and isinstance(observations, pd.Series):

            if inverse_cdf:
                self.inverse_cdf = get_inverse_cdf(observations)
            else:
                values_with_probabilities = observations.value_counts(normalize=True)
                values = np.array([index for index, _ in values_with_probabilities.iteritems()])
                probabilities = [probability for _, probability in values_with_probabilities.iteritems()]

        if values is not None and probabilities is not None:
            self.configure(values, probabilities)

    def configure(self, values, probabilities):
        """
        Configures the rv_discrete instance that will generate the variates.
        :param values: Values to produce.
        :param probabilities: Probability of each of these values.
        :return: None
        """
        self.values = values
        self.probabilities = probabilities

        self.disc_distribution = rv_discrete(values=(range(len(values)), self.probabilities))

    def generate(self):
        """
        Samples from the empirical distribution.
        :return: Random variate
        """

        if self.disc_distribution is not None:
            # This procedure was inspired by:
            # http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy

            variate_index = self.disc_distribution.rvs(size=1)[0]

            if isinstance(self.values[variate_index], np.ndarray):
                return self.values[variate_index][0]
            else:
                return self.values[variate_index]

        if self.inverse_cdf is not None:
            # This is the Quantile Method implementation for discrete variables, according to Discrete-Event Simulation
            # by G. Fishman (page 463)

            rand_uniform = uniform.rvs(size=1)[0]
            rand_variate = self.inverse_cdf(rand_uniform)
            return math.floor(rand_variate)

    def get_probabilities(self):
        """
        Returns a dictionary with the supported values with their corresponding probabilities.
        :return: Dictionary with probabilities.
        """
        probability_map = {value: self.probabilities[index] for index, value in enumerate(self.values)}
        return defaultdict(float, probability_map)

    def copy(self, name=""):
        """
        Generates a copy of the current generator.
        :param name: Name of the copy.
        :return: A generator copy.
        """
        return DiscreteEmpiricalDistribution(name=name, values=self.values, probabilities=self.probabilities)

    def __str__(self):
        return str(self.get_probabilities())


def get_inverse_cdf(observations, n_bins=40):
    """
    Inverse cumulative distribution function, required for inverse transformation sampling.

    Code inspired from:
    http://www.nehalemlabs.net/prototype/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/

    An Empirical Distribution Function based exclusively on the data, and generates samples from them
    during the simulation (From: Discrete Event Simulation by G. Fishman, Chapter 10)

    Potentially, we are generating an estimatot of the inverse distribution function, called the Quantile Method on
    Discrete Event Simulation by G. Fishman, Chapter 10. Our parameter n_bins would be called k in the book.

    :return: Inverse CDF instance
    """

    hist, bin_edges = np.histogram(observations, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)

    return inv_cdf


def remove_drive_in_testers(reporters_config, min_reports):
    """
    Removes drive-in testers, defined as the testers has a number of active days bigger than a threshold.

    :param reporters_config: Reporter configuration.
    :param min_reports: Minimum number of reports to be considered non drive-by.
    :return: Filtered list of reporters config.
    """
    # reporter_metrics = [np.mean(config['interarrival_time_gen'].observations) for config in reporters_config]
    reporter_metrics = [len(config['inter_arrival_sample']) for config in reporters_config]

    overall_mean = np.mean(reporter_metrics)
    overall_std = np.std(reporter_metrics)
    overall_max = np.max(reporter_metrics)
    overall_min = np.min(reporter_metrics)

    print "remove_drive_in_testers->min_reports ", min_reports, "overall_mean ", overall_mean, "overall_std ", overall_std, "overall_max", overall_max, "overall_min", overall_min

    engaged_testers = [config for config in reporters_config if
                       len(config['inter_arrival_sample']) >= min_reports]

    drive_by_testers = [config for config in reporters_config if
                        len(config['inter_arrival_sample']) < min_reports]

    return engaged_testers, drive_by_testers


def get_reporter_behavior_dataframe(reporters_config):
    """
    Returns a dataframe describing bug reporter behaviour.
    :param reporters_config: List of reporter configuration.
    :return: Dataframe with behaviour information.
    """

    reporter_records = []
    reporter_names = []
    for config in reporters_config:
        total_nonsevere = float(config['reports_per_priority'][simdata.NON_SEVERE_PRIORITY])
        total_severe = float(config['reports_per_priority'][simdata.SEVERE_PRIORITY])

        non_severe_false = config['modified_details']['priority_1_false']
        severe_false = config['modified_details']['priority_3_false']

        reporter_names.append(config['name'])
        reporter_records.append(
            [non_severe_false / total_nonsevere if total_nonsevere != 0 else 0,
             severe_false / total_severe if total_severe != 0 else 0])

    reporter_dataframe = pd.DataFrame(data=reporter_records, index=reporter_names)

    reporter_dataframe.columns = REPORTER_COLUMNS
    return reporter_dataframe


def elbow_method_for_reporters(reporter_configuration, file_prefix=""):
    """
    Elbow method implementation for estimating the optimal number of clusters for a given task.
    :param reporter_configuration: List of reporter configuration info.
    :return: Generates the elbow method plot as a file.
    """
    distortions = []

    reporters_with_corrections = [config for config in reporter_configuration if
                                  config['with_modified_priority'] > 0]
    reporter_dataframe = get_reporter_behavior_dataframe(reporters_with_corrections)
    print "Extracting strategies from ", len(reporter_dataframe.index), " reporters with third-party corrections"

    report_features = reporter_dataframe.values

    for clusters in range(1, 11):
        kmeans = KMeans(n_clusters=clusters,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)

        kmeans.fit(report_features)
        distortions.append(kmeans.inertia_)

    plt.clf()
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')

    file_name = "img/" + file_prefix + "_elbow_for_reporters.png"
    plt.savefig(file_name, bbox_inches='tight')
    print "Elbow-method plot saved at ", file_name


def assign_strategies(reporters_config, training_issues, n_clusters=3):
    """
    Assigns an inflation pattern to the reporter based on clustering.
    :param reporters_config: Reporter configuration.
    :param training_issues: Training dataset.
    :return: Reporting Configuration including inflation pattern.
    """

    global_priority_map = DiscreteEmpiricalDistribution(
        observations=training_issues[simdata.SIMPLE_PRIORITY_COLUMN]).get_probabilities()

    logger.debug("global_priority_map: " + str(global_priority_map))

    reporter_dataframe = get_reporter_behavior_dataframe(reporters_config)

    # TODO(cgavidia): Refactor this heavily!

    # Removing scaling because of cluster quality.
    # scaler = StandardScaler()
    # report_features = scaler.fit_transform(reporter_dataframe.values)
    # global_features = scaler.transform(
    #     [global_priority_map[simdata.NON_SEVERE_PRIORITY], global_priority_map[simdata.NORMAL_PRIORITY],
    #      global_priority_map[simdata.SEVERE_PRIORITY], 0.0])

    expected_corrections = 0.0
    global_features = [global_priority_map[simdata.NON_SEVERE_PRIORITY], global_priority_map[simdata.NORMAL_PRIORITY],
                       global_priority_map[simdata.SEVERE_PRIORITY], expected_corrections]
    report_features = reporter_dataframe.values

    print "Starting clustering algorithms ..."
    k_means = KMeans(n_clusters=n_clusters,
                     init='k-means++',
                     n_init=10,
                     max_iter=300,
                     random_state=0)

    predicted_clusters = k_means.fit_predict(report_features)

    main_cluster = k_means.predict(global_features)

    strategy_column = 'strategy'
    reporter_dataframe[strategy_column] = [
        simmodel.NOT_INFLATE_STRATEGY if cluster == main_cluster else simmodel.INFLATE_STRATEGY for
        cluster in
        predicted_clusters]

    for index, strategy in enumerate(reporter_dataframe[strategy_column].values):
        reporters_config[index][simmodel.STRATEGY_KEY] = strategy

    for strategy in [simmodel.NOT_INFLATE_STRATEGY, simmodel.INFLATE_STRATEGY]:
        reporters_per_strategy = reporter_dataframe[reporter_dataframe[strategy_column] == strategy]
        # print "Strategy: ", strategy, " reporters: ", len(reporters_per_strategy.index), " avg corrections: ", \
        #     reporters_per_strategy[CORRECTION_COLUMN].mean(), " avg non-severe prob: ", reporters_per_strategy[
        #     NON_SEVERE_COLUMN].mean(), " avg severe prob: ", reporters_per_strategy[SEVERE_COLUMN].mean()


def magnitude_relative_error(estimate, actual, balanced=False):
    """
    Normalizes the difference between actual and predicted values.
    :param estimate:Estimated by the model.
    :param actual: Real value in data.
    :return: MRE
    """

    if not balanced:
        denominator = actual
    else:
        denominator = min(estimate, actual)

    if denominator == 0:
        # 1 is our normalizing value
        # Source: http://math.stackexchange.com/questions/677852/how-to-calculate-relative-error-when-true-value-is-zero
        denominator = 1

    mre = abs(estimate - actual) / float(denominator)
    return mre


def get_mre_values(total_completed, total_predicted, balanced=False):
    """
    Returns a list of MRE values.
    :param total_completed: List of real values.
    :param total_predicted: List of predictions.
    :param balanced: True if we're using the balanced version of the MRE
    :return:
    """

    mre_values = [magnitude_relative_error(estimate, actual, balanced)
                  for estimate, actual in zip(total_completed, total_predicted)]

    return mre_values


def mean_magnitude_relative_error(total_completed, total_predicted, balanced=False):
    """
    The mean of absolute percentage errors.
    :param total_completed: List of real values.
    :param total_predicted: List of predictions.
    :return: MMRE
    """

    mre_values = get_mre_values(total_completed, total_predicted, balanced)
    return 100 * np.mean(mre_values)


def median_magnitude_relative_error(total_completed, total_predicted):
    """
    The median of absolute percentage errors.
    :param total_completed: List of real values.
    :param total_predicted: List of predictions.
    :return: MdMRE
    """

    mre_values = get_mre_values(total_completed, total_predicted)
    return 100 * np.median(mre_values)


def plot_correlation(total_predicted, total_completed, title, figtext, plot):
    """
    A scatter plot for seeing how correlation goes.
    :param total_predicted: List of predicted values.
    :param total_completed: List of real values.
    :return:
    """
    plt.clf()

    plt.scatter(total_predicted, total_completed)
    plt.title(title)
    plt.xlabel("Predicted Resolved")
    plt.ylabel("Actual Resolved")
    plt.plot([min(total_completed), max(total_completed)], [[min(total_completed)], [max(total_completed)]])
    plt.text(0.5, 0.5, figtext,
             horizontalalignment='center',
             verticalalignment='center')

    if plot:
        plt.show()
    else:
        plt.savefig("img/" + title + ".png", bbox_inches='tight')


def launch_simulation_parallel(simulation_config,
                               max_iterations,
                               parallel_blocks=gtconfig.parallel_blocks,
                               show_progress=True):
    """
    Parallel version of the simulation launch, to maximize CPU utilization.

    :param catalog_size: Number of defects present on the system.
    :param priority_generator: Generator for the priority of the defects.
    :param team_capacity:
    :param reporters_config:
    :param resolution_time_gen:
    :param max_iterations:
    :param max_time:
    :param dev_team_bandwidth:
    :param gatekeeper_config:
    :param inflation_factor:
    :param quota_system:
    :param parallel_blocks:
    :return:
    """
    pool = Pool(processes=parallel_blocks)
    samples_per_worker = max_iterations / parallel_blocks

    print "Making {} replications per worker...".format(samples_per_worker)

    worker_inputs = []

    for block_id in range(parallel_blocks):
        worker_input = {'simulation_config': simulation_config,
                        'max_iterations': samples_per_worker,
                        'block_id': block_id,
                        'show_progress': False}

        worker_inputs.append(worker_input)

    # Showing progress bar of first batch
    worker_inputs[0]['show_progress'] = show_progress
    worker_outputs = pool.map(launch_simulation_wrapper, worker_inputs)

    print "Workers in pool finished. Consolidating outputs..."
    simulation_metrics = SimulationMetrics()

    for output in worker_outputs:
        simulation_metrics.append_results(output)

    return simulation_metrics


def launch_simulation_wrapper(input_params):
    """
    A wrapper for the launch_simulation methods
    :param input_params: A dict with the input parameters.
    :return: A dict with the simulation output.
    """
    simulation_results = launch_simulation(
        max_iterations=input_params['max_iterations'],
        show_progress=input_params['show_progress'],
        block_id=input_params['block_id'],
        simulation_config=input_params['simulation_config'])

    return simulation_results


def print_strategy_report(reporters_config):
    """
    Informative information regarding the strategies adopted by the reporters
    :param reporters_config:
    :return:
    """

    default_strategy = simmodel.HONEST_STRATEGY

    strategies = set(
        [reporter[simmodel.STRATEGY_KEY].name if simmodel.STRATEGY_KEY in reporter else default_strategy for reporter in
         reporters_config])
    print "Strategies found: ", len(strategies)

    for strategy_name in strategies:

        if strategy_name is not default_strategy:
            reporters_with_strategy = [reporter for reporter in reporters_config if
                                       simmodel.STRATEGY_KEY in reporter and reporter[
                                           simmodel.STRATEGY_KEY].name == strategy_name]
        else:
            reporters_with_strategy = [reporter for reporter in reporters_config if
                                       simmodel.STRATEGY_KEY not in reporter or reporter[
                                           simmodel.STRATEGY_KEY].name == strategy_name]

        print "Strategy: ", strategy_name, " Reporters: ", len(reporters_with_strategy)


def launch_simulation(simulation_config, max_iterations, show_progress=True, block_id=-1):
    """
    Triggers the simulation according a given configuration. It includes the seed reset behaviour.

    :param quota_system: True to enable the quota-throttling system.
    :param gatekeeper_config: True to enable the gatekeeper mechanism.
    :param dev_team_bandwidth: Number of developer hours available.
    :param max_iterations: Maximum number of simulation executions.
    :param team_capacity: Number of developers in the team.
    :param reporters_config: Bug reporter configuration.
    :param resolution_time_gen: Resolution time required by developers.
    :param max_time: Simulation time.
    :return: List containing the number of fixed reports.
    """
    simulation_metrics = SimulationMetrics()

    if show_progress:
        print_strategy_report(simulation_config.reporters_config)

        print "Running ", max_iterations, " replications ..."
        print str(simulation_config)

    progress_bar = None
    if show_progress:
        progress_bar = progressbar.ProgressBar(max_iterations)

    start_time = time.time()

    for replication_index in range(max_iterations):
        current_seed = int(time.time()) + block_id
        np.random.seed(seed=current_seed)

        simulation_config.replication_id = "BLOCK" + str(block_id) + "-REP-" + str(replication_index) + "-SEED-" + str(
            current_seed)
        reporter_monitors, priority_monitors, reporting_time = simmodel.run_model(simulation_config)

        simulation_metrics.process_simulation_output(reporter_monitors, priority_monitors, reporting_time)

        if progress_bar is not None:
            progress_bar.progress(replication_index + 1)

    if show_progress:
        print max_iterations, " replications finished. Execution time: ", (time.time() - start_time), " (s)"

    return simulation_metrics


def gather_reporter_statistics(reporter_monitors, metric_name):
    """
    Extracts a per-reporter map of an specific metric.
    :param reporter_monitors: List of all reporter monitors.
    :param metric_name: Monitor of interest.
    :return: Map, containing the metric information.
    """
    metric_per_reporter = {reporter_name: reporter_info[metric_name] for reporter_name, reporter_info
                           in
                           reporter_monitors.iteritems()}

    return metric_per_reporter


def collect_metrics(true_values, predicted_values, store_metrics=False, file_prefix=""):
    """
    Returns a list of regression quality metrics.
    :param true_values: The list containing the true values.
    :param predicted_values:  The list containing the predicted values.
    :return: List of metrics.
    """

    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mar = mean_absolute_error(true_values, predicted_values)
    medar = median_absolute_error(true_values, predicted_values)
    mmre = mean_magnitude_relative_error(true_values, predicted_values, balanced=False)
    bmmre = mean_magnitude_relative_error(true_values, predicted_values, balanced=True)
    mdmre = median_magnitude_relative_error(true_values, predicted_values)
    r_squared = r2_score(true_values, predicted_values)

    if store_metrics:
        mre_values = get_mre_values(true_values, predicted_values)

        metrics_dataframe = pd.DataFrame({'true_values': true_values,
                                          'predicted_values': predicted_values,
                                          'mre_values': mre_values})

        metrics_dataframe.to_csv("csv/pred_metrics_" + file_prefix + ".csv", index=False)
    return mse, rmse, mar, medar, mmre, bmmre, mdmre, r_squared


def collect_and_print(project_key, description, total_completed, total_predicted):
    """
    Calls the collect metrics and prints the results.
    :param project_key:
    :param description:
    :param total_completed:
    :param total_predicted:
    :return:
    """

    mse, rmse, mar, medar, mmre, bmmre, mdmre, r_squared = collect_metrics(total_completed, total_predicted,
                                                                           store_metrics=True, file_prefix="_".join(
            project_key) + "_" + description)

    print  description, " in Project ", project_key, " on ", len(
        total_predicted), " datapoints ->  Root Mean Squared Error (RMSE):", rmse, " Mean Squared Error (MSE): ", mse, " Mean Absolute Error (MAE): ", \
        mar, " Median Absolute Error (MdAE): ", medar, " Mean Magnitude Relative Error (MMRE): ", mmre, " Balanced MMRE :", \
        bmmre, "Median Magnitude Relative Error (MdMRE): ", mdmre, " R-squared ", r_squared

    return mmre, mdmre
