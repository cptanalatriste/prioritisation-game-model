"""
Utility types for supporting the simulation.
"""
from collections import defaultdict

import numpy as np
import sys

import pandas as pd

from scipy.stats import uniform
from scipy.stats import rv_discrete

# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

import matplotlib.pyplot as plt

import simdata
import simmodel

MINIMUM_OBSERVATIONS = 3


class ContinuousEmpiricalDistribution:
    def __init__(self, observations=None, distribution=None, parameters=None):
        self.distribution = None
        self.parameters = None

        if observations is not None:
            if len(observations) < MINIMUM_OBSERVATIONS:
                raise ValueError("Only " + str(len(observations)) + " were provided.")

            self.observations = observations
            self.sorted_observations = sorted(set(self.observations))

            items = float(len(self.observations))
            self.empirical_cdf = [sum(1 for item in observations if item <= observation) / items
                                  for observation in self.sorted_observations]

        if distribution is not None and parameters is not None:
            self.distribution = distribution
            self.parameters = parameters

    def generate_from_scipy(self):
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

        Based on: http://sms.victoria.ac.nz/foswiki/pub/Courses/OPRE354_2016T1/Python/bites_of_python.pdf

        :return:Random variate
        """

        if rand_uniform is None:
            rand_uniform = uniform.rvs(size=1)[0]

        if self.distribution is not None and self.parameters is not None:
            return self.generate_from_scipy()

        k = 0
        for index, cdf in enumerate(self.empirical_cdf):
            if cdf > rand_uniform:
                if index > 0:
                    k = index - 1
                    element_k = self.sorted_observations[k]
                    cdf_k = self.empirical_cdf[k]

                    element_k_next = self.sorted_observations[k + 1]
                    cdf_k_next = self.empirical_cdf[k + 1]
                    break
                else:
                    return self.sorted_observations[k]

        # print "k", k, "element_k ", element_k, " cdf_k ", cdf_k, " element_k_next ", element_k_next, " cdf_k_next ", cdf_k_next, " rand_uniform ", rand_uniform
        #
        # print "rand_uniform ", rand_uniform
        # print "self.empirical_cdf ", self.empirical_cdf

        rand_variate = element_k + (rand_uniform - cdf_k) / \
                                   float(cdf_k_next - cdf_k) * (element_k_next - element_k)

        return rand_variate


class DiscreteEmpiricalDistribution:
    def __init__(self, name="", observations=None, values=None, probabilities=None):
        if observations is not None:
            values_with_probabilities = observations.value_counts(normalize=True)
            values = np.array([index for index, _ in values_with_probabilities.iteritems()])
            probabilities = [probability for _, probability in values_with_probabilities.iteritems()]

        self.name = name
        self.values = values
        self.probabilities = probabilities

        self.disc_distribution = rv_discrete(values=(range(len(values)), self.probabilities))

    def generate(self, rand_uniform=None):
        """
        Samples from the empirical distribution. Inspired in:
        http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy

        :return: Random variate
        """
        variate_index = self.disc_distribution.rvs(size=1)
        return self.values[variate_index]

    def get_probabilities(self):
        probability_map = {value: self.probabilities[index] for index, value in enumerate(self.values)}
        return defaultdict(float, probability_map)


def remove_drive_in_testers(reporters_config, min_reports):
    """
    Removes drive-in testers, defined as the testers who's mean interrival time is larger than range_in_std standard deviations of the
    overall average interarrival times.
    :param reporters_config: Reporter configuration.
    :param range_in_std: Number of standard deviations to be considered in range.
    :return: Filtered list of reporters config.
    """
    # reporter_metrics = [np.mean(config['interarrival_time_gen'].observations) for config in reporters_config]
    reporter_metrics = [len(config['interarrival_time_gen'].observations) for config in reporters_config]

    overall_mean = np.mean(reporter_metrics)
    overall_std = np.std(reporter_metrics)
    overall_max = np.max(reporter_metrics)
    overall_min = np.min(reporter_metrics)

    print "remove_drive_in_testers->min_reports ", min_reports, "overall_mean ", overall_mean, "overall_std ", overall_std, "overall_max", overall_max, "overall_min", overall_min

    engaged_testers = [config for config in reporters_config if
                       len(config['interarrival_time_gen'].observations) >= min_reports]

    return engaged_testers


def assign_strategies(reporters_config, training_issues, debug=False):
    """
    Assigns an inflation pattern to the reporter based on clustering.
    :param reporters_config: Reporter configuration.
    :param training_issues: Training dataset.
    :return: Reporting Configuration including inflation pattern.
    """
    reporter_records = [
        [config['priority_map'][simdata.NON_SEVERE_PRIORITY], config['priority_map'][simdata.NORMAL_PRIORITY],
         config['priority_map'][simdata.SEVERE_PRIORITY],
         config['with_modified_priority']] for config in reporters_config]

    global_priority_map = DiscreteEmpiricalDistribution(
        observations=training_issues[simdata.SIMPLE_PRIORITY_COLUMN]).get_probabilities()

    if debug:
        print "global_priority_map: ", global_priority_map

    reporter_dataframe = pd.DataFrame(reporter_records)
    correction_column = "Corrections"
    non_severe_column = "Non-Severe"
    severe_column = "Severe"
    normal_column = "Normal"

    reporter_dataframe.columns = [non_severe_column, normal_column, severe_column, correction_column]

    # Removing scaling because of cluster quality.
    # scaler = StandardScaler()
    # report_features = scaler.fit_transform(reporter_dataframe.values)
    # global_features = scaler.transform(
    #     [global_priority_map[simdata.NON_SEVERE_PRIORITY], global_priority_map[simdata.NORMAL_PRIORITY],
    #      global_priority_map[simdata.SEVERE_PRIORITY], 0.0])

    global_features = [global_priority_map[simdata.NON_SEVERE_PRIORITY], global_priority_map[simdata.NORMAL_PRIORITY],
                       global_priority_map[simdata.SEVERE_PRIORITY], 0.0]
    report_features = reporter_dataframe.values

    print "Starting clustering algorithms ..."
    k_means = KMeans(n_clusters=2,
                     init='random',
                     max_iter=300,
                     tol=1e-04,
                     random_state=0)

    predicted_clusters = k_means.fit_predict(report_features)

    main_cluster = k_means.predict(global_features)

    strategy_column = 'strategy'
    reporter_dataframe[strategy_column] = [
        simmodel.NOT_INFLATE_STRATEGY if cluster == main_cluster else simmodel.INFLATE_STRATEGY for
        cluster in
        predicted_clusters]

    for index, strategy in enumerate(reporter_dataframe[strategy_column].values):
        reporters_config[index]['strategy'] = strategy

    for strategy in [simmodel.NOT_INFLATE_STRATEGY, simmodel.INFLATE_STRATEGY]:
        reporters_per_strategy = reporter_dataframe[reporter_dataframe[strategy_column] == strategy]
        print "Strategy: ", strategy, " reporters: ", len(reporters_per_strategy.index), " avg corrections: ", \
            reporters_per_strategy[correction_column].mean(), " avg non-severe prob: ", reporters_per_strategy[
            non_severe_column].mean(), " avg normal prob: ", reporters_per_strategy[
            normal_column].mean(), " avg severe prob: ", reporters_per_strategy[severe_column].mean()


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


def mean_magnitude_relative_error(total_completed, total_predicted, balanced=False):
    """
    The mean of absolute percentage errors.
    :param total_completed: List of real values.
    :param total_predicted: List of predictions.
    :return: MMRE
    """
    return 100 * np.mean(
        [magnitude_relative_error(estimate, actual, balanced)
         for estimate, actual in zip(total_completed, total_predicted)])


def median_magnitude_relative_error(total_completed, total_predicted):
    """
    The median of absolute percentage errors.
    :param total_completed: List of real values.
    :param total_predicted: List of predictions.
    :return: MdMRE
    """
    return 100 * np.median(
        [magnitude_relative_error(estimate, actual)
         for estimate, actual in zip(total_completed, total_predicted)])


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
        plt.savefig("img/" + title + ".png")


def launch_simulation(team_capacity, report_number, reporters_config, resolution_time_gen, priority_gen,
                      max_time, max_iterations, dev_team_bandwith=sys.maxint, gatekeeper_config=False,
                      quota_system=False):
    """
    Triggers the simulation according a given configuration.

    :param max_iterations: Maximum number of simulation executions.
    :param team_capacity: Number of developers in the team.
    :param report_number: Number of bugs for the period.
    :param reporters_config: Bug reporter configuration.
    :param resolution_time_gen: Resolution time required by developers.
    :param priority_gen: The priority contained on the bug reports.
    :param max_time: Simulation time.
    :return: List containing the number of fixed reports.
    """

    completed_per_reporter = []
    completed_per_priority = []
    reported_per_priotity = []
    bugs_per_reporter = []
    reports_per_reporter = []

    for _ in range(max_iterations):
        np.random.seed()
        reporter_monitors, priority_monitors = simmodel.run_model(team_capacity=team_capacity,
                                                                  report_number=report_number,
                                                                  reporters_config=reporters_config,
                                                                  resolution_time_gen=resolution_time_gen,
                                                                  priority_gen=priority_gen,
                                                                  max_time=max_time,
                                                                  dev_team_bandwith=dev_team_bandwith,
                                                                  gatekeeper_config=gatekeeper_config,
                                                                  quota_system=quota_system)

        result_per_reporter = {reporter_name: reporter_info['resolved_monitor'].count() for reporter_name, reporter_info
                               in
                               reporter_monitors.iteritems()}
        completed_per_reporter.append(result_per_reporter)

        result_per_priority = {priority: monitors['completed'].count() for priority, monitors in
                               priority_monitors.iteritems()}
        completed_per_priority.append(result_per_priority)

        reports_per_priority = {priority: monitors['reported'] for priority, monitors in
                                priority_monitors.iteritems()}
        reported_per_priotity.append(reports_per_priority)

        found_per_reporter = {reporter_name: reporter_info['priority_counters'] for reporter_name, reporter_info
                              in
                              reporter_monitors.iteritems()}
        bugs_per_reporter.append(found_per_reporter)

        reported_per_reporter = {reporter_name: reporter_info['report_counters'] for reporter_name, reporter_info
                                 in
                                 reporter_monitors.iteritems()}
        reports_per_reporter.append(reported_per_reporter)

    return completed_per_reporter, completed_per_priority, bugs_per_reporter, reports_per_reporter, reported_per_priotity


def collect_metrics(true_values, predicted_values):
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

    return mse, rmse, mar, medar, mmre, bmmre, mdmre


def collect_and_print(project_key, description, total_completed, total_predicted):
    """
    Calls the collect metrics and prints the results.
    :param project_key:
    :param description:
    :param total_completed:
    :param total_predicted:
    :return:
    """

    mse, rmse, mar, medar, mmre, bmmre, mdmre = collect_metrics(total_completed, total_predicted)

    print  description, " in Project ", project_key, " on ", len(
        total_predicted), " datapoints ->  Root Mean Squared Error (RMSE):", rmse, " Mean Squared Error (MSE): ", mse, " Mean Absolute Error (MAE): ", \
        mar, " Median Absolute Error (MdAE): ", medar, " Mean Magnitude Relative Error (MMRE): ", mmre, " Balanced MMRE :", \
        bmmre, "Median Magnitude Relative Error (MdMRE): ", mdmre

    return mmre, mdmre
