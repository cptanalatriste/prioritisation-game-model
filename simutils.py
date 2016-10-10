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
import simutils

MINIMUM_OBSERVATIONS = 3

NON_SEVERE_TRUE_COLUMN = "Non-Severe-True"
NON_SEVERE_INFLATED_COLUMN = "Non-Severe-Inflated"
SEVERE_TRUE_COLUMN = "Severe-True"
SEVERE_DEFLATED_COLUMN = "Severe-Deflated"
CORRECTION_COLUMN = "Corrections"
NONSEVERE_CORRECTION_COLUMN = "NonSevere-Corrections"
SEVERE_CORRECTION_COLUMN = "Severe-Corrections"

REPORTER_COLUMNS = [NON_SEVERE_INFLATED_COLUMN, SEVERE_DEFLATED_COLUMN, CORRECTION_COLUMN, NONSEVERE_CORRECTION_COLUMN,
                    SEVERE_CORRECTION_COLUMN]


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
    :param min_reports: Minimum number of reports to be considered non drive-by.
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


def get_reporter_behavior_dataframe(reporters_config):
    """
    Returns a dataframe describing bug reporter behaviour.
    :param reporters_config: List of reporter configuration.
    :return: Dataframe with behaviour information.
    """

    reporter_records = []
    for config in reporters_config:
        total_nonsevere = float(config['reports_per_priority'][simdata.NON_SEVERE_PRIORITY])
        total_severe = float(config['reports_per_priority'][simdata.SEVERE_PRIORITY])
        total_modified = float(config['with_modified_priority'])

        non_severe_true = config['modified_details']['priority_1_true']
        non_severe_false = config['modified_details']['priority_1_false']
        non_severe_modified = non_severe_true + non_severe_false

        severe_true = config['modified_details']['priority_3_true']
        severe_false = config['modified_details']['priority_3_false']
        severe_modified = severe_true + severe_false

        reporter_records.append(
            [non_severe_false / non_severe_modified if non_severe_modified != 0 else 0,
             severe_false / severe_modified if severe_modified != 0 else 0,
             total_modified / (total_nonsevere + total_severe),
             non_severe_modified / total_nonsevere if total_nonsevere != 0 else 0,
             severe_modified / total_severe if total_severe != 0 else 0])

    reporter_dataframe = pd.DataFrame(reporter_records)

    reporter_dataframe.columns = REPORTER_COLUMNS
    return reporter_dataframe


def elbow_method_for_reporters(reporter_configuration):
    """
    Elbow method implementation for estimating the optimal number of clusters for a given task.
    :param reporter_configuration: List of reporter configuration info.
    :return: Generates the elbow method plot as a file.
    """
    distortions = []

    reporters_with_corrections = [config for config in reporter_configuration if
                                  config['with_modified_priority'] > 0]
    reporter_dataframe = get_reporter_behavior_dataframe(reporters_with_corrections)

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
    plt.savefig("img/elbow_for_reporters.png", bbox_inches='tight')


def assign_strategies(reporters_config, training_issues, n_clusters=3, debug=False):
    """
    Assigns an inflation pattern to the reporter based on clustering.
    :param reporters_config: Reporter configuration.
    :param training_issues: Training dataset.
    :return: Reporting Configuration including inflation pattern.
    """

    global_priority_map = DiscreteEmpiricalDistribution(
        observations=training_issues[simdata.SIMPLE_PRIORITY_COLUMN]).get_probabilities()

    if debug:
        print "global_priority_map: ", global_priority_map

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
        reporters_config[index]['strategy'] = strategy

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
        plt.savefig("img/" + title + ".png", bbox_inches='tight')


def launch_simulation(team_capacity, bugs_by_priority, reporters_config, resolution_time_gen,
                      max_iterations,
                      max_time=sys.maxint, dev_team_bandwidth=sys.maxint, gatekeeper_config=False,
                      quota_system=False):
    """
    Triggers the simulation according a given configuration. It includes the seed reset behaviour.

    :param quota_system: True to enable the quota-throttling system.
    :param gatekeeper_config: True to enable the gatekeeper mechanism.
    :param dev_team_bandwidth: Number of developer hours available.
    :param max_iterations: Maximum number of simulation executions.
    :param team_capacity: Number of developers in the team.
    :param bugs_by_priority: Number of bugs for the period per priority.
    :param reporters_config: Bug reporter configuration.
    :param resolution_time_gen: Resolution time required by developers.
    :param max_time: Simulation time.
    :return: List containing the number of fixed reports.
    """

    # TODO(cgavidia): This also screams refactoring.
    completed_per_reporter = []
    completed_per_priority = []
    reported_per_priotity = []
    bugs_per_reporter = []
    reports_per_reporter = []
    resolved_per_reporter = []

    print "Running ", max_iterations, " replications: ",
    for replication_index in range(max_iterations):
        print replication_index + 1, " ",
        np.random.seed()
        reporter_monitors, priority_monitors = simmodel.run_model(team_capacity=team_capacity,
                                                                  bugs_by_priority=bugs_by_priority,
                                                                  reporters_config=reporters_config,
                                                                  resolution_time_gen=resolution_time_gen,
                                                                  max_time=max_time,
                                                                  dev_team_bandwith=dev_team_bandwidth,
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

        bugs_per_reporter.append(gather_reporter_statistics(reporter_monitors, 'priority_counters'))
        reports_per_reporter.append(gather_reporter_statistics(reporter_monitors, 'report_counters'))
        resolved_per_reporter.append(gather_reporter_statistics(reporter_monitors, 'resolved_counters'))

    print
    return completed_per_reporter, completed_per_priority, bugs_per_reporter, reports_per_reporter, resolved_per_reporter


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
