"""
This modules does the analysis required to find the probability distributions and its parameters for the simulation input
"""

import pandas as pd
import numpy as np
import dateutil.parser
import matplotlib.pyplot as plt
from scipy import stats
from scipy import arange

VALID_RESOLUTION_VALUES = ['Done', 'Implemented', 'Fixed']

ALL_ISSUES_CSV = "C:\Users\Carlos G. Gavidia\git\github-data-miner\UNFILTERED\Release_Counter_UNFILTERED_SPARK.csv"


# ALL_ISSUES_CSV = "C:\Users\Carlos G. Gavidia\git\github-data-miner\UNFILTERED\Release_Counter_UNFILTERED.csv"


def get_fix_effort(report_series):
    """
    Calculates the fix effort in days. It is defined as the days between the resolution and the "In Progress" status
    change by the resolver.

    :param report_series: Bug report as Series.
    :return: Fix effort in days.
    """

    first_contact_str = report_series['JIRA Resolver In Progress']
    resolution_date_str = report_series['JIRA Resolved Date']

    if isinstance(first_contact_str, basestring) and isinstance(resolution_date_str, basestring):
        first_contact = dateutil.parser.parse(first_contact_str)
        resolution_date = dateutil.parser.parse(resolution_date_str)

        return (resolution_date - first_contact).total_seconds() / (60.0 * 60.0 * 24.0)

    return None


def plot_empirical_data(data_series):
    """
    Adds the empirical data to the plot
    :param data_series:
    :return:
    """
    # Using the Freedman Diacones Estimator for bin count.
    values, edges = np.histogram(data_series, bins="auto")
    data_series.hist(bins=edges, color='w')


def plot_probability_distribution(dist_name, distribution, data_series, xmin, xmax):
    """
    Plots and fitted distribution using the maximum likelihood estimation. Also, before that the Kolmogorov-Smirnov test is
    performed.

    :param dist_name: Distribution name
    :param distribution: Function representing the distribution from the scipy.stats module.
    :param data_series: Data to fit.
    :param xmin: Minimum value to plot
    :param xmax: Maximum value to plot
    :return: None
    """

    # Distribution fitting through maximum likelihood estimation.
    parameter_tuple = distribution.fit(data_series)
    print "Fitted distribution params for ", dist_name, ": ", parameter_tuple

    if not xmin:
        xmin = 0

    if not xmax:
        xmax = data_series.max()

    x_values = arange(start=xmin, stop=xmax)
    cdf_function = None

    if len(parameter_tuple) == 2:
        loc = parameter_tuple[0]
        scale = parameter_tuple[1]
        counts = distribution.pdf(x_values, loc=loc, scale=scale) * data_series.count()
        cdf_function = lambda x: distribution.cdf(x, loc=loc, scale=scale)

    elif len(parameter_tuple) == 3:
        shape = parameter_tuple[0]
        loc = parameter_tuple[1]
        scale = parameter_tuple[2]

        counts = distribution.pdf(x_values, shape, loc=loc,
                                  scale=scale) * data_series.count()
        cdf_function = lambda x: distribution.cdf(x, shape, loc=loc, scale=scale)

    d, p_value = stats.kstest(data_series, cdf_function)
    print "Kolmogorov-Smirnov Test for ", dist_name, ": d ", d, " p_value: ", p_value

    plt.plot(counts, label=dist_name)


def launch_input_analysis(data_series, show_data_plot=True):
    """
    The input analysis includes the following activities: Show data statistics, plot an histogram of the data points,
    fit theoretical distributions, start a ks-test of the fitted distribution, plot the theoretical distributions.

    :param data_series: Data points.
    :param show_data_plot: True for showing the plot, false otherwise.
    :return: None.
    """
    print "data_series: \n", data_series.describe()

    xmin = 0
    xmax = None

    plot_empirical_data(data_series)
    plot_probability_distribution("uniform", stats.uniform, data_series, xmin, xmax)
    plot_probability_distribution("triang", stats.triang, data_series, xmin, xmax)
    plot_probability_distribution("norm", stats.norm, data_series, xmin, xmax)
    plot_probability_distribution("gamma", stats.gamma, data_series, xmin, xmax)
    plot_probability_distribution("lognorm", stats.lognorm, data_series, xmin, xmax)

    if show_data_plot:
        # plt.xlim(xmin, xmax)
        plt.legend(loc='upper right')
        plt.show()


def main():
    dataframe = pd.read_csv(ALL_ISSUES_CSV)
    print "Original dataframe issues ", len(dataframe.index)

    resolved_issues = dataframe[dataframe['Status'].isin(['Closed', 'Resolved'])]
    resolved_issues = resolved_issues[resolved_issues['Resolution'].isin(VALID_RESOLUTION_VALUES)]
    resolved_issues = resolved_issues[resolved_issues['Commits'] > 0]
    # resolved_issues = resolved_issues[resolved_issues['Reported By'] != resolved_issues['JIRA Resolved By']]

    fix_effort_data = resolved_issues.apply(get_fix_effort, axis=1)
    fix_effort_data = fix_effort_data.dropna()

    launch_input_analysis(fix_effort_data, True)


if __name__ == "__main__":
    main()
