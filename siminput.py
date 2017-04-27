"""
This modules does the analysis required to find the probability distributions and its parameters for the simulation input
"""
import simdata
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy import arange

import gtconfig

import matplotlib

if not gtconfig.is_windows:
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# ALL_ISSUES_CSV = "C:\Users\Carlos G. Gavidia\git\github-data-miner\UNFILTERED\Release_Counter_UNFILTERED_SPARK.csv"

def date_as_string(report_series):
    """
    Returns a string representation of the created
    :param report_series:
    :return:
    """
    parsed_date = simdata.parse_create_date(report_series)
    return str(parsed_date.year) + "-" + str(parsed_date.month)


def plot_empirical_data(data_series):
    """
    Adds the empirical data to the plot
    :param data_series:
    :return:
    """
    # Using the Freedman Diacones Estimator for bin count.
    hist, bin_edges = np.histogram(data_series.values, bins="auto")
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], color='white', alpha=0.5)
    plt.grid(True)
    plt.xlim(min(bin_edges), max(bin_edges))


def apply_anderson_darling(dist_name, data_series, debug=False):
    """
    Applies the Anderson-Darling Test for Goodness-of-Fit

    H0: data_series are distributed as dist_name.
    Reject H0 if statistic > critical_value at significance_level

    :param dist_name: Distribution name.
    :param data_series: Data point.
    :return: None.
    """
    if dist_name in ["norm", "expon"]:
        # According to Ivezic Anderson-Darling is better for normal. Also, scipy says this works for exponential too.
        statistic, critical_values, significance_level = stats.anderson(data_series, dist_name)

        if debug:
            print "Anderson-Darling Test for ", dist_name, ": statistic ", statistic
            for critical_value, significance_level in zip(critical_values, significance_level):
                print "Critical Value: ", critical_value, " Significance Level: ", significance_level
    else:
        if debug:
            print "Anderson-Darling is not suitable for ", dist_name


def apply_kolmogorov_smirnov(dist_name, cdf_function, data_series, debug=False):
    """
    Applies the Kolmogorov-Smirnov Test for Goodness-of-Fit.

    H0: data_series is equal to the distribution dist_name with cdf_function.
    Accept H0 if p_value is high

    :param dist_name: Distribution name.
    :param cdf_function: Cummulative distribution function.
    :param data_series: Data points.
    :return: None.
    """
    if True:
        # if dist_name not in ["norm", "expon"]:
        # According to Ivezic, Kolmogorv-Smirnov is a poor choice for those distributions.
        d, p_value = stats.kstest(data_series, cdf_function)

        if debug:
            print "Kolmogorov-Smirnov Test for ", dist_name, ": d ", d, " p_value: ", p_value

    return p_value


def fit_probability_distribution(dist_name, distribution, data_series, xmin, xmax, debug=True):
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

    if not xmin:
        xmin = data_series.min()

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

    ks_p_value = apply_kolmogorov_smirnov(dist_name, cdf_function, data_series)
    apply_anderson_darling(dist_name, data_series)
    plt.plot(counts, label=dist_name)

    if debug:
        print "Fitted distribution params for ", dist_name, ": ", parameter_tuple, " ks_p_value: ", ks_p_value

    return {"dist_name": dist_name,
            "ks_p_value": ks_p_value,
            "distribution": distribution,
            "parameters": parameter_tuple}


def launch_input_analysis(data_series, desc="default", show_data_plot=True, save_plot=True):
    """
    The input analysis includes the following activities: Show data statistics, plot an histogram of the data points,
    fit theoretical distributions, start a ks-test of the fitted distribution, plot the theoretical distributions.

    :param desc: Series description, for file generation analysis.
    :param data_series: Data points.
    :param show_data_plot: True for showing the plot, false otherwise.
    :return: None.
    """
    xmin = None
    xmax = None

    plt.clf()
    plot_empirical_data(data_series)

    p_values = []

    p_values.append(fit_probability_distribution("uniform", stats.uniform, data_series, xmin, xmax))
    p_values.append(fit_probability_distribution("triang", stats.triang, data_series, xmin, xmax))
    p_values.append(fit_probability_distribution("norm", stats.norm, data_series, xmin, xmax))
    p_values.append(fit_probability_distribution("gamma", stats.gamma, data_series, xmin, xmax))
    p_values.append(fit_probability_distribution("lognorm", stats.lognorm, data_series, xmin, xmax))
    p_values.append(fit_probability_distribution("expon", stats.expon, data_series, xmin, xmax))
    p_values.append(fit_probability_distribution("powerlaw", stats.powerlaw, data_series, xmin, xmax))

    plt.legend(loc='upper right')

    if show_data_plot:
        plt.show()

    if save_plot:
        plt.savefig("img/" + desc + ".png")

    return max(p_values, key=lambda dist: dist["ks_p_value"])


def get_discrete_distribution(data_series):
    print "Relative frequencies: \n", data_series.value_counts(normalize=True)

    values_with_probabilities = data_series.value_counts(normalize=True)
    values = np.array([index for index, _ in values_with_probabilities.iteritems()])
    probabilities = [probability for _, probability in values_with_probabilities.iteritems()]

    disc_distribution = stats.rv_discrete(values=(range(len(values_with_probabilities)), probabilities))

    return values, disc_distribution


def main():
    dataframe = pd.read_csv(simdata.ALL_ISSUES_CSV)
    dataframe = simdata.enhace_report_dataframe(dataframe)

    print "Original dataframe issues ", len(dataframe.index)

    resolved_issues = simdata.filter_resolved(dataframe)
    fix_effort_data = resolved_issues[simdata.RESOLUTION_TIME_COLUMN]
    fix_effort_data = fix_effort_data.dropna()

    # print "Input analysis for Fix Effort ..."
    # launch_input_analysis(fix_effort_data, False)

    # Note that we're considering all reports. No filtering is done so far.
    priorities_data = dataframe['Priority']
    values, priorities_dist = get_discrete_distribution(priorities_data)

    # Considering only resolved and third-party reports
    # with_fix_effort = resolved_issues.dropna(subset=[fix_effort_column])
    author_column = 'Reported By'
    group_by_column = 'Month'
    # group_by_column = author_column

    grouped_reports = dataframe.groupby([group_by_column]).size().order(ascending=False)
    print "grouped_reports \n", grouped_reports.describe()

    category_index = 0
    # category_index = 14

    category_value = grouped_reports.index[category_index]
    category_bugs = dataframe[dataframe[group_by_column] == category_value]

    print "category_value ", category_value, " category_bugs ", len(category_bugs.index)

    # Further author filtering
    print "Before author filtering ", len(category_bugs.index), " bugs"
    author_value = "mengxr"
    category_bugs = category_bugs[category_bugs[author_column] == author_value]
    print "author_value ", author_value, " category_bugs ", len(category_bugs.index)

    report_date = category_bugs.apply(simdata.parse_create_date, axis=1)

    report_date = report_date.order()
    print "report_date: head", report_date.head(1), "tail: ", report_date.tail(1)

    batches = []
    for position, created_date in enumerate(report_date.values):

        if len(batches) == 0:
            batches.append({"batch_head": created_date,
                            "batch_count": 1})
        else:
            last_batch_head = batches[-1]["batch_head"]
            distance = created_date - last_batch_head

            batch_size = 1
            if distance / np.timedelta64(1, 'h') <= batch_size:
                batches[-1]["batch_count"] += 1
            else:
                batches.append({"batch_head": created_date,
                                "batch_count": 1})

    interrival_times = []
    # arrival_times = report_date.values
    arrival_times = [batch["batch_head"] for batch in batches]

    print "batches ", batches
    print "arrival_times ", arrival_times
    for position, created_date in enumerate(arrival_times):
        if position > 0:
            distance = created_date - report_date.values[position - 1]

            if isinstance(distance, datetime.timedelta):
                time = distance.total_seconds() / simdata.TIME_FACTOR
            else:
                time = distance / np.timedelta64(1, 's') / simdata.TIME_FACTOR

            interrival_times.append(time)

    interrival_data = pd.Series(data=interrival_times)

    print "Input analysis for Interarrival Time ..."
    launch_input_analysis(interrival_data, True)

    print "Input analysis for Batch Size..."
    batch_sizes = [batch["batch_count"]]


if __name__ == "__main__":
    main()
