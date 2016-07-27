"""
Utility types for supporting the simulation.
"""
from collections import defaultdict

import numpy as np

from scipy.stats import uniform
from scipy.stats import rv_discrete

MINIMUM_OBSERVATIONS = 10


class ContinuousEmpiricalDistribution:
    def __init__(self, observations):
        if len(observations) < MINIMUM_OBSERVATIONS:
            raise ValueError("Only " + str(len(observations)) + " were provided.")

        self.sorted_observations = sorted(set(observations))

        items = float(len(observations))
        self.empirical_cdf = [sum(1 for item in observations if item <= observation) / items
                              for observation in self.sorted_observations]

    def generate(self, rand_uniform=None):
        """
        Samples from this empirical distribution using the Inverse Transform method.

        Based on: http://sms.victoria.ac.nz/foswiki/pub/Courses/OPRE354_2016T1/Python/bites_of_python.pdf

        :return:Random variate
        """

        if rand_uniform is None:
            rand_uniform = uniform.rvs(size=1)[0]

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
    def __init__(self, observations=None, values=None, probabilities=None):
        if observations is not None:
            values_with_probabilities = observations.value_counts(normalize=True)
            values = np.array([index for index, _ in values_with_probabilities.iteritems()])
            probabilities = [probability for _, probability in values_with_probabilities.iteritems()]

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
