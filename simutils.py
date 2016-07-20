"""
Utility types for supporting the simulation.
"""

from scipy.stats import uniform


class EmpiricalDistribution:
    def __init__(self, observations):
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
                k = index - 1
                element_k = self.sorted_observations[k]
                cdf_k = self.empirical_cdf[k]

                element_k_next = self.sorted_observations[index]
                cdf_k_next = cdf
                break

        rand_variate = element_k + (rand_uniform - cdf_k) / \
                                   float(cdf_k_next - cdf_k) * (element_k_next - element_k)

        return rand_variate
