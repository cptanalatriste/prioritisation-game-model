import unittest
import simutils


class TestContinuousEmpiricalDistribution(unittest.TestCase):
    def test_generate(self):
        """
        This test case was taken from: http://sms.victoria.ac.nz/foswiki/pub/Courses/OPRE354_2016T1/Python/bites_of_python.pdf

        :return:
        """

        observations = [3.8, 7.5, 8.0, 1.9, 4.5, 6.6, 7.1, 7.5, 2.8, 4.5]
        distribution = simutils.ContinuousEmpiricalDistribution(observations)

        random_variate = distribution.generate(rand_uniform=0.8)
        expected = 7.3
        self.assertAlmostEqual(expected, random_variate)
