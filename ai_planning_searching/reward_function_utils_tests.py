"""
python -m unittest reward_functino_utils_tests.py
"""

import unittest
from dataset.arrow_dataset import Dataset
from reward_function_utils import load_apps, problem_unit_tests


class testMCTSUtils(unittest.TestCase):

	def test_load_apps(self):
		ds = load_apps("train")
		self.assertIsInstance(ds, Dataset)


	def test_problem_unit_tests(self):
		ds = load_apps("test")
		problem_unit_test_fn = problem_unit_tests(ds)

		for question, solution, inputs, outputs in problem_unit_test_fn:
			self.assertIsInstance(question, str)
			self.assertIsInstance(solution, str)
			self.assertIsInstance(inputs, list[str])
			self.assertIsInstance(outputs, list[str])
