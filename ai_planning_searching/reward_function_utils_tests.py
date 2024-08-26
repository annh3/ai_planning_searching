"""
python -m unittest reward_function_utils_tests.py
"""

import unittest
import datasets
from datasets.arrow_dataset import Dataset
from reward_function_utils import load_apps, problem_unit_tests, run_apps_evals
from transformers import GPT2LMHeadModel, GPT2TokenizerFast 


class testMCTSUtils(unittest.TestCase):

    def setUp(self):
        # setup a hugging face model here
        self.pretrained_weights = 'gpt2'
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.pretrained_weights)
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_weights)

    def tearDown(self):
        pass

    def test_load_apps(self):
        ds = load_apps("train")
        self.assertIsInstance(ds, Dataset)


    """
    obj = ["apple", "cherry", "melon", "strawberry"]
    code = "print([sliced[:4] for sliced in obj if 'a' not in sliced])"

    exec(code)

    .........
    ['cher', 'melo']
    """
    def test_exec_program(self):
        x = 5
        # assume program is the return of the lm_call
        program = """
        def exponentiate(x):
          return x**2
        """ 
        res = exec(program)
        self.assertEqual(res, 25)

    """
    TODO(annhe): Rewrite this function.

    Construct a dummy dataset iterator, pass to problem_unit_tests, then
    pass to run_apps_evals
    """
    def test_run_apps_evals(self):
        unit_test_inputs = [5]
        unit_test_outputs = [25]
        program = """
        def exponentiate(x):
          return x**2
        """ 
        pass_rate = run_apps_evals(program, unit_test_inputs, unit_test_outputs)
        self.assertEqual(int(pass_rate), 1)

    """
    TODO(annhe): Rewrite this function.

    Construct a dummy dataset iterator, pass to problem_unit_tests
    """
    def test_problem_unit_tests(self):
        ds = load_apps("test")
        problem_unit_test_fn = problem_unit_tests(ds)

        for question, solution, inputs, outputs in problem_unit_test_fn:
            self.assertIsInstance(question, str)
            self.assertIsInstance(solution, str)
            self.assertIsInstance(inputs, list)
            self.assertIsInstance(outputs, list)
