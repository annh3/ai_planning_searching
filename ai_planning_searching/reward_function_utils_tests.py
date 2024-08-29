"""
python -m unittest reward_function_utils_tests.py
"""

import unittest
import datasets
import json
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


    def countdown(n):
        i = n 
        while i >= 0:
            yield i
            i -= 1


    """
    obj = ["apple", "cherry", "melon", "strawberry"]
    code = "print([sliced[:4] for sliced in obj if 'a' not in sliced])"

    exec(code)

    .........
    ['cher', 'melo']
    """
    def test_exec_program(self):
        program = """
x = 5

return_val = x**2"""
        loc = {}
        exec(program, globals(), loc)
        res = loc['return_val']
        self.assertEqual(res, 25)


    def test_run_apps_evals(self):
        data = [["question", "solution_program", [5],[25]]]
        program = f"""
        x = {input_value}

        output_value = x**2"""
        pass_rate = run_apps_evals(program, data)
        self.assertIsInstance(pass_rate, float)

"""
    def test_problem_unit_tests(self):
        #ds_iterator = pretend_dataset_list()
        data = problem_unit_tests(ds_iterator)

        for program, solution, unit_test_inputs, unit_test_outputs in data:
            self.assertIsInstance(program, str)
            self.assertIsInstance(solution, str)
            self.assertIsInstance(unit_test_inputs, list)
            self.assertIsInstance(unit_test_outputs, list)
"""