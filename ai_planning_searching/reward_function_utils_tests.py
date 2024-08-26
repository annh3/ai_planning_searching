"""
python -m unittest reward_function_utils_tests.py
"""

import unittest
import datasets
import json
from datasets.arrow_dataset import Dataset
from reward_function_utils import load_apps, problem_unit_tests, run_apps_evals
from transformers import GPT2LMHeadModel, GPT2TokenizerFast 



str_1 = json.dumps({ "inputs": [ "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n" ], "outputs": [ "1\n3 \n-1\n0\n\n2\n1 2 \n" ] })
str_2 = json.dumps({ "inputs": [ "9\n1 10 7 1\n3 3 3 0\n8 2 10 4\n8 2 10 100\n-10 20 -17 2\n-3 2 2 0\n-3 1 2 0\n2 3 2 3\n-1 3 -2 2\n" ], "outputs": [ "7\n0\n4\n0\n30\n5\n4\n0\n3\n" ] })

pretend_dataset_list = [
    {"question": "question_1", "solutions": "solution_1", "input_outputs": str_1},
    {"question": "question_2", "solutions": "solution_2", "input_outputs": str_2}
]


class PretendDataset:
    def __init__(self, lst):
        self.lst = lst
        self.counter = 0
        self.max = len(self.lst)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.max:
            return self.lst[self.n]
            self.counter += 1
        else:
            raise StopIteration



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
        x = 5
        # assume program is the return of the lm_call
        program = """
        def exponentiate(x):
          return x**2
        """ 
        res = exec(program)
        self.assertEqual(res, 25)


    def test_run_apps_evals(self):
        ds_iterator = iter(PretendDataset(pretend_dataset_list))
        data = problem_unit_tests(ds_iterator)
        pass_rates = 0
        for program, solution, unit_test_inputs, unit_test_outputs in problem_unit_tests:
            pass_rate = run_apps_evals(program, unit_test_inputs, unit_test_outputs)
            pass_rates += pass_rate
        self.assertIsInstance(pass_rates, int)


    def test_problem_unit_tests(self):
        ds_iterator = iter(PretendDataset(pretend_dataset_list))
        data = problem_unit_tests(ds_iterator)

        for program, solution, unit_test_inputs, unit_test_outputs in problem_unit_tests:
            self.assertIsInstance(question, str)
            self.assertIsInstance(solution, str)
            self.assertIsInstance(inputs, list)
            self.assertIsInstance(outputs, list)
