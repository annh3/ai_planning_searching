"""
This file loads the APPS dataset.

Dataset Difficulty

"introductory"
"interview"
"competition"

Problem Format

1. Call-Based Format
	* Starter code
	* A special prompt
2. Standard Input Format
	* The model is provided with the problem and must output its answers to the 
	STDOUT stream, such as by using print statements

Test Case Quality
	* Average number of test cases is 21.2 per problem, some problems 
	only have two test cases

Metrics
	* Average number of test cases passed

"""
from datasets import load_dataset
import json


def problem_unit_tests(ds):
	"""
	Wraps around dataset

	Returns question, solution, unit_test_questions, unit_test_sols
	"""
	return (example["question"], example["solution"], json.loads(example["input_output"])["inputs"], json.loads(example["input_output"])["outputs"]for example in ds)


def load_apps(split: str):
	ds = load_dataset("codeparrot/apps", split=split)
	return ds