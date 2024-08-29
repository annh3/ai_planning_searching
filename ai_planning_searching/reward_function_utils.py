"""
TODO(annhe): 

PAUSE on the design of these utils as we have not yet decided
whether to try finetuning on the APPS dataset at all.

If we decide to finetune on the APPS dataset we will need to match
the formatting of that dataset.

O.W., can try to curate our own dataset.



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
	* pass_rate: Average number of test cases passed

"""
from datasets import load_dataset
import json
import pdb
from transformers import GPT2LMHeadModel, GPT2TokenizerFast # TODO(annhe): is there a generic form of this?


#TODO(annhe): rewrite this so that unit_test_questions, unit_test_sols
# is returned as list of integers, list of integers
def problem_unit_tests(ds):
	"""
	Wraps around dataset

	Returns question, solution, unit_test_questions, unit_test_sols
	"""
	return ((example["question"], example["solutions"], json.loads(example["input_output"])["inputs"], json.loads(example["input_output"])["outputs"]) for example in ds)


def run_single_test_case():
	pass


def run_apps_evals(candidate_program, data):
	"""
	Args:
		candidate_program: python program in the form of a string
		data: list[list[str], list[str]] of inputs and outputs, or similar iterable


	Returns:
		pass_rate: Average number of test cases passed
	"""
	tests_passed = 0
	total_tests = 0
	# pdb.set_trace()

	for _, _, unit_test_questions, unit_test_solutions in data:
		for input_value, gt_output_value in zip(unit_test_questions, unit_test_solutions):
			loc = {}
			exec(candidate_program, globals(), loc)
			fn_output = loc['output_value']
			if gt_output_value == fn_output:
				tests_passed += 1
			total_tests += 1
	return float(tests_passed) / total_tests
			


def load_apps(split: str):
	ds = load_dataset("codeparrot/apps", split=split)
	return ds











