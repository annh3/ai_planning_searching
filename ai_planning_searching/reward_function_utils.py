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
	* pass_rate: Average number of test cases passed

"""
from datasets import load_dataset
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast # TODO(annhe): is there a generic form of this?


def problem_unit_tests(ds):
	"""
	Wraps around dataset

	Returns question, solution, unit_test_questions, unit_test_sols
	"""
	return ((example["question"], example["solutions"], json.loads(example["input_output"])["inputs"], json.loads(example["input_output"])["outputs"]) for example in ds)



def run_apps_evals(candidate_program, unit_test_inputs, unit_test_outputs, model, tokenizer):
	"""
	Args:
		candidate_program: python program in the form of a string
		unit_test_inputs: list[str] of inputs
		unit_test_outputs: list[str] of outputs


	Returns:
		pass_rate: Average number of test cases passed
	"""
	tests_passed = 0
	total_tests = 0

	reward_call = f"""
	lm.generate(
    {tokens}['input_ids'],
    max_new_tokens=1,
    output_logits=True,
    output_scores=True,
    early_stopping=True)
	"""

	for question, _, unit_test_questions, unit_test_solutions in problem_unit_tests:
		for x,y in zip(unit_test_questions, unit_test_solutions):
			# generate an f-string for execution
			tokens = x
			fn_output = exec(reward_call)
			if y == fn_output:
				tests_passed += 1
			total_tests += 1
	return float(tests_passed) / total_tests
			


def load_apps(split: str):
	ds = load_dataset("codeparrot/apps", split=split)
	return ds











