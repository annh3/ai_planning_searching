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


def load_apps(path:str, difficulty_level:str):
	"""
	Returns a pandas data frame with columns including
	'prompt' for the natural language and a column 'unit_test'
	for some delimiter separated string of the unit tests for that 
	coding question
	"""
	pass