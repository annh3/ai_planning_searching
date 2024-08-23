import datasets
from datasets import load_dataset
import json
import pdb



def run():
	print("hello")
	ds = load_dataset("codeparrot/apps", "all", split=test)
	"""
	sample = next(iter(ds))
	# non-empty solutions and input_output features can be parsed from text format this way:
	sample["solutions"] = json.loads(sample["solutions"])
	sample["input_output"] = json.loads(sample["input_output"])
	print(sample)
	pdb.set_trace()
	"""


if __name__ == "__main__":
	run()
