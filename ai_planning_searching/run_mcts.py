import json
import pdb
from mlcroissant import Dataset



def run():
	print("hello")
	ds = Dataset(jsonld="https://huggingface.co/api/datasets/codeparrot/apps/croissant")
	records = ds.records(record_set="all")
	for i, record in enumerate(records):
		print(record)
		if i > 10:
			break


if __name__ == "__main__":
	run()
