import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed

class Node:
	token: str
	value: float
	Q_s_a: dict[str,float]
	P_UCB_s_a: dict[str, float]
	visits: int

	def __init__(self, token=token, value=0, visits=0):
		self.token = token
		self.value = value
		self.visits = visits
		self.Q_s_a = {}
		self.P_UCB_s_a = {}

"""
generator = pipeline('text-generation', model='gpt2')
response = generator(prompt, max_length=400)
print(response[0]['generated_text'])
"""