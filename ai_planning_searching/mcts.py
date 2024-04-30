import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed

class Node:
	token: str # we assume that the token is the concatenation of all generated on this path
	value: float
	Q_s_a: dict[str,float]
	P_UCB_s_a: dict[str, float]
	visits: int
	children: list[Node]

	def __init__(self, token=token, value=0, visits=0):
		self.token = token
		self.value = value
		self.visits = visits
		self.Q_s_a = {}
		self.P_UCB_s_a = {}
		self.children = []


def select(root:Node) -> Node:
	while len(root.children) > 0:
		UCB_values = [P_UCB_s_a[child.str] for child in root.children]
		arg_max, max_UCB = max(list(enumerate(UCB_values)), key=lambda x: x) 
		root = root.children[arg_max]
	return root


def expand(root:Node, model, k, max_beam_len):
	# get top k
	beam_output = model.generate(
    root.tokens,
    max_new_tokens=40,
    num_beams=k,
    early_stopping=True)

    beams = [tokenizer.decode(beam_k, skip_special_tokens=True) for beam_k in beam_output]
    ## roll this out
    beam_list = [root.tokens + beam for beam in beams]
    for _ in range(max_beam_len-1):
    	new_list = []
    	for current_path in beam_list:
    		beam_output = model.generate(
    		current_path,
    		max_new_tokens=40,
    		num_beams=k,
    		early_stopping=True)

    		beams = [tokenizer.decode(beam_k, skip_special_tokens=True) for beam_k in beam_output]
    		new_list.extend([current_path + beam for beam in beams])
    	beam_list = new_list
    return beam_list # should be list of k^(max_beam_len) full paths






"""
generator = pipeline('text-generation', model='gpt2')
response = generator(prompt, max_length=400)
print(response[0]['generated_text'])
"""

"""
8

Each time you are at a node, you expand a node according to these rules :
if a child node has never been expanded before, then expand one of the 
unexplored child at random (and you can immediately unwind from this child
 node)
otherwise, each child node has been visited at least once. Compute 
for all of them the "exploration/exploitation" value and expand the 
child node with highest value
"""
