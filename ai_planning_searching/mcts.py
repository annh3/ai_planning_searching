import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
lm = transformers.AutoModelForCausalLM.from_pretrained('gpt2')

class Node:
	token: str # we assume that the token is the concatenation of all generated on this path
	Q_s_a: dict[str,float]
	P_UCB_s_a: dict[str, float]
	P_s_a : torch.Tensor # keep the logits
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
	path_nodes = []
	path_nodes.append(root)
	while len(root.children) > 0:
		UCB_values = [P_UCB_s_a[child.str] for child in root.children]
		arg_max, max_UCB = max(list(enumerate(UCB_values)), key=lambda x: x) 
		root = root.children[arg_max]
		path_nodes.append(root)
	return root, path_nodes


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

    		# todo(annhe): Add scores to node.P_s_a
    		# https://discuss.huggingface.co/t/how-to-get-sequences-scores-from-scores-in-generate-method/6048
    		# print(beam_output['scores'][0].size())

    		beams = [tokenizer.decode(beam_k, skip_special_tokens=True) for beam_k in beam_output]
    		new_list.extend([current_path + beam for beam in beams])
    	beam_list = new_list
    return beam_list # should be list of k^(max_beam_len) full paths


def evaluate_full_paths(beam_list: list[str], model, eval_prompt): # consider using a different reward model
	scores = [model.score(beam) for beam in beam_list]
	# suggested in https://openreview.net/pdf?id=Lr8cOOtYbfL to use max(score)
	return max(score)


def backpropagate_statistics(path_nodes, max_rollout_reward, c_base, c):
	# 1. add 1 to all state visit counts
	# 2. recalculate P_UCB_s_a recursively (backwards)
	# 3. update Q_s_a with max_rollout_reward
	# According to the paper, this is
	# Q(s'',a'') <-- max(Q(s'',a''),r)
	# P(a|s) given by last log softmax in the return value
	
	# We need to update path_nodes in reverse order
	for node in reversed(path_nodes):
		node.visits += 1
		# Q(s'',a'') <-- max(Q(s'',a''),r)
		new_q_s_a = {k: max(v, max_rollout_reward) for k,v in node.Q_s_a.items()}
		# Recalculated P_UCB_s_a
		beta = math.log((node.visits + c_base + 1) / c_base) + c






	pass








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
