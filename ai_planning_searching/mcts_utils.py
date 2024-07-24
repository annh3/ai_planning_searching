import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
lm = transformers.AutoModelForCausalLM.from_pretrained('gpt2')

class Node:
	# todo(annhe): refactor this so that the node is just a single token
	current_token: torch.Tensor
	sequence_tokens: torch.Tensor # we assume that the token is the concatenation of all generated on this path
	string: str # string representation of current token
	sequence_string: str # string representation of current sequence
	Q_s_a: dict[str,float] # Q values of the node's children
	P_UCB_s_a: dict[str, float] # P-UCB values of the node's children
	P_s_a : torch.Tensor # keep the logits of the node's children
	visits: int
	children: list[Node]

	def __init__(self, current_token='', value=0, visits=0):
		self.current_token = token
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


def expand(root:Node, tokenizer, model, k, max_beam_len):
	"""
	Note that we can do a more fine-grained exploration of expand by interpolating
	between the expansion and the roll-out, both of which are done with top-k decoding / beam search.
	"""
	# get top k - one of these will become the best action, and we will need to keep track of that token
	beam_output = model.generate(
    root.tokens,
    max_new_tokens=1,
    num_beams=k,
    num_return_sequences=k,
    return_dict_in_generate=True,
    output_scores=True,
    early_stopping=True)

    # todo(annhe): Add scores of beams to node.P_s_a
    # https://discuss.huggingface.co/t/how-to-get-sequences-scores-from-scores-in-generate-method/6048
    # print(beam_output['scores'][0].size())
    root.P_s_a = beam_output.sequences_scores # size 

    # you have to do some chunking here
    # ???
    scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))
    beams = list(torch.chunk(beam_output.sequences,chunks=k,dim=0))
    beam_list = [(s,b) for s,b in zip(scores,beams)]

    for _ in range(max_beam_len-1):
    	new_list = []
    	for current_path in beam_list:
    		beam_output = model.generate(
    		current_path,
    		max_new_tokens=1,
    		num_beams=k,
    		num_return_sequences=k,
    		return_dict_in_generate=True,
    		output_scores=True,
    		early_stopping=True)
    		current_beams = list(torch.chunk(beam_output.sequences,chunks=k,dim=0))
    		scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))
    		current_beam_list = [(s,b) for s,b in zip(scores,current_beams)]

    		new_list.extend(current_beam_list)

    	beam_list = new_list
    # should be list of k^(max_beam_len) full paths
    return beam_list 


def evaluate_full_paths(beam_list: list[tuple[torch.Tensor, torch.Tensor]]): 
	# returns full decoded path and its score
	# consider using a different reward model
	# suggested in https://openreview.net/pdf?id=Lr8cOOtYbfL to use max(score)
	res = sorted(beam_list,key=lambda x: x[1], reverse=True)[0]
	return res
	# todo, instead of likelihood, the reward should be from unit tests
	# todo, also return the current_token



def backpropagate_statistics(path_nodes, max_rollout_reward, c_base, c):
	# 1. add 1 to all state visit counts
	# 2. recalculate P_UCB_s_a recursively (backwards)
	# 3. update Q_s_a with max_rollout_reward
	# According to the paper, this is
	# Q(s'',a'') <-- max(Q(s'',a''),r)
	# P(a|s) given by last log softmax in the return value
	
	# We need to update path_nodes in reverse order
	reversed_path_nodes = list(reversed(path_nodes))
	for i, node in enumerate(reversed_path_nodes):
		node.visits += 1
		# Q(s'',a'') <-- max(Q(s'',a''),r)
		new_q_s_a = {k: max(v, max_rollout_reward) for k,v in node.Q_s_a.items()}
		node.Q_s_a = new_q_s_a
		# Recalculated P_UCB_s_a
		beta = math.log((node.visits + c_base + 1) / c_base) + c
		if i > 0:
			s_prime_visits = reversed_path_nodes[i-1].visits 
		else:
			s_prime_visits = 0
		for i, (k, v) in enumerate(new_q_s_a.items()):
    		print(i, k, v)
    		node.P_UCB_s_a[k] = v + beta * node.P_s_a[i] * math.sqrt(torch.log(node.visits)) / (1 + s_prime_visits)


def main_algorithm(prompt, max_rollouts) -> str:
	program_dictionary = dict() # to store fully generated programs
	# program_ditionary[program] = rollout_reward
	root_node = Node(prompt)

	for _ in max_rollouts:
		root_vo, path_nodes = select(node)
		beams_list = expand(root_vo)
		max_rollout_reward, top_action, top_program = evaluate_full_paths(beams_list)
		### TODO ###
		# create a new node with top_action
		############
		program_dictionary[top_program] = max_rollout_reward
		root_vo.children.append(Node(top_action,...)) # todo
		backpropagate_statistics(path_nodes, reward,...) #todo

	v = list(program_dictionary.values())
	k = list(program_dictionary.keys())
	return k[v.index(max(v))] # return the program with the max rollout reward








