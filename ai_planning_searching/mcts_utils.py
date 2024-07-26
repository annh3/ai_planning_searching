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
	"""
	Given the root node v_o of the MCTS tree,
	using P-UCB as a criterion, recursively
	traverse the tree to find a node that has not been
	previously expanded (a node without children nodes).


	Args: root, the root node of the MCTS tree
	Returns: 
		root: the newly added node to the tree, v_n
		path_nodes: the list of node.str in the tree which were traversed, starting
		from the root node v_o, i.e. [v_o.str, v_1.str,...,v_n.str]

	(todo): I'm thinking that instead of the node, which is copied,
	we should return node.str, which allows us to traverse the path of the 
	real MCTS tree's data structure in backpropagate_statistics, since Q_s_a 
	is indexed by string
	"""
	path_nodes = []
	path_nodes.append(root.str) # name of the current node
	while len(root.children) > 0:
		UCB_values = [P_UCB_s_a[child.str] for child in root.children]
		arg_max, max_UCB = max(list(enumerate(UCB_values)), key=lambda x: x) 
		root = root.children[arg_max]
		path_nodes.append(root.str)
	return root, path_nodes


def logits_to_token_strings(logits):
	"""
	Helper function, computes tokens and str representations given logit representation

	Args:
		logits: (batch_size, vocab_size)
	Returns:
		tokens: list[Torch.tensor] of tokens of length batch_size
		str: a list of length batch_size
	"""
	k = logits.shape[0]
	next_tokens = torch.log_softmax(beam_output.logits[0], dim=1)
    next_tokens = torch.argmax(next_tokens, dim=1)
    str_repr = tokenizer.batch_decode(next_tokens)
    next_tokens = list(torch.chunk(next_tokens,chunks=k,dim=0))
    return next_tokens, str_repr


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
    # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
    # sequences_scores are the sum of the log probabilities of the generated tokens
    # in the sequence, i.e. log p(y1y2...yk), which decomposes sum-wise
    root.P_s_a = beam_output.sequences_scores # size 

   	# Note: these are the candidates for v_n
   	next_tokens, str_repr = logits_to_token_strings(beam_output.logits[0])

   	# you'll have to add the same decoding code here
    scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))
    beams = list(torch.chunk(beam_output.sequences,chunks=k,dim=0))
    beam_list = [(s,b) for s,b in zip(scores,beams)]

    for _ in range(max_beam_len-1):
    	new_list = []
    	for current_path in beam_list:
    		"""
    		https://huggingface.co/docs/transformers/v4.43.3/en/main_classes/text_generation#transformers.GenerationMixin.generate

    		Q: what is the output of model.generate?
    		
    		It returns ModelOutput, i.e. GenerateBeamEncoderDecoderOutput
    		https://huggingface.co/docs/transformers/v4.43.3/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput

    		"""
    		beam_output = model.generate(
    		current_path,
    		max_new_tokens=1,
    		num_beams=k,
    		num_return_sequences=k,
    		return_dict_in_generate=True,
    		output_logits=True,
    		output_scores=True,
    		early_stopping=True)

    		next_tokens, str_repr = logits_to_token_strings(beam_output.logits[0])

    		current_beams = list(torch.chunk(beam_output.sequences,chunks=k,dim=0))
    		scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))
    		current_beam_list = [(s,b) for s,b in zip(scores,current_beams,str_repr,next_tokens)]

    		new_list.extend(current_beam_list)

    	beam_list = new_list
    # should be list of k^(max_beam_len) full paths
    return beam_list 


"""
I notice here that we return the full program.
We need to find the right truncation 

by convention of how the function is written, 
this is simply the first "node" in a "beam_list",
call it v_n, like in the diagram

But we also need all of the nodes on the path 
from the real root of the MCTS tree, v_o, which
contains the original prompt that was the input
when the MCTS algorithm was called



"""

def evaluate_full_paths(beam_list: list[tuple[torch.Tensor, torch.Tensor]]): 
	# returns full decoded path and its score
	# consider using a different reward model
	# suggested in https://openreview.net/pdf?id=Lr8cOOtYbfL to use max(score)

	"""
	Args:
		beams_list: list of candidate beams

	Returns:
		max_rollout_reward: the reward from the full sequence
		top_action: top next action as a string, i.e. v_n.str
		top_program: the entire sequence representing the full program
	"""
	res = sorted(beam_list,key=lambda x: x[1], reverse=True)[0]
	# scores,current_beams,str_repr,next_tokens
	program = [' '.join([r[2] for r in res])]
	return res[0], res[2], program



def backpropagate_statistics_V1(path_nodes, max_rollout_reward, c_base, c):
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
		# create a new node with top_action in the MCTS tree
		# i.e. will need to traverse the tree
		############
		program_dictionary[top_program] = max_rollout_reward
		root_vo.children.append(Node(top_action,...)) # todo
		backpropagate_statistics(path_nodes, reward,...) #todo

	v = list(program_dictionary.values())
	k = list(program_dictionary.keys())
	return k[v.index(max(v))] # return the program with the max rollout reward








