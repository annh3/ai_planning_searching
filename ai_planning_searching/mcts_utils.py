import numpy as np
import torch
import collections
import json
import math
from transformers import pipeline, set_seed
import transformers
import pdb

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
lm = transformers.AutoModelForCausalLM.from_pretrained('gpt2')

"""
Today, July 26, I want to get the structure of this code right (DONE)

Then set up skeleton functions in the _test.py file to test it (DONE)

Then the next step is to correctly write backpropagate_statistics

Then, the next steps _should_ be to test it in the jupyter 
notebook, and then to write the real unit tests (which is lower
priority than testing the quality of decoding... 

Which you can at that point brainstorm how to do.
"""

class Node:
    # todo(annhe): refactor this so that the node is just a single token
    current_token: torch.Tensor
    string: str # string representation of current token
    Q_s_a: dict[str,float] # Q values of the node's children
    P_UCB_s_a: dict[str, float] # P-UCB values of the node's children
    P_s_a : torch.Tensor # keep the logits of the node's children
    visits: int
    children: list['Node']

    def __init__(self, current_token, string, visits=0):
        self.current_token = current_token
        self.string = string
        self.visits = visits
        self.Q_s_a = dict()
        self.P_UCB_s_a = dict()
        self.P_s_a = None
        self.children = []


def select(root:Node, node_dictionary) -> tuple[list[torch.Tensor], float, list[str]]:
    """
    Given the root node v_o of the MCTS tree,
    using P-UCB as a criterion, recursively
    traverse the tree to find a node that has not been
    previously expanded (a node without children nodes).


    Args: 
        root: the root node of the MCTS tree
        node_dictionary: a dict[str, Node]

    Returns: 
        top_program: list[torch.Tensor] of tokens representing the program from expand with the
            maximum rollout reward
        max_rollout_reward: float or torch.Tensor with dtype float?
        path_nodes: list[str] of node names which index into node_dictionary

    (todo): I'm thinking that instead of the node, which is copied,
    we should return node.str, which allows us to traverse the path of the 
    real MCTS tree's data structure in backpropagate_statistics, since Q_s_a 
    is indexed by string
    """
    mcts_tree_root = root # save this
    path_nodes = []
    counter = 0
    root_node_name = root.str + '_' + str(counter)
    path_nodes.append(root_node_name) # name of the current node
    

    while len(root.children) > 0:
        counter += 1
        UCB_values = [P_UCB_s_a[child.str] for child in root.children]
        arg_max, max_UCB = max(list(enumerate(UCB_values)), key=lambda x: x) 
        root = root.children[arg_max]
        path_nodes.append(root.str + '_' + str(counter))

    # return root, path_nodes
    """
    call expand and evaluate
    """
    # root now points to the root v_o to be expanded
    beams_list = expand(root)
    max_rollout_reward, top_action, top_program = evaluate_full_paths(beams_list)

    # add the new best action to the Tree
    # we need to name this node and add it to the tree, dictionary, and the path
    # make sure you add this to the previous node's children list
    new_node_name = top_action + '_' + str(counter)
    new_node = Node(string=top_action)
    # add the best action to path_nodes
    path_nodes.append(new_node_name)
    node_dictionary[new_node_name] = new_node

    return top_program, max_rollout_reward, path_nodes


    
def logits_to_token_strings(logits):
    """
    Helper function, computes tokens and str representations given logit representation.
    Intended to be used as part of beam decoding, for the next beam_width
    tokens.

    Args:
        logits: (beam_size, vocab_size)
    Returns:
        next_tokens: list[Torch.tensor] of tokens of length beam_size
        str_repr: a list[str] of length beam_size
    """
    k = logits.shape[0]
    next_tokens = torch.log_softmax(logits, dim=1)
    next_tokens = torch.argmax(next_tokens, dim=1)
    str_repr = tokenizer.batch_decode(next_tokens)
    next_tokens = list(torch.chunk(next_tokens,chunks=k,dim=0))
    return next_tokens, str_repr


def expand(root:Node, tokenizer, model, k, max_beam_len):
    """
    Each item of beam is a tuple like

    beam_item = (sequence_score, partial_tokens, partial_program)
    ...       = (torch.Tensor([0.5]), ['hello', 'world', ..., '!'], [torch.Tensor([1]),...,torch.Tensor([12])])
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


    beams_list = [(a,b,c) for a,b,c in zip(scores, next_tokens, str_repr)]

    for _ in range(max_beam_len-1):
        new_list = []
        for current_path in beam_list:
            """
            https://huggingface.co/docs/transformers/v4.43.3/en/main_classes/text_generation#transformers.GenerationMixin.generate

            Q: what is the output of model.generate?
            
            It returns ModelOutput, i.e. GenerateBeamEncoderDecoderOutput
            https://huggingface.co/docs/transformers/v4.43.3/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput

            """
            current_tokens = torch.cat(current_path[1])
            beam_output = model.generate(
            current_tokens,
            max_new_tokens=1,
            num_beams=k,
            num_return_sequences=k,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            early_stopping=True)

            next_tokens, str_repr = logits_to_token_strings(beam_output.logits[0])

            scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))
            for score,string,next_token in zip(scores,str_repr,next_tokens):
                # add to beams
                cur = (score,current_path[2]+[string],current_path[1]+[next_token])
                new_list.append(cur)

        beam_list = new_list
    # should be list of k^(max_beam_len) full paths
    return beam_list 


def evaluate_full_paths(beam_list): 
    # returns full decoded path and its score
    # consider using a different reward model
    # suggested in https://openreview.net/pdf?id=Lr8cOOtYbfL to use max(score)

    """
    Args:
        beams_list: list of candidate beams

    Returns:
        max_rollout_reward: the reward from the full sequence
        top_action: top next action as a string, i.e. v_n.str, the first string in the list
        top_program: the entire sequence representing the full program
    """
    res = sorted(beam_list,key=lambda x: x[0], reverse=True)[0]
    # returns score as a torch.Tensor float, the first string representing
    # the next action from MCTS search, and the program as a list of torch.Tensor tokens
    return res[0], res[2][0], res[1]


def backpropagate_statistics(path_nodes, max_rollout_reward, c_base, c, node_dictionary):
    # 1. add 1 to all state visit counts
    # 2. recalculate P_UCB_s_a recursively (backwards)
    # 3. update Q_s_a with max_rollout_reward
    # According to the paper, this is
    # Q(s'',a'') <-- max(Q(s'',a''),r)
    # P(a|s) given by last log softmax in the return value
    
    # We need to update path_nodes in reverse order
    reversed_path_nodes = list(reversed(path_nodes))

    # we need to keep a dictionary from node string names to Nodes
    for i, node_name in enumerate(reversed_path_nodes):
        node = node_dictionary[node_name]
        node.visits += 1
        # Q(s'',a'') <-- max(Q(s'',a''),r)
        if node.Q_s_a is not None: # assume node.Q_s_a is none for the newest node v_n
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

"""
TODO(annhe): write tests for this
"""
def main_algorithm(prompt, max_rollouts) -> str:
    program_dictionary = dict() # to store fully generated programs
    # program_ditionary[program] = rollout_reward
    
    # TODO(annhe)
    root_node = Node(prompt)

    # Create a dictionary mapping node names to nodes
    # Note that in this case we'll have to append the depth to the token
    # string to avoid hash collision

    node_dictionary = dict()
    root_node_label = prompt + '_0'
    node_dictionary[root_node_label] = root_node

    for _ in max_rollouts:
        top_program, max_rollout_reward, path_nodes = select(node, node_dictionary)
        program_dictionary[top_program] = max_rollout_reward
        backpropagate_statistics(path_nodes, max_rollout_reward) #todo

    v = list(program_dictionary.values())
    k = list(program_dictionary.keys())
    return k[v.index(max(v))] # return the program with the max rollout reward








