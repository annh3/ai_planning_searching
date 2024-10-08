import numpy as np
import torch
import collections
import json
import math
from transformers import pipeline, set_seed
import transformers
import pdb
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast



class Node:
    current_token: torch.Tensor
    string: str # string representation of current token
    Q_s_a: dict[str,float] # Q values of the node's children
    P_UCB_s_a: dict[str, float] # P-UCB values of the node's children
    P_s_a : dict[str, float] # probability of the node's children
    visits: int
    children: list['Node']

    def __init__(self, current_token, string, visits=0):
        self.current_token = current_token
        self.string = string
        self.visits = visits
        self.Q_s_a = dict()
        self.P_UCB_s_a = dict()
        self.P_s_a = dict()
        self.children = []

    def __str__(self):
        return f"""
        current_token: {self.current_token}
        string: {self.string}
        visits: {self.visits}
        Q_s_a: {self.Q_s_a}
        P_UCB_s_a: {self.P_UCB_s_a}
        P_s_a: {self.P_s_a}
        children: {self.children}
        """

def select(root:Node, node_dictionary: dict[str, Node]) -> tuple[Node, list[str]]:
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
        path_nodes: list[str] of node names which index into node_dictionary
                    [v_0,...,v_n] where v_0 is the MCTS root and v_n is the newest addition
                    to the MCTS tree
        counter: To keep track of the depth of the path in the current rollout
        path_strings: Node strings without the counter

    """
    counter = 0
    mcts_tree_root = root # save this
    path_nodes = []
    path_strings = []
    root_node_name = root.string + '_' + str(counter)
    path_nodes.append(root_node_name) # name of the current node
    node_dictionary[root_node_name] = mcts_tree_root
    path_strings.append(root.string)
    

    while len(root.children) > 0:
        counter += 1
        UCB_values = [root.P_UCB_s_a[child.string] for child in root.children]
        arg_max, max_UCB = max(list(enumerate(UCB_values)), key=lambda x: x) 
        root = root.children[arg_max]
        node_dictionary[root.string + '_' + str(counter)] = root
        path_nodes.append(root.string + '_' + str(counter))
        path_strings.append(root.string)

    
    return root, path_nodes, counter, path_strings


    
def logits_to_token_strings(logits, tokenizer):
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

    beam_item = (first_transition_score, sequence_score, partial_tokens, partial_program)
    ...       = (torch.Tensor([0.18]), torch.Tensor([0.5]), ['hello', 'world', ..., '!'], [torch.Tensor([1]),...,torch.Tensor([12])])
    """
    # get top k - one of these will become the best action, and we will need to keep track of that token
    current_tokens = root.current_token.unsqueeze(0)
    beam_output = model.generate(
    current_tokens,
    max_new_tokens=1,
    num_beams=k,
    num_return_sequences=k,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
    early_stopping=True,
    output_logits=True)

    transition_scores = list(torch.chunk(beam_output.scores[0],chunks=k,dim=0))
    transition_scores = [t[-1] for t in transition_scores]
    # keep the score of this first expand only

    # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
    # sequences_scores are the sum of the log probabilities of the generated tokens
    # in the sequence, i.e. log p(y1y2...yk), which decomposes sum-wise

    # Note: these are the candidates for v_n
    next_tokens, str_repr = logits_to_token_strings(beam_output.logits[0], tokenizer)

    sequence_scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))


    beam_list = [(t,a,[b],[c]) for t,a,b,c in zip(transition_scores, sequence_scores, next_tokens, str_repr)]

    for _ in range(max_beam_len-1):
        new_list = []
        for current_path in beam_list:
            """
            https://huggingface.co/docs/transformers/v4.43.3/en/main_classes/text_generation#transformers.GenerationMixin.generate

            Q: what is the output of model.generate?
            
            It returns ModelOutput, i.e. GenerateBeamEncoderDecoderOutput
            https://huggingface.co/docs/transformers/v4.43.3/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput

            """
            current_tokens = torch.cat(current_path[2])
            current_tokens = current_tokens.unsqueeze(0)
            beam_output = model.generate(
            current_tokens,
            max_new_tokens=1,
            num_beams=k,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            early_stopping=True)

            next_tokens, str_repr = logits_to_token_strings(beam_output.logits[0], tokenizer)

            sequence_scores = list(torch.chunk(beam_output.sequences_scores,chunks=k,dim=0))
            for score,string,next_token in zip(sequence_scores,str_repr,next_tokens):
                cur = (current_path[0],score,current_path[2]+[next_token],current_path[3]+[string])
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
    res = sorted(beam_list,key=lambda x: x[1], reverse=True)[0]
    # returns score as a torch.Tensor float, the first string representing
    # the next action from MCTS search, and the program as a list of torch.Tensor tokens
    # pdb.set_trace()
    # max_rollout_reward, top_action, top_program, top_action_proba, top_action_token
    # beam_list = [(t,a,[b],[c]) for t,a,b,c in zip(transition_scores, sequence_scores, next_tokens, str_repr)]
    # (current_path[0],score,current_path[2]+[next_token],current_path[3]+[string])
    return res[1], res[3][0], res[2], res[0], res[2][0]


def evaluate_full_paths_with_reward_fn(beam_list, reward_function, model, tokenizer): 
    # returns full decoded path and its score
    # consider using a different reward model
    # suggested in https://openreview.net/pdf?id=Lr8cOOtYbfL to use max(score)

    """
    Args:
        beams_list: list of candidate beams
        reward_function

    Returns:
        max_rollout_reward: the reward from the full sequence
        top_action: top next action as a string, i.e. v_n.str, the first string in the list
        top_program: the entire sequence representing the full program
    """
    pass


# v_n is the newest node, we need to skip q_a as v_n does not have a child node
def backpropagate_statistics(path_nodes, path_strings, max_rollout_reward, c_base, c, node_dictionary):
    # 1. add 1 to all state visit counts
    # 2. recalculate P_UCB_s_a recursively (backwards)
    # 3. update Q_s_a with max_rollout_reward
    # According to the paper, this is
    # Q(s'',a'') <-- max(Q(s'',a''),r)
    # P(a|s) given by last log softmax in the return value
    
    # We need to update path_nodes in reverse order
    reversed_path_nodes = list(reversed(path_nodes))
    list_len = len(reversed_path_nodes)
    path_strings = list(reversed(path_strings))

    for i, node_name in enumerate(reversed_path_nodes):
        if i == 0:
            node = node_dictionary[node_name]
            node.visits += 1
            continue
        # pdb.set_trace()
        node = node_dictionary[node_name]
        node.visits += 1
        # Q(s'',a'') <-- max(Q(s'',a''),r)
        if len(node.Q_s_a) == 0:
            # create an entry, how do you know what the next action is?
            # it's path_strings[i-1]
            node.Q_s_a[path_strings[i-1]] = max_rollout_reward
            s_prime_visits = node_dictionary[reversed_path_nodes[i-1]].visits 
            node.P_UCB_s_a[path_strings[i-1]] = max_rollout_reward + node.P_s_a[path_strings[i-1]] * math.sqrt(math.log(node.visits)) / (1 + s_prime_visits)
        else: 
            new_q_s_a = {k: max(v, max_rollout_reward) for k,v in node.Q_s_a.items()}
            node.Q_s_a = new_q_s_a
            # Recalculated P_UCB_s_a
            beta = math.log((node.visits + c_base + 1) / c_base) + c
            s_prime_visits = node_dictionary[reversed_path_nodes[i-1]].visits 
            for i, (k, v) in enumerate(new_q_s_a.items()):
                node.P_UCB_s_a[k] = v + beta * node.P_s_a[k] * math.sqrt(math.log(node.visits)) / (1 + s_prime_visits)



def main_algorithm(prompt, max_rollouts, k, max_beam_len, c_base=1, c=0.5) -> str:

    pretrained_weights = 'gpt2'
    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
    model = GPT2LMHeadModel.from_pretrained(pretrained_weights)
    program_dictionary = dict() # to store fully generated programs
    
    # convert the prompt to tokens
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens = torch.Tensor(prompt_tokens)
    prompt_tokens = prompt_tokens.type(torch.LongTensor)
    root_node = Node(current_token=prompt_tokens,string=prompt)

    for rollout_i in range(max_rollouts):
        # print("rollout_i: ", rollout_i)
        node_dictionary = dict()
        node_to_expand, path_nodes, counter, path_strings = select(root_node, node_dictionary)
        beams_list = expand(node_to_expand, tokenizer, model, k, max_beam_len)
        max_rollout_reward, top_action, top_program, top_action_proba, top_action_token = evaluate_full_paths(beams_list)
        top_program_tensor = torch.cat(top_program)
        program_dictionary[top_program_tensor] = max_rollout_reward
        counter += 1
        new_node_name = top_action + '_' + str(counter)
        new_node = Node(current_token=top_action_token,string=top_action)
        # print("top action string: ", top_action)
        # print("top action token: ", top_action_token)
        # add the best action to path_nodes
        path_nodes.append(new_node_name)
        path_strings.append(top_action)
        node_dictionary[new_node_name] = new_node
        # add the child node to the previous node
        node_to_expand.P_s_a[top_action] = top_action_proba
        node_to_expand.children.append(new_node)
        # print("path_nodes: ", path_nodes)
        # print("path_strings: ", path_strings)
        # print("node_dictionary keys: ", node_dictionary.keys())
        backpropagate_statistics(path_nodes, path_strings, max_rollout_reward,  c_base, c, node_dictionary) #todo
        #print("\n\n")

    v = list(program_dictionary.values())
    k = list(program_dictionary.keys())
    return k[v.index(max(v))], root_node # return the program with the max rollout reward, and the mcts root








