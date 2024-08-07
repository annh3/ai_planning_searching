"""
python -m unittest mcts_utils_tests.py
"""
import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed
import unittest
from mcts_utils import Node, logits_to_token_strings, evaluate_full_paths, expand, main_algorithm, backpropagate_statistics, select
from transformers import GPT2LMHeadModel, GPT2TokenizerFast



class testMCTSUtils(unittest.TestCase):

    def create_mock_tree(self):
        node_0 = Node(current_token=torch.Tensor([0]), string='0')
        node_0.P_UCB_s_a['1'] = 0
        node_0.P_UCB_s_a['3'] = 0

        node_1 = Node(current_token=torch.Tensor([1]), string='1')
        node_1.P_UCB_s_a['2'] = 0

        node_2 = Node(current_token=torch.Tensor([2]), string='2')

        node_3 = Node(current_token=torch.Tensor([3]), string='3')
        node_3.P_UCB_s_a['4'] = 5
        node_4 = Node(current_token=torch.Tensor([4]), string='4')

        node_0.children = [node_1, node_3]
        node_1.children = [node_2]
        node_3.children = [node_4]
        # are these going to be persistent on the stack?
        # I think you're going to need to write a different function
        # to instantiate them

        self.mcts_root_node = node_0


    ####
    # TODO: for backpropagate statistics
    ####

    def create_mock_tree_2(self):
        # lets just instantiate all nodes with the uniform distribution for now
        beam_width = 5
        node_dictionary = dict()


        node_0 = Node(current_token=torch.Tensor([0]), string='0')
        node_0.P_UCB_s_a['1'] = 0
        node_0.P_UCB_s_a['3'] = 0
        node_0.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_0.P_s_a = node_0.P_s_a / beam_width
        node_dictionary['0'] = node_0

        node_1 = Node(current_token=torch.Tensor([1]), string='1')
        node_1.P_UCB_s_a['2'] = 0
        node_1.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_1.P_s_a = node_1.P_s_a / beam_width
        node_dictionary['1'] = node_1

        node_2 = Node(current_token=torch.Tensor([2]), string='2')
        node_2.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_2.P_s_a = node_2.P_s_a / beam_width
        node_dictionary['2'] = node_2


        node_3 = Node(current_token=torch.Tensor([3]), string='3')
        node_3.P_UCB_s_a['4'] = 5
        node_3.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_3.P_s_a = node_3.P_s_a / beam_width
        node_dictionary['3'] = node_3

        node_4 = Node(current_token=torch.Tensor([4]), string='4')
        node_4.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_4.P_s_a = node_4.P_s_a / beam_width
        node_dictionary['4'] = node_4

        node_0.children = [node_1, node_3]
        node_1.children = [node_2]
        node_3.children = [node_4]

        self.mcts_root_node = node_0
        return node_dictionary



    def setUp(self):
        # setup a hugging face model here
        self.pretrained_weights = 'gpt2'
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.pretrained_weights)
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_weights)


    def tearDown(self):
        pass

    def testNode(self):
        # test that we can create a basic node
        node_token = torch.Tensor([2])
        string = 'hello'
        new_node = Node(node_token, string)
        new_node.visits = 1
        new_node.Q_s_a['another_node'] = 0.9
        new_node.P_UCB_s_a['another_node'] = 11
        new_node.P_s_a = torch.ones((5,))
        self.assertEqual(new_node.string, 'hello')
        self.assertEqual(new_node.current_token, torch.Tensor([2]))

    def test_logits_to_token_strings(self):
        # need to call model to get a beam output
        batch_size = 5
        path_length = 3
        current_path = torch.ones((5,3),dtype=torch.long)
        k = 2
        beam_output = self.model.generate(
            current_path,
            max_new_tokens=1,
            num_beams=k,
            num_return_sequences=k,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            early_stopping=True)

        next_tokens, str_repr = logits_to_token_strings(beam_output.logits[0])
        self.assertIsInstance(next_tokens[0], torch.Tensor)
        self.assertIsInstance(str_repr[0], str)


    def test_select(self):
        # create a mock tree
        #     [0], '0', 0
        #  [1], '1', 0      [3], '3', 0
        #  [2], '2', 0      [4], '4', 5
        #
        # [token], string, Q value of each node, which is implicit
        # Note here that node [4] should be selected
        self.create_mock_tree()

        _, max_rollout_reward, path_nodes = select(node_0)
        self.assertEqual(max_rollout_reward, 5)
        self.assertEqual(path_nodes, ['0', '3', '4'])

    def test_expand(self):

        beam_width = 3
        max_beam_len = 5

        node_0 = Node(current_token=torch.Tensor([0]), string='0')
        node_0.P_UCB_s_a['1'] = 0
        node_0.P_UCB_s_a['3'] = 0
        node_0.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_0.P_s_a = node_0.P_s_a / beam_width

        beam_list = expand(node_0, self.tokenizer, self.model, beam_width, max_beam_len)
        self.assertEqual(len(beam_list),max_beam_len**beam_width)

    def test_evaluate_full_paths(self):
        seq_len = 2
        beam_list = [(torch.Tensor([0.5]), [torch.Tensor([4]),torch.Tensor([6])] ,['hello', 'world']), (torch.Tensor([0.7]), [torch.Tensor([4]),torch.Tensor([11])] ,['hello', 'moon'])]
        max_rollout_reward, top_action, top_program = evaluate_full_paths(beam_list)
        self.assertIsInstance(top_action, str)
        # TODO(annhe): add more tests
        

    def test_backpropagate_statistics(self):
        c = 0
        c_base = 1
        path_nodes = ['0', '3', '4']
        path_strings = ['0', '3', '4']
        max_rollout_reward = 5
        #node_dictionary = self.create_mock_tree_2()

        #### Testing this instead ####
        beam_width = 5
        node_dictionary = dict()


        node_0 = Node(current_token=torch.Tensor([0]), string='0')
        node_0.P_UCB_s_a['1'] = 0
        node_0.P_UCB_s_a['3'] = 0
        node_0.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_0.P_s_a = node_0.P_s_a / beam_width
        node_dictionary['0'] = node_0

        node_1 = Node(current_token=torch.Tensor([1]), string='1')
        node_1.P_UCB_s_a['2'] = 0
        node_1.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_1.P_s_a = node_1.P_s_a / beam_width
        node_dictionary['1'] = node_1

        node_2 = Node(current_token=torch.Tensor([2]), string='2')
        node_2.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_2.P_s_a = node_2.P_s_a / beam_width
        node_dictionary['2'] = node_2


        node_3 = Node(current_token=torch.Tensor([3]), string='3')
        node_3.P_UCB_s_a['4'] = 5
        node_3.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_3.P_s_a = node_3.P_s_a / beam_width
        node_dictionary['3'] = node_3

        node_4 = Node(current_token=torch.Tensor([4]), string='4')
        node_4.P_s_a = torch.ones((beam_width,),dtype=torch.float)
        node_4.P_s_a = node_4.P_s_a / beam_width
        node_dictionary['4'] = node_4

        node_0.children = [node_1, node_3]
        node_1.children = [node_2]
        node_3.children = [node_4]


        ##############################




        backpropagate_statistics(path_nodes, path_strings, max_rollout_reward, c_base, c, node_dictionary)
        # test that all of the Q_s_a values of the nodes on the path are updated
        # with max_rollout_reward
        # Node 0 should have Q_s_a['3'] = max_rollout_reward
        # Node 3 should have Q_s_a['4'] = max_rollout_reward

        # print the nodes to debug
        #for k,v in node_dictionary.items():
        #    print(k)
        #    print(v)
        #    print('\n\n')

        self.assertEqual(node_dictionary['0'].Q_s_a['3'], max_rollout_reward)
        self.assertEqual(node_dictionary['3'].Q_s_a['4'], max_rollout_reward)



    def test_main_algorthim(self):
        prompt = "Hello my name is: "
        max_rollouts = 3
        # a simple test, check the return type
        program = main_algorithm(prompt, max_rollouts)
        self.assertIsIstance(program[0], torch.Tensor)


