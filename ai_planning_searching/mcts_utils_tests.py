import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed
import unittest
from mcts_utils import Node, logits_to_token_strings
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

        self.mcts_root_node = node_0


    ####
    # TODO: for backpropagate statistics
    ####

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

        self.mcts_root_node = node_0



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
        create_mock_tree()

        _, max_rollout_reward, path_nodes = test_select(node_0)
        self.assertEqual(max_rollout_reward, 5)
        self.assertEqual(path_nodes, ['0', '3', '4'])

    def test_expand(self):

        beam_width = 3
        max_beam_len = 5

        beam_list = expand(self.mcts_root_node, self.tokenizer, self.model, beam_width, max_bean_len)
        self.assertEqual(len(beam_list),max_beam_len**beam_width)

    def test_evaluate_full_paths(self):
        pass 

    def test_backpropagate_statistics(self):
        c = 0
        c_base = 1

        pass

    def test_main_algorthim(self):
        pass

