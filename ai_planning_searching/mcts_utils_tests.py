import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed
import unittest
from mcts_utils import Node, logits_to_token_strings
from transformers import GPT2LMHeadModel, GPT2TokenizerFast



class testMCTSUtils(unittest.TestCase):

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
        self.assertIsInstance(next_tokens, torch.Tensor)
        self.assertIsInstance(str_repr, str)


    def test_select(self):
        pass

    def test_expand(self):
        pass

    def test_evaluate_full_paths(self):
        pass 

    def backpropagate_statistics(self):
        pass

    def test_main_algorthim(self):
        pass

