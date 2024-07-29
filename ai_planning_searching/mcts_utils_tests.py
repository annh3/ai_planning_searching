import numpy as np
import torch
import collections
import json
from transformers import pipeline, set_seed
import unittest
from mcts_utils import Node
from transformers import AutoModel



class testMCTSUtils(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		# setup a hugging face model here
		model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-gpt2")

	@classmethod
	def tearDownClass(cls):
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
		pass

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

