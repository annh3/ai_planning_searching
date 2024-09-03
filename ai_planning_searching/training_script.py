import numpy as np
import torch
from torch.nn import functional as F
import transformers
import pdb
from transformers import get_linear_schedule_with_warmup, AdamW, GPT2LMHeadModel, GPT2TokenizerFast



pretrained_weights = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(pretrained_weights)
# models are usually loaded in eval() mode, so set this to train()
model.train()
# initialize the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
# using weight decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)


"""
Now set up a simple training batch using tokenizer's __call__()

It returns a BatchEncoding() instance which prepares everything 
needed to be passed to a model.

Note that this is where you fetch the attention_mask.
"""

tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)

text_batch = ["I am reading Anna Karenina", "I am also reading Hackers and Painters"]

encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

"""
TODO: write a sampling loop for the model 

For now, we assume the output of the sampling loop is just
the next token, not the next t tokens.
"""

def sampling_loop(model, input_ids, attention_mask):
	pass

"""
Note: to train on GPU, you can call to('cuda') on the model
and inputs as usual
"""

outputs = sampling_loop(model, input_ids, attention_mask)

labels = torch.tensor([1,0]).unsqueeze(0) # this should be a 1-hot vector of vocabulary |V| size.
loss = F.cross_entropy(labels, outputs.logitd)
loss.backward()
optimizer.step()

"""
For now, use a linear learning rate scheduler. 
The learning rate linearly decays to 0 by the end of training.

Can try an exponentially
Decaying learning rate scheduler as well later.

Just call scheduler.step() after optimizer.step()
"""

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

scheduler.step()


"""
Try using the hugging face Trainer()

https://huggingface.co/transformers/v3.3.1/training.html#pytorch
"""

