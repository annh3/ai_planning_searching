"""
Note: to train on GPU, you can call to('cuda') on the model
and inputs as usual
"""

import numpy as np
import torch
from torch.nn import functional as F
import transformers
import pdb
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup, AdamW, GPT2LMHeadModel, GPT2TokenizerFast


def get_dataset():
    train_dataset = load_dataset("rotten_tomatoes", split="train")
    eval_dataset = load_dataset("rotten_tomatoes", split="validation")


pretrained_weights = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(pretrained_weights)
model = GPT2TokenizerFast.from_pretrained(pretrained_weights)
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

"""
in model.generate you need to set return_dict_in_generate=False
to return torch.LongTensor
"""
def sampling_loop(model, input_ids, attention_mask, num_decode_steps=10):
    output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=num_decode_steps)


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

# get train and eval datasets
train_dataset, eval_dataset = get_dataset()

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

