### Training with Hugging Face

I started this project initially as an inference-only project. Why? Because at the current moment of writing this write-up, I have no GPUs nor do I have TPUs (although I may consider training on a colab with GPUs or requesting some cloud TPUs.) 

I found out that, actually, writing a training loop for large language model training using open source software is not too complicated. Who would've thought?

Since the intent of this exploration was to have a proof of concept of how to set up training for finetuning on the APPs dataset, all that mattered for the proof of concept dataset is that we chose a text dataset and trained it in an auto-regressive manner.

What does this mean?

This means that the input to 
