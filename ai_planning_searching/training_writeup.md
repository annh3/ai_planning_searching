### Training with Hugging Face

I started this project initially as an inference-only project. Why? Because at the current moment of writing this write-up, I have no GPUs nor do I have TPUs (although I may consider training on a colab with GPUs or requesting some cloud TPUs.) 

I found out that, actually, writing a training loop for large language model training using open source software is not too complicated. Who would've thought?

Since the intent of this exploration was to have a proof of concept of how to set up training for finetuning on the APPs dataset, all that mattered for the proof of concept dataset is that we chose a text dataset and trained it in an auto-regressive manner.

I chose the Taylor Swift dataset. Here's an example data point.

<img width="747" alt="Screenshot 2024-09-10 at 11 03 16 AM" src="https://github.com/user-attachments/assets/6adbd937-59c8-4d50-a280-ef765190223c">



What does this mean?

This means that the input to the LLM $f(x)$ is a string and the output, or target, y, of the LLM is the string shifted to the right by one index.

To accomodate this, I wrote a preprocessing function which does exactly that.


<img width="830" alt="Screenshot 2024-09-10 at 11 01 19 AM" src="https://github.com/user-attachments/assets/f7496f5e-01af-41ba-9d5d-b035b798fa2a">


