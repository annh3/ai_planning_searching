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

I also downsampled the the dataset since I'm training on CPU, and also since this is a proof of concept.

<img width="257" alt="Screenshot 2024-09-10 at 11 13 35 AM" src="https://github.com/user-attachments/assets/c4dd83ba-93c9-40b5-838f-98b2b33467c9">

I used basically the default settings which I copied from a Hugging Face tutorial on how to use the Hugging Face Trainer module. I guess the hyper-parameter I was mostly interested in this time around was the learning rate. I'm not sure why, I guess it's because I'm starting to get more interested in the dynamics of deep learning optimization. 

The default learning rate scheduler in the Hugging Face Trainer module changes at a linear rate. You can observe the effect of this on the evaluation loss, which shows an accelerating decay.

<img width="106" alt="Screenshot 2024-09-10 at 11 41 34 AM" src="https://github.com/user-attachments/assets/795614e5-6e39-461e-b07d-e00f3da3feab">

<img width="645" alt="Screenshot 2024-09-10 at 11 41 17 AM" src="https://github.com/user-attachments/assets/9be7d155-686d-4c8d-866b-6e724a73e021">

If I were to actually do this in practice, I would first sweep over constant learning rates, through a grid search and through a random search, to find a good learning rate. I would also be able to train on the entire dataset (a much larger dataset than the Taylor Swift dataset) and for enough steps / epochs to observe convergence in the validation loss. After finding an initial constant learning rate that is good, i.e. achieves optimal loss, i.e. close to what is acheivable factoring in [irreducible noise](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff#:~:text=The%20bias%E2%80%93variance%20decomposition%20is,noise%20in%20the%20problem%20itself.). I would try and accelerate the learning rate linearly or through some other schedule to optimize the amount of total compute used in training.


