# ai_planning_searching


"On Decoding", or, how to learn the model $$ \log p(y | x ) = \sum_{j=0}^{m-1} \log p(y_{j+1} | y_{\le_{j}}, x)$ and estimate $y^{opt} = \arg\max_{y} p(y|x) $$, using a transformer architecture. (Note that the sum is over terms of the form $\log p(y_{j+1} | y_{\le_{j}}, x)$ as this is the meaning of being autoregressive--to generative the locally next token, we condition on all previously generated tokens.) \\
This estimation is exponential in the length of the decoded string $y$. (To be precise it is $y^{|V|}$ where $V$ is the vocabulary size).\\
The following methods reduce the search space of large language model inference while approximating the optimal $y^{opt}$.\\
