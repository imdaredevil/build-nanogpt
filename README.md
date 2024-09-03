# My own version of building GPT2 from scratch. 

In this repo, I am reproducing the changes in karpathy/build-nanogpt. 

We have the following objectives.

- learning to train a NN from a paper
- learning the nuances of pytorch/parallel programming/GPU
- trying out the things mentioned as future works in the video.

## My own notes

- Start from a small dataset ( here shakespeare )
- Do not do anything blindly just because its best practice.
- Try it out yourself to see whether those methods make sense
- When we share weights, use the linear layer weights to initialize the weight. If you do it other way like below, the starting point of the optimization will be different. Specifically, the linear model uses Xavier initialization. But, the embedding layer uses normal initialization. Normal initialization is not favourable for linear layers. But, Xavier initialization work for embedding layers.
- To run with data parallel
```
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```
