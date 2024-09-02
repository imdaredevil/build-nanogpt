from model import GPT2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import tiktoken
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

encoder = tiktoken.encoding_for_model("gpt2")
config = GPT2Config()
model = GPT2(config)

BATCH_SIZE = 8
NUM_TOKENS = 16

TINY_SHAKESPEARE_PATH = "data/tiny_shakespeare.txt"
with open(TINY_SHAKESPEARE_PATH, "r") as f:
    shakespeare_dataset = f.read()

# running just a single batch
dataset_tokens = encoder.encode(shakespeare_dataset)
token_ids = dataset_tokens[:(BATCH_SIZE * NUM_TOKENS + 1)]  # [BATCH_SIZE * NUM_TOKENS]
token_ids = torch.tensor(token_ids, dtype=torch.long)
x = token_ids[:-1].view(BATCH_SIZE, NUM_TOKENS)
y = token_ids[1:].view(BATCH_SIZE, NUM_TOKENS)


# loss and optimizer
loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4)
optimizer.param_groups[0]["lr"] = 3e-4

# training loop
for _ in range(2):
    # forward pass
    logits = model(x)
    print(logits.shape)
    print(y.shape)
    loss_value = loss(logits.view(logits.shape[0] * logits.shape[1], logits.shape[-1]), y.view(y.shape[-1] * y.shape[-2]))
    print(loss_value.item()) 
    # backward pass
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()