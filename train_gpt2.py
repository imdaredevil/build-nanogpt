from model import GPT2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import tiktoken
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import time
import os
import numpy as np

torch.manual_seed(42)

encoder = tiktoken.encoding_for_model("gpt2")
eot = encoder._special_tokens['<|endoftext|>']
config = GPT2Config(vocab_size=50304)
model = GPT2(config)
model = torch.compile(model)

EPOCHS = 1
BATCH_SIZE = 64
MINI_BATCH_SIZE = 16 # we use gradient accumulation here.
NUM_TOKENS = 1024
MAX_STEPS = 50
assert BATCH_SIZE % MINI_BATCH_SIZE == 0
GRAD_ACCUM_STEPS = BATCH_SIZE // MINI_BATCH_SIZE
print(f"grad accumulation steps: {GRAD_ACCUM_STEPS}")
# creating dataloader

class Dataloader:
    def __init__(self, folder_path, batch_size, num_tokens, pad_token, split = "train"):
        self.shards_path = os.path.join(folder_path, split)
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.shard_files = os.listdir(self.shards_path)
        self.pad_token = pad_token
        self.reset()


    def __iter__(self):
        return self

    def __next__(self):
        batch_token_size = self.batch_size * self.num_tokens + 1
        tokens = self.curr_shard_tokens[self.curr_index : self.curr_index + batch_token_size]

        if len(tokens) < batch_token_size: # need to fetch from next shard
            self.curr_shard_idx += 1
            if self.curr_shard_idx >= len(self.shard_files): # reached end of all shards
                if len(tokens) > 0: # return what we have after padding
                    tokens = np.pad(tokens, batch_token_size - len(tokens), mode="constant", constant_values=self.pad_token)
                    # move curr_idx to end
                    self.curr_index = len(self.curr_shard_tokens) # this will end the loop next time 
                else:
                    self.reset()
                    raise StopIteration 
            self.curr_shard = np.load(os.path.join(self.shards_path, f"{self.shard_files[self.curr_shard_idx]}"))
            tokens = np.concatenate(tokens, self.curr_shard[:(batch_token_size - len(tokens))], axis=0) 
            self.curr_index = batch_token_size - len(tokens) - 1
        else:
            self.curr_index += batch_token_size - 1
        x = tokens[:-1]
        y = tokens[1:]
        x = torch.tensor(x, dtype=torch.long).view(self.batch_size, self.num_tokens)
        y = torch.tensor(y, dtype=torch.long).view(self.batch_size, self.num_tokens)
        return x, y

    def reset(self):
        self.curr_index = 0
        self.curr_shard_idx = 0
        self.curr_shard_tokens = np.load(os.path.join(self.shards_path, f"{self.shard_files[self.curr_shard_idx]}"))
        self.remaining_tokens = None


# TINY_SHAKESPEARE_PATH = "data/tiny_shakespeare.txt"
# data_loader = Dataloader(TINY_SHAKESPEARE_PATH, MINI_BATCH_SIZE, NUM_TOKENS, encoder)
# print(f"Number of mini-batches: {len(data_loader)}")
# NUM_BATCHES = 50 # len(data_loader) // GRAD_ACCUM_STEPS
# print(f"Number of batches: {NUM_BATCHES}")

FINEWEB_PATH = "data/fineweb/"
NUM_TOKEN_TOTAL = int(1e10)
NUM_BATCHES = NUM_TOKEN_TOTAL // (BATCH_SIZE * NUM_TOKENS)
print(f"Number of steps: {NUM_BATCHES}")
data_loader = Dataloader(FINEWEB_PATH, MINI_BATCH_SIZE, NUM_TOKENS, eot)
NUM_BATCHES = 50


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
# device = "cpu"
print(f"using device: {device}")

model.to(device)

# loss and optimizer
loss = CrossEntropyLoss()
loss.to(device)
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
MAX_LR = 6e-4
WARMUP_STEPS = 24 # this we calculated from gpt 3 paper. There they use first 375 M token for a 300 B token dataset.
COSINE_DECAY_STEPS = 16530 # again in GPT3 they use first 260B tokens out of 300B dataset. 

MIN_LR = MAX_LR / 10
INIT_LR = MAX_LR / WARMUP_STEPS

def get_lr(curr_lr, step):
    if step < WARMUP_STEPS:
        return min(MAX_LR, curr_lr + MAX_LR / WARMUP_STEPS)
    elif step < COSINE_DECAY_STEPS:
        step_no = step - WARMUP_STEPS
        total_steps = COSINE_DECAY_STEPS - WARMUP_STEPS
        cos_value = np.cos((step_no / total_steps) * np.pi) * 0.5 + 0.5 # scaling cos to given value
        return MIN_LR + 0.9 * MAX_LR * cos_value
    else:
        return MIN_LR



# training loop
lr = 0
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    data_loader_iterator = iter(data_loader)
    for step in range(NUM_BATCHES):
        if step >= MAX_STEPS:
            break
        st = time.time()
        # forward pass
        loss_scalar = 0
        lr = get_lr(lr, step)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_loader_iterator)
            except StopIteration:
                print("looping back")
                pass  # end of iterator. looping back to start
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(x)
                loss_value = loss(logits.view(logits.shape[0] * logits.shape[1], logits.shape[-1]), y.view(y.shape[-1] * y.shape[-2]))
                loss_value /= GRAD_ACCUM_STEPS
            # backward pass
            loss_value.backward()
            loss_scalar += loss_value.item()

        # clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        time_taken = end - st
        print(f"Step: {step}, loss: {loss_scalar:.4f}, norm: {norm:.4f}, lr: {lr:.4e}, time taken: {time_taken * 1000:.4f} ms, tokens per second: {(BATCH_SIZE * NUM_TOKENS/time_taken):.4f}")

torch.save(model.state_dict(), "model.pt")