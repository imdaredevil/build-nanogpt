from model import GPT2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import tiktoken
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import time
import os
import numpy as np
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.nn.functional as F
import inspect
import matplotlib.pyplot as plt

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

ddp = int(os.environ.get('RANK', -1)) != -1 # check whether using ddp
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
else:
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    # device = "cpu"
is_main_process = (ddp_rank == 0)
if is_main_process:
    print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"


encoder = tiktoken.encoding_for_model("gpt2")
eot = encoder._special_tokens['<|endoftext|>']
config = GPT2Config(vocab_size=50304)
model = GPT2(config)
model.to(device)
if device != "mps":
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


EPOCHS = 1
BATCH_SIZE = 16
MINI_BATCH_SIZE = 4 # we use gradient accumulation here.
NUM_TOKENS = 128
MAX_STEPS = 50
VAL_STEPS = 20
VAL_FREQUENCY = 200
SAMPLE_FREQUENCY = 20
sample_start = "I am a large language model. "
sample_start_tokens = encoder.encode(sample_start)
sample_start_tokens = np.array(sample_start_tokens, dtype=np.int32)
sample_start_tokens = torch.tensor(sample_start_tokens, device=device)
NUM_SAMPLES = 5
sample_start_tokens = torch.stack([ sample_start_tokens for _ in range(NUM_SAMPLES)], dim=0)
MAX_SAMPLE_LENGTH = 20
SAVE_FREQUENCY = 20
CHECKPOINT_DIR = "./checkpoints"
LOG_FREQUENCY = 10
LOG_FILE = "./train.log"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
assert BATCH_SIZE % (MINI_BATCH_SIZE * ddp_world_size ) == 0
GRAD_ACCUM_STEPS = BATCH_SIZE // (MINI_BATCH_SIZE * ddp_world_size)
if is_main_process:
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
        batch_token_size = self.batch_size * self.num_tokens * ddp_world_size + 1
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
            else:
                self.curr_shard = np.load(os.path.join(self.shards_path, f"{self.shard_files[self.curr_shard_idx]}"))
                tokens = np.concatenate(tokens, self.curr_shard[:(batch_token_size - len(tokens))], axis=0) 
                self.curr_index = batch_token_size - len(tokens) - 1
        else:
            self.curr_index += batch_token_size - 1
        curr_device_batch_size = self.batch_size * self.num_tokens
        tokens = tokens[ddp_local_rank * curr_device_batch_size:((ddp_local_rank + 1) * curr_device_batch_size + 1)]
        x = tokens[:-1]
        y = tokens[1:]
        x = torch.tensor(x.astype(np.int32)).view(self.batch_size, self.num_tokens)
        y = torch.tensor(y.astype(np.int32)).view(self.batch_size, self.num_tokens)
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
if is_main_process:
    print(f"Number of steps: {NUM_BATCHES}")
train_data_loader = Dataloader(FINEWEB_PATH, MINI_BATCH_SIZE, NUM_TOKENS, eot)
val_data_loader = Dataloader(FINEWEB_PATH, MINI_BATCH_SIZE, NUM_TOKENS, eot, split="val")

# loss and optimizer
loss = CrossEntropyLoss()
loss.to(device)
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == "cuda"
if is_main_process:
    print(f"Fused param in optimizer: {use_fused}")
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=use_fused)
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


def val_model():
    model.eval()
    with torch.no_grad():
        val_data_loader.reset()
        val_data_loader_iterator = iter(val_data_loader)
        val_loss_scalar = 0.0
        num_val_batches = 0
        for _ in range(20):
            x, y = next(val_data_loader_iterator)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x)
                loss_value = loss(logits.view(logits.shape[0] * logits.shape[1], logits.shape[-1]), y.view(y.shape[-1] * y.shape[-2]))
                val_loss_scalar += loss_value
            num_val_batches += 1
        val_loss_scalar /= num_val_batches
        if ddp:
            dist.all_reduce(val_loss_scalar, op=dist.ReduceOp.AVG)
        return val_loss_scalar


def sample_model():
    """create samples from model using topk = 50"""
    model.eval()
    sample_sentences = []
    with torch.no_grad():
        # stacking sample start tokens
        curr_sample_start_tokens = sample_start_tokens.view(*sample_start_tokens.shape).to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            for _ in range(MAX_SAMPLE_LENGTH):
                logits = model(curr_sample_start_tokens)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                topk = probs.topk(50)
                tokens, probability = topk.indices, topk.values
                next_token = torch.multinomial(probability, num_samples=1)
                next_token = tokens.gather(dim=-1, index=next_token)
                # update sample start tokens
                curr_sample_start_tokens = torch.cat((curr_sample_start_tokens, next_token), dim=1)
            for i in range(curr_sample_start_tokens.shape[0]):
                sample_sentences.append(encoder.decode(curr_sample_start_tokens[i].cpu().numpy().tolist()))
    return sample_sentences

# training loop
lr = 0
logs = []
train_losses = []
val_losses = []
f = open(LOG_FILE, "w")
f.close()
for epoch in range(EPOCHS):
    if is_main_process:
        print(f"Epoch: {epoch}")
    train_data_loader_iterator = iter(train_data_loader)
    for step in range(NUM_BATCHES):
        model.eval()
        if step >= MAX_STEPS:
            break
        if step % VAL_FREQUENCY == 0:
            val_loss_scalar = val_model()
            if is_main_process:
                log_str = f"Validation loss: {val_loss_scalar:.4f} at step {step}"
                logs.append(log_str)
                print(log_str)
        if step % SAMPLE_FREQUENCY == 0:
            if is_main_process:
                print(f"Sampling at step {step}")
                samples = sample_model()
                for sample in samples:
                    print(sample)
        if step % SAVE_FREQUENCY == 0:
            if is_main_process:
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_{step}.pt")
        if step % LOG_FREQUENCY == 0:
            if is_main_process:
                with open(LOG_FILE, "a") as f:
                    f.write("\n".join(logs))
                logs = []
                plt.figure()
                plt.plot(train_losses)
                plt.plot(val_losses)
                plt.legend(["train", "val"])
                plt.savefig(f"{CHECKPOINT_DIR}/losses_{step}.png")
        model.train()
        st = time.time()
        # forward pass
        loss_scalar = 0.0
        lr = get_lr(lr, step)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()
        for mini_step in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_data_loader_iterator)
            except StopIteration:
                pass  # end of iterator. looping back to start
            x = x.to(device)
            y = y.to(device)
            if ddp:
                model.require_backward_grad_sync = (mini_step == GRAD_ACCUM_STEPS - 1)# sync gradients only in the last mini step
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x)
                loss_value = loss(logits.view(logits.shape[0] * logits.shape[1], logits.shape[-1]), y.view(y.shape[-1] * y.shape[-2]))
                loss_value /= GRAD_ACCUM_STEPS
            # backward pass
            loss_value.backward()
            loss_scalar += loss_value
        if ddp:
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.AVG)

        # clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        time_taken = end - st
        if is_main_process:
            log_str = f"Step: {step}, loss: {loss_scalar.item():.4f}, norm: {norm:.4f}, lr: {lr:.4e}, time taken: {time_taken * 1000:.4f} ms, tokens per second: {(BATCH_SIZE * NUM_TOKENS/time_taken):.4f}"
            logs.append(log_str)
            print(log_str)
            train_losses.append(loss_scalar.item())
            val_losses.append(val_loss_scalar.item())

if ddp:
    destroy_process_group()

# final validation
val_loss_scalar = val_model()
if is_main_process:
    log_str = f"Final validation loss: {val_loss_scalar:.4f}"
    logs.append(log_str)
    print(log_str)
    val_losses.append(val_loss_scalar.item())

# final sample
if is_main_process:
    samples = sample_model()
    for sample in samples:
        print(sample)

# save model
if is_main_process:
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model.pt")

# save logs
with open(LOG_FILE, "a") as f:
    f.write("\n".join(logs))
plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(["train", "val"])
plt.savefig(f"{CHECKPOINT_DIR}/losses.png")