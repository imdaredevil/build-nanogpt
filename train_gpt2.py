from model import GPT2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import tiktoken
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import time

torch.manual_seed(42)

encoder = tiktoken.encoding_for_model("gpt2")
config = GPT2Config(vocab_size=50304)
model = GPT2(config)
model = torch.compile(model)

EPOCHS = 1
BATCH_SIZE = 128
MINI_BATCH_SIZE = 32 # we use gradient accumulation here.
NUM_TOKENS = 512
MAX_STEPS = 50
assert BATCH_SIZE % MINI_BATCH_SIZE == 0
GRAD_ACCUM_STEPS = BATCH_SIZE // MINI_BATCH_SIZE
print(f"grad accumulation steps: {GRAD_ACCUM_STEPS}")
# creating dataloader

class Dataloader:
    def __init__(self, file_name, batch_size, num_tokens, encoder):
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        with open(self.file_name, "r") as f:
            dataset = f.read()
        self.dataset_tokens = encoder.encode(dataset)
        self.num_batches = len(self.dataset_tokens) // (self.batch_size * self.num_tokens)
        self.curr_index = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        batch_token_size = self.batch_size * self.num_tokens
        if (self.curr_index + batch_token_size) >= len(self.dataset_tokens):
            self.reset()
            raise StopIteration
        x = self.dataset_tokens[self.curr_index : self.curr_index + batch_token_size]
        y = self.dataset_tokens[self.curr_index + 1 : self.curr_index + batch_token_size + 1]
        self.curr_index += self.batch_size * self.num_tokens
        x = torch.tensor(x, dtype=torch.long).view(self.batch_size, self.num_tokens)
        y = torch.tensor(y, dtype=torch.long).view(self.batch_size, self.num_tokens)
        return x, y

    def reset(self):
        self.curr_index = 0

TINY_SHAKESPEARE_PATH = "data/tiny_shakespeare.txt"
data_loader = Dataloader(TINY_SHAKESPEARE_PATH, MINI_BATCH_SIZE, NUM_TOKENS, encoder)
print(f"Number of mini-batches: {len(data_loader)}")
NUM_BATCHES = 50 # len(data_loader) // GRAD_ACCUM_STEPS
print(f"Number of batches: {NUM_BATCHES}")



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
optimizer = Adam(model.parameters(), lr=3e-4)
optimizer.param_groups[0]["lr"] = 3e-4

# training loop
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    data_loader_iterator = iter(data_loader)
    for step in range(NUM_BATCHES):
        if step >= MAX_STEPS:
            break
        st = time.time()
        # forward pass
        loss_scalar = 0
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_loader_iterator)
            except StopIteration:
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
        print(f"Step: {step}, loss: {loss_scalar:.4f}, norm: {norm:.4f}, time taken: {time_taken * 1000:.4f} ms, tokens per second: {(BATCH_SIZE * NUM_TOKENS/time_taken):.4f}")

torch.save(model.state_dict(), "model.pt")
model.state_dict()