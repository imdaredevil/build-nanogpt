from model import GPT2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import tiktoken

encoder = tiktoken.encoding_for_model("gpt2")
config = GPT2Config()
model = GPT2(config)

BATCH_SIZE = 8
NUM_TOKENS = 16

TINY_SHAKESPEARE_PATH = "data/tiny_shakespeare.txt"
with open(TINY_SHAKESPEARE_PATH, "r") as f:
    shakespeare_dataset = f.read()
