from huggingface_hub import snapshot_download
import tiktoken
import multiprocessing as mp
import os
import numpy as np

download_folder = snapshot_download(
                "HuggingFaceFW/fineweb", 
                repo_type="dataset",
                local_dir="./fineweb/",
                # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
                allow_patterns="sample/10BT/01*")

encoder = tiktoken.encoding_for_model("gpt2")


def tokenize(file_name):
    with open(file_name, "r") as f:
        dataset = f.read()
    dataset_tokens = encoder.encode(dataset)
    dataset_tokens = dataset_tokens + [encoder.eot]  # adding eot to delimit
    dataset_tokens = np.array(dataset_tokens, dtype=np.uint16)


filenames = os.listdir(download_folder)

BATCH_SIZE = 2 # number of files in each batch
SHARD_SIZE = int(1e7) # 10M tokens per shard
# multiprocessing
curr_tokens = np.empty((SHARD_SIZE,), dtype=np.uint16)
curr_index = 0
curr_shard = 0
for i in range(0, len(filenames), BATCH_SIZE):
    with mp.Pool(mp.cpu_count()) as p:
        for tokens in p.imap(tokenize, filenames, chunksize=16):
            remaining_tokens = None
            if len(tokens) > (SHARD_SIZE - curr_index):
                remaining_tokens = tokens[SHARD_SIZE - curr_index:]
                tokens = tokens[:SHARD_SIZE - curr_index]
            curr_tokens[curr_index:curr_index + len(tokens)] = tokens
            curr_index += len(tokens)
            if remaining_tokens is not None:
                # writing current shard
                split = "val" if curr_shard == 0 else "train"
                file_dir = f"./fineweb/{split}/"
                os.makedirs(file_dir, exist_ok=True)
                np.save(f"./fineweb/{split}/{curr_shard}.npy", curr_tokens)
                curr_shard += 1
                curr_tokens = np.empty((SHARD_SIZE,), dtype=np.uint16)
                # filling remaining tokens
                curr_tokens[:len(remaining_tokens)] = remaining_tokens
                curr_index = len(remaining_tokens)

# writing last shard
if curr_index > 0:
    split = "val" if curr_shard == 0 else "train"
    file_dir = f"./fineweb/{split}/"
    os.makedirs(file_dir, exist_ok=True)
    np.save(f"./fineweb/{split}/{curr_shard}.npy", curr_tokens)
        
        