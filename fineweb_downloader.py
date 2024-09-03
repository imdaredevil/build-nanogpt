from huggingface_hub import snapshot_download
import tiktoken
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm

download_folder = snapshot_download(
                "HuggingFaceFW/fineweb", 
                repo_type="dataset",
                local_dir="./data/fineweb/",
                # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
                allow_patterns="sample/10BT/*")


encoder = tiktoken.encoding_for_model("gpt2")
eot = encoder._special_tokens['<|endoftext|>']

def tokenize(row):
    # read parquet file and get text from "text" column
    dataset = row["text"]

    dataset_tokens = encoder.encode_ordinary(dataset)
    dataset_tokens = dataset_tokens + [eot]  # adding eot to delimit
    dataset_tokens = np.array(dataset_tokens, dtype=np.uint16)
    return dataset_tokens

def file_generator(filenames):
    for filename in filenames:
        df = pd.read_parquet(filename)
        for _, row in df.iterrows():
            yield row

FILE_PATH = f"{download_folder}/sample/10BT/"
filenames = os.listdir(FILE_PATH)
filenames = [FILE_PATH + f for f in filenames]
SHARD_SIZE = int(1e8) # 10M tokens per shard
# multiprocessing
curr_tokens = np.empty((SHARD_SIZE,), dtype=np.uint16)
curr_index = 0
curr_shard = 0
progress_bar = None
with mp.Pool(mp.cpu_count()) as p:
    for tokens in p.imap(tokenize, file_generator(filenames), chunksize=16):
        remaining_tokens = None
        if len(tokens) > (SHARD_SIZE - curr_index):
            remaining_tokens = tokens[SHARD_SIZE - curr_index:]
            tokens = tokens[:SHARD_SIZE - curr_index]
        curr_tokens[curr_index:curr_index + len(tokens)] = tokens
        curr_index += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=SHARD_SIZE, desc=f"Shard {curr_shard}: ", unit="tokens")
        progress_bar.update(curr_index)
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
            progress_bar = None

# writing last shard
if curr_index > 0:
    split = "val" if curr_shard == 0 else "train"
    file_dir = f"./fineweb/{split}/"
    os.makedirs(file_dir, exist_ok=True)
    np.save(f"./fineweb/{split}/{curr_shard}.npy", curr_tokens)
        
        