########################################################################################
# Summary script implementing all of chapter 2. This is a combination of all the Jupyter
# notebooks in # this section.
########################################################################################

import re
import torch
import tiktoken
import urllib.request

from pathlib import Path
from tiktoken import Encoding

from llm_scratch import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    create_dataloader_v1,
    get_default_device,
    move_batch_to_device,
)


########################################################################################
# CUDA checks
device = get_default_device()
print("Using device:", device)

if device.type == "cuda":
    print("CUDA Available!")
    print("Device Name:", torch.cuda.get_device_name(0))
    
    # Reusable pattern: create standalone tensors directly on the selected device.
    cuda_device_tensor = torch.tensor([1.0, 2.0], device=device)
    print("Test Tensor on GPU:", cuda_device_tensor)
    print("Calculation Success!", cuda_device_tensor + cuda_device_tensor)
else:
    print("No CUDA-enabled graphics card detected.")

print()
print()

########################################################################################
# Download "The Verdict" for use as sample training text
file_path = Path(__file__).resolve().parent / "the-verdict.txt"

if file_path.is_file() and file_path.stat().st_size > 0:
    pass
else:
    print("Downloading 'The Verdict'")
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")
    urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    verdict_text = f.read()
print("Total number of character:", len(verdict_text))
print(verdict_text[:99])

print()
print()

########################################################################################
# Word tokenizer
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("word tokens:", {len(preprocessed)})
print(preprocessed[:99])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("unique word tokens:", vocab_size)
print(all_words[:99])

# Assign token IDs (this doesn't really matter for word encoding)
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("ids:", ids)
print("text:", tokenizer.decode(ids))

# Now with some text not in the vocabulary
text = "Hello, do you like tea?"
try:
    print(tokenizer.encode(text))
except KeyError as exc:
    print(f"Unable to encode text; token {exc} is not in the vocabulary.")

# Special Word Tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print("new vocab length: ", len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# Now with Tokenizer V2
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("new text:", text)

tokenizer = SimpleTokenizerV2(vocab)
print("new token ids:", tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

print()
print()

########################################################################################
# BPE tokenizer
tokenizer: Encoding = tiktoken.get_encoding("gpt2")   # vocabulary of 50,257 byte pairs

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("bpe ids:", integers)

strings = tokenizer.decode(integers)
print("bpe text:", strings)

# Let's turn BPE on the-verdict
enc_text = tokenizer.encode(verdict_text)
print("len of verdict bpe:", len(enc_text))

# take a 51 token sample for demonstration
enc_sample = enc_text[50:]

# sliding window of 4 tokens
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
print()

print("Illustrate next word prediction tasks")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

print()

print("Test the dataloader with a batch size of 1 for an LLM with a context size of 4 to develop an intuition")
# these parameters will create tensors of 1 row (batch size) of length 4 (max length), that shifts by 1 (stride) from batch to batch
dataloader = create_dataloader_v1(
    verdict_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False,
    pin_memory=device.type == "cuda",
)
data_iter = iter(dataloader)
# Reusable pattern: DataLoader batches start on CPU; move each batch at the boundary.
first_batch = move_batch_to_device(next(data_iter), device)
print("first batch:", first_batch)

# To understand the meaning of stride=1, let’s fetch another batch from this dataset:
# notice how second batch X is the same as first batch Y
second_batch = move_batch_to_device(next(data_iter), device)
print("second batch:", second_batch)
print()

dataloader = create_dataloader_v1(
    verdict_text,
    batch_size=2,
    max_length=8,
    stride=1,
    shuffle=False,
    pin_memory=device.type == "cuda",
)
data_iter = iter(dataloader)
first_batch = move_batch_to_device(next(data_iter), device)
print("first batch:", first_batch)
second_batch = move_batch_to_device(next(data_iter), device)
print("second batch:", second_batch)
print()

dataloader = create_dataloader_v1(
    verdict_text,
    batch_size=8,
    max_length=4,
    stride=2,
    shuffle=False,
    pin_memory=device.type == "cuda",
)
data_iter = iter(dataloader)
first_batch = move_batch_to_device(next(data_iter), device)
print("first batch:")
print(first_batch[0])
print(first_batch[1])
second_batch = move_batch_to_device(next(data_iter), device)
print("second batch:")
print(second_batch[0])
print(second_batch[1])
print()

########################################################################################
# Creating token embeddings
# Reusable pattern: pass device=... for tensors created outside a DataLoader.

# these are arbitrary hyper parameters to fiddle with
vocab_size = 6
output_dim = 3

# intializes a random tensor as a 6 row (each item in the vocab) 3 column matrix
# but this embedding is unintialized and random small values
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print("item [3]:", embedding_layer(torch.tensor([3])))

input_ids = torch.tensor([2, 3, 5, 1])
# gets the embedding vector for each token id
print(embedding_layer(input_ids))
print()
print()

########################################################################################
# Encoding word positions with bigger embedding space
vocab_size = 50257  # gpt 2 size
output_dim = 256    # encode the input tokens into a 256-dimensional vector representation
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim, device=device)

max_length = 4
dataloader = create_dataloader_v1(verdict_text,
                                  batch_size=8,
                                  max_length=max_length,
                                  stride=max_length,
                                  shuffle=False,
                                  pin_memory=device.type == "cuda")
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

print()
print()
