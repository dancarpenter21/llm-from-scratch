########################################################################################
# Summary script implementing all of chapter 2. This is a combination of all the Jupyter
# notebooks in # this section.
########################################################################################

import re
import torch
import urllib.request

from pathlib import Path

from llm_scratch.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2


########################################################################################
# CUDA checks
if torch.cuda.is_available():
    print("CUDA Available!")
    print("Device Name:", torch.cuda.get_device_name(0))
    
    # Verify it is working by pushing a tensor to the GPU and doing math
    cuda_device_tensor = torch.tensor([1.0, 2.0]).cuda()
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