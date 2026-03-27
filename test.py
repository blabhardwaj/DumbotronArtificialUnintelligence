import numpy as np
import sentencepiece as spm

tokenIds = np.load("data/tokenIds.npy")
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/data/tokenizer.model")

print(f"Total tokens    : {len(tokenIds)}")
print(f"Vocab size      : {sp.GetPieceSize()}")
print(f"Max token ID    : {tokenIds.max()}")

with open("data/dataset.txt", "r") as f:
    lines = f.readlines()
print(f"Total sentences : {len(lines)}")
