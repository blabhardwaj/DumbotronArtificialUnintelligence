import sentencepiece as spm
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import globalSettings

def tokenizeDataset():
    sp = spm.SentencePieceProcessor()
    sp.load(globalSettings.TOKENIZER_PREFIX + ".model")
    
    with open(globalSettings.DATASET_LOCATION, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokenIds = sp.encode(text)
    
    print("First 50 tokens:", tokenIds[:50])
    print("Total tokens:", len(tokenIds))
    print("Vocab size:", sp.vocab_size())
    
    np.save("data/tokenIds.npy", tokenIds)
    print("Saved tokenIds.npy")

if __name__ == "__main__":
    tokenizeDataset()
