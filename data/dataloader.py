import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import globalSettings
class TokenDataLoader:
    def __init__(self, tokenFilePath, batchSize, maxContextLength, splitRatio=0.9):
        self.batchSize = batchSize
        self.maxContextLength = maxContextLength
        self.currentIndex = 0
        
        if not os.path.exists(tokenFilePath):
            raise FileNotFoundError(f"Token file not found at: {tokenFilePath}. Did you run the tokenizer first?")
            
        print(f"Loading tokens from {tokenFilePath}...")
        self.tokens = np.load(tokenFilePath)
        
        splitIndex = int(len(self.tokens) * splitRatio)
        self.trainData = self.tokens[:splitIndex]
        self.valData = self.tokens[splitIndex:]
        
        print(f"Dataset Split: {len(self.trainData):,} Train tokens | {len(self.valData):,} Val tokens")

    def ResetEpoch(self):
        self.currentIndex = 0
        # Build shuffled start indices for this epoch
        data = self.trainData
        availableLength = len(data) - self.maxContextLength
        if availableLength > 0:
            self.shuffledIndices = np.arange(0, availableLength, self.maxContextLength)
            np.random.shuffle(self.shuffledIndices)
        else:
            self.shuffledIndices = np.array([])
        
    def GetBatch(self, split="train"):
        data = self.trainData if split == "train" else self.valData
        
        availableLength = len(data) - self.maxContextLength
        if availableLength <= 0:
             raise ValueError(f"Dataset for split '{split}' is too small ({len(data)} tokens) for context length {self.maxContextLength}.")
             
        if split == "val":
            # For validation, use random sampling (single batch)
            startIndices = np.random.randint(0, availableLength, size=self.batchSize)
        else:
            # For training, use shuffled sequential access
            if self.currentIndex >= len(self.shuffledIndices):
                raise StopIteration("End of epoch")
                
            endIdx = min(self.currentIndex + self.batchSize, len(self.shuffledIndices))
            startIndices = self.shuffledIndices[self.currentIndex:endIdx]
            self.currentIndex = endIdx
            
            if len(startIndices) == 0:
                raise StopIteration("End of epoch")

        xList = [torch.tensor(data[i: i + self.maxContextLength], dtype=torch.long) for i in startIndices]
        yList = [torch.tensor(data[i + 1: i + self.maxContextLength + 1], dtype=torch.long) for i in startIndices]
        
        x = torch.stack(xList)
        y = torch.stack(yList)
        
        return x, y

if __name__ == "__main__":
    TOKEN_FILE = "data/tokenIds.npy"
    BATCH_SIZE = 4
    MAX_CONTEXT_LENGTH = globalSettings.MAX_CONTEXT_LENGTH
    
    loader = TokenDataLoader(TOKEN_FILE, BATCH_SIZE, MAX_CONTEXT_LENGTH)
    
    xTest, yTest = loader.GetBatch("train")
    print(f"\nTrain Batch X shape: {xTest.shape}")
    print(f"Train Batch Y shape: {yTest.shape}")
    
    print("\nFirst sequence X (input): ", xTest[0, :10].tolist(), "...")
    print("First sequence Y (target): ", yTest[0, :10].tolist(), "...")
