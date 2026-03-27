import os
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader, random_split

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model.transformer import Transformer


class tokenDataset(Dataset):
    def __init__(self, inputData, targetData, contextWindow):
        self.inputData = torch.tensor(inputData, dtype=torch.long)
        self.targetData = torch.tensor(targetData, dtype=torch.long)
        self.contextWindow = contextWindow

    def __len__(self):
        return len(self.inputData) - self.contextWindow

    def __getitem__(self, idx):
        x = self.inputData[idx : idx + self.contextWindow]
        y = self.targetData[idx : idx + self.contextWindow]
        return x, y


def train(
    vocabSize: int,
    embedDim: int,
    numHeads: int,
    ffDim: int,
    numLayers: int,
    encodingBase: float,
    contextWindow: int,
    dropout: float,
    batchSize: int,
    learningRate: float,
    epochs: int,
    modelSavePath: str,
):
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Load data ---
    inputData = np.load("data/input.npy")
    targetData = np.load("data/target.npy")

    dataset = tokenDataset(inputData, targetData, contextWindow)
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    # --- Model ---
    model = Transformer(
        vocabSize=vocabSize,
        embedDim=embedDim,
        numHeads=numHeads,
        ffDim=ffDim,
        numLayers=numLayers,
        encodingBase=encodingBase,
        contextWindow=contextWindow,
        dropout=dropout,
    ).to(device)

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    lossFunc = nn.CrossEntropyLoss()

    # --- Split dataset ---
    g = torch.Generator().manual_seed(42)  # any fixed number

    trainSize = int(0.9 * len(dataset))
    valSize = len(dataset) - trainSize

    trainDataset, valDataset = random_split(
        dataset,
        [trainSize, valSize],
        generator=g,
    )

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        trainLoss = 0.0

        for batchIdx, (x, y) in enumerate(trainLoader):
            x, y = x.to(device), y.to(device)
            logits = model(x).view(-1, vocabSize)
            y = y.view(-1)
            loss = lossFunc(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            trainLoss += loss.item()
            print(
                f"  Batch {batchIdx + 1}/{len(trainLoader)}  |  Loss: {loss.item():.4f}",
                end="\r",
            )

        avgTrainLoss = trainLoss / len(trainLoader)

        # --- Validation ---
        model.eval()
        valLoss = 0.0
        with torch.no_grad():
            for x, y in valLoader:
                x, y = x.to(device), y.to(device)
                logits = model(x).view(-1, vocabSize)
                y = y.view(-1)
                valLoss += lossFunc(logits, y).item()

        avgValLoss = valLoss / len(valLoader)
        print(
            f"Epoch {epoch + 1}/{epochs}  |  Train Loss: {avgTrainLoss:.4f}  |  Val Loss: {avgValLoss:.4f} | Diff Loss: {(avgValLoss - avgTrainLoss):.4f}  |  Perplexity: {math.exp(avgValLoss):.2f}"
        )

    # --- Save model ---
    os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)
    torch.save(model.state_dict(), modelSavePath)
    print(f"Model saved to {modelSavePath}")
