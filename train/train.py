import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import TokenDataLoader
from transformer.pytorchModel import BuildPytorchModel
import globalSettings

def StartTrainingLoop(
        tokenFilePath, vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate,
        batchSize=32, learningRate=3e-4, numEpochs=10, evalInterval=100
    ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on {device}...")
    
    loader = TokenDataLoader(tokenFilePath, batchSize, maxContextLength)
    
    model = BuildPytorchModel(vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate)
    model.to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01)
    
    # Estimate total training steps for the scheduler
    trainTokens = len(loader.trainData)
    stepsPerEpoch = max(1, trainTokens // (batchSize * maxContextLength))
    totalSteps = stepsPerEpoch * numEpochs
    warmupSteps = min(100, totalSteps // 10)
    
    def lrLambda(step):
        if step < warmupSteps:
            return step / max(1, warmupSteps)
        progress = (step - warmupSteps) / max(1, totalSteps - warmupSteps)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)
    
    print(f"Steps per epoch: ~{stepsPerEpoch} | Total steps: ~{totalSteps} | Warmup: {warmupSteps}")
    
    os.makedirs("checkpoints", exist_ok=True)
    globalStep = 0
    
    for epoch in range(numEpochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{numEpochs} ---")
        loader.ResetEpoch()
        iterNum = 0
        
        while True:
            model.train()
            
            try:
                x, y = loader.GetBatch("train")
            except StopIteration:
                break # Epoch finished
                
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            
            B, T, C = logits.shape
            logitsFlattened = logits.view(B * T, C)
            yFlattened = y.view(B * T)
            #print(logits.mean().item(), logits.std().item())
            loss = F.cross_entropy(logitsFlattened, yFlattened)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            globalStep += 1
            
            if iterNum % evalInterval == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        loaderVal = TokenDataLoader(tokenFilePath, batchSize, maxContextLength)
                        loaderVal.ResetEpoch()
                        xVal, yVal = loaderVal.GetBatch("val")
                        xVal, yVal = xVal.to(device), yVal.to(device)
                        
                        valLogits = model(xVal)
                        valLoss = F.cross_entropy(valLogits.view(-1, C), yVal.view(-1))
                        
                        valPredictions = torch.argmax(valLogits, dim=-1)
                        valAccuracy = (valPredictions == yVal).float().mean().item()
                        
                        print(f"Epoch {epoch + 1} | Step {iterNum} | Train Loss: {loss.item():.4f} | Val Loss: {valLoss.item():.4f} | Val Acc: {valAccuracy:.4f}")
                        
                        checkpointPath = f"checkpoints/model_epoch_{epoch+1}_step_{iterNum}.pt"
                        torch.save(model.state_dict(), checkpointPath)
                    except StopIteration:
                        pass
            iterNum += 1

    print("Training Complete!")
    torch.save(model.state_dict(), "checkpoints/model_final.pt")
    
if __name__ == "__main__":
    StartTrainingLoop(
        tokenFilePath="data/tokenIds.npy",
        vocabSize=globalSettings.VOCAB_SIZE,
        dModel=globalSettings.D_MODEL,
        numHeads=globalSettings.NUM_HEADS,
        dff=globalSettings.DFF,
        maxContextLength=globalSettings.MAX_CONTEXT_LENGTH,
        numLayers=globalSettings.NUM_LAYERS,
        dropoutRate=globalSettings.DROPOUT_RATE,
        batchSize=4,
        numEpochs=10,
        evalInterval=5
    )
