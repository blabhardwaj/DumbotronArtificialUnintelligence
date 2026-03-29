import torch
import torch.nn.functional as F
import sentencepiece as spm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.pytorchModel import BuildPytorchModel
import globalSettings

def remapStateDict(oldStateDict):
    """
    Remaps keys from the old transformer.pt checkpoint format
    to match the current PytorchModel architecture.
    """
    keyMapping = {
        "embedding.tokenEmb.weight": "tokenEmbedding.weight",
        "embedding.posEnc.pe": "posEncoding.pe",
        "norm.weight": "finalLayernorm.weight",
        "norm.bias": "finalLayernorm.bias",
        "outputHead.weight": "lmHead.weight",
    }
    
    blockMapping = {
        "normOne.weight": "layernorm1.weight",
        "normOne.bias": "layernorm1.bias",
        "normTwo.weight": "layernorm2.weight",
        "normTwo.bias": "layernorm2.bias",
        "attention.attention.in_proj_weight": "attnLayer.qkvProjection.weight",
        "attention.attention.in_proj_bias": "attnLayer.qkvProjection.bias",
        "attention.attention.out_proj.weight": "attnLayer.outputProjection.weight",
        "attention.attention.out_proj.bias": "attnLayer.outputProjection.bias",
        "feedForward.net.0.weight": "ffnLayer.0.weight",
        "feedForward.net.0.bias": "ffnLayer.0.bias",
        "feedForward.net.3.weight": "ffnLayer.2.weight",
        "feedForward.net.3.bias": "ffnLayer.2.bias",
    }
    
    newStateDict = {}
    skipped = []
    
    for oldKey, value in oldStateDict.items():
        # Direct top-level mapping
        if oldKey in keyMapping:
            newStateDict[keyMapping[oldKey]] = value
            continue
        
        # Block-level mapping (blocks.N.old_suffix -> blocks.N.new_suffix)
        matched = False
        if oldKey.startswith("blocks."):
            parts = oldKey.split(".", 2)  # ['blocks', 'N', 'rest']
            if len(parts) == 3:
                blockPrefix = f"blocks.{parts[1]}."
                suffix = parts[2]
                if suffix in blockMapping:
                    newStateDict[blockPrefix + blockMapping[suffix]] = value
                    matched = True
        
        if not matched:
            # Skip keys that don't exist in the new model (e.g. causalMask)
            skipped.append(oldKey)
    
    if skipped:
        print(f"Skipped {len(skipped)} keys from old checkpoint: {skipped}")
    
    return newStateDict

def loadModelAndTokenizer(modelPath="checkpoints/transformer.pt", tokenizerPath="tokenizer/data/tokenizer.model"):
    sp = spm.SentencePieceProcessor()
    if not sp.load(tokenizerPath):
        raise FileNotFoundError(f"Could not load tokenizer at {tokenizerPath}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(modelPath):
        print(f"WARNING: Weights {modelPath} not found!")
        return None, sp, device
    
    # Load old checkpoint and remap keys
    oldStateDict = torch.load(modelPath, map_location=device, weights_only=True)
    newStateDict = remapStateDict(oldStateDict)
    
    # Auto-detect model dimensions from the checkpoint
    dModel = newStateDict["tokenEmbedding.weight"].shape[1]
    vocabSize = newStateDict["tokenEmbedding.weight"].shape[0]
    dff = newStateDict["blocks.0.ffnLayer.0.weight"].shape[0]
    
    # Count number of blocks
    numLayers = 0
    while f"blocks.{numLayers}.layernorm1.weight" in newStateDict:
        numLayers += 1
    
    # Detect numHeads from qkvProjection (3 * dModel)
    # qkvProjection.weight shape is [3*dModel, dModel], default to 4 heads
    numHeads = globalSettings.NUM_HEADS
    
    print(f"Detected from checkpoint: vocabSize={vocabSize}, dModel={dModel}, dff={dff}, numLayers={numLayers}")
    
    model = BuildPytorchModel(
        vocabSize=vocabSize,
        dModel=dModel,
        numHeads=numHeads,
        dff=dff,
        maxContextLength=globalSettings.MAX_CONTEXT_LENGTH,
        numLayers=numLayers,
        dropoutRate=globalSettings.DROPOUT_RATE
    )
    
    model.load_state_dict(newStateDict)
    print(f"Loaded and remapped weights from {modelPath}")
        
    model.to(device)
    model.eval()
    
    return model, sp, device

def generateText(prompt, model, sp, device, maxNewTokens=50, temperature=0.8, topK=40):
    tokenIds = sp.encode(prompt)
    context = torch.tensor([tokenIds], dtype=torch.long, device=device)
    
    print(f"\nPrompt: '{prompt}'")
    print("Generating...", end="", flush=True)
    
    generatedTokens = []
    
    for _ in range(maxNewTokens):
        contextCropped = context[:, -model.maxContextLength:]
        
        with torch.no_grad():
            logits = model(contextCropped)
            
        next_token_logits = logits[:, -1, :] / temperature
        
        if topK is not None:
            v, _ = torch.topk(next_token_logits, min(topK, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generatedTokens.append(next_token.item())
        context = torch.cat((context, next_token), dim=1)
        
        if sp.eos_id() is not None and next_token.item() == sp.eos_id():
            break
        
    outputString = sp.decode(generatedTokens)
    print(f"\nOutput: {outputString}\n")
    return outputString

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generateOldCheckpoint.py \"Your prompt here\"")
        sys.exit(1)
        
    prompt = sys.argv[1]
    
    model, sp, device = loadModelAndTokenizer()
    generateText(prompt, model, sp, device, maxNewTokens=50, temperature=0.8)
