import torch
import torch.nn.functional as F
import sentencepiece as spm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.pytorchModel import BuildPytorchModel
import globalSettings

def loadModelAndTokenizer(modelPath="checkpoints/model_epoch_22_step_0.pt", tokenizerPath="tokenizer/data/tokenizer.model"):
    sp = spm.SentencePieceProcessor()
    if not sp.load(tokenizerPath):
        raise FileNotFoundError(f"Could not load tokenizer at {tokenizerPath}")
        
    vocabSize = globalSettings.VOCAB_SIZE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BuildPytorchModel(
        vocabSize=vocabSize,
        dModel=globalSettings.D_MODEL,
        numHeads=globalSettings.NUM_HEADS,
        dff=globalSettings.DFF,
        maxContextLength=globalSettings.MAX_CONTEXT_LENGTH,
        numLayers=globalSettings.NUM_LAYERS,
        dropoutRate=globalSettings.DROPOUT_RATE
    )
    
    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath, map_location=device, weights_only=True))
        print(f"Loaded weights from {modelPath}")
    else:
        print(f"WARNING: Weights {modelPath} not found. Generating with an untrained random model!")
        
    model.to(device)
    model.eval()
    
    return model, sp, device

def generateText(prompt, model, sp, device, maxNewTokens=50, temperature=0.8, topK=40):
    tokenIds = sp.encode(prompt)
    # Clamp any out-of-range token IDs to UNK (3) to prevent embedding crash
    vocabSize = model.tokenEmbedding.weight.shape[0]
    tokenIds = [t if t < vocabSize else 3 for t in tokenIds]
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
        
        # Stop early if the model generates an EOS token
        if sp.eos_id() is not None and next_token.item() == sp.eos_id():
            break
        
    outputString = sp.decode(generatedTokens)
    print(f"\nOutput: {outputString}\n")
    print("context shape:", context.shape)
    print("logits shape:", logits.shape)
    return outputString

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python generate.py \"Your prompt here\"")
        sys.exit(1)
        
    prompt = sys.argv[1]
    
    model, sp, device = loadModelAndTokenizer()
    generateText(prompt, model, sp, device, maxNewTokens=50, temperature=0.8)
