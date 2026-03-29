import torch
import sentencepiece as spm

from model.transformer import Transformer


def generate(
    prompt: str,
    vocabSize: int,
    embedDim: int,
    numHeads: int,
    ffDim: int,
    numLayers: int,
    encodingBase: float,
    contextWindow: int,
    modelSavePath: str,
    tokenizerPath: str,
    maxNewTokens: int = 200,
    temperature: float = 1.0,
    topK: int = 50,
):
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load tokenizer ---
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizerPath)

    # --- Load model ---
    model = Transformer(
        vocabSize=vocabSize,
        embedDim=embedDim,
        numHeads=numHeads,
        ffDim=ffDim,
        numLayers=numLayers,
        encodingBase=encodingBase,
        contextWindow=contextWindow,
        dropout=0.0,  # no dropout during inference
    ).to(device)

    model.load_state_dict(torch.load(modelSavePath, map_location=device))
    model.eval()

    # --- Encode prompt ---
    inputIds = tokenizer.Encode(prompt, out_type=int)
    tokens = (
        torch.tensor(inputIds, dtype=torch.long).unsqueeze(0).to(device)
    )  # (1, seq)

    # --- Generation loop ---
    with torch.no_grad():
        for _ in range(maxNewTokens):
            tokensCropped = tokens[:, -contextWindow:]

            logits = model(tokensCropped)
            nextLogits = logits[:, -1, :]

            # Clamp to valid vocab range — prevents out of range token IDs
            nextLogits[:, vocabSize:] = float("-inf")

            nextLogits = nextLogits / temperature

            if topK > 0:
                topKValues, _ = torch.topk(nextLogits, topK)
                minTopK = topKValues[:, -1].unsqueeze(-1)
                nextLogits = nextLogits.masked_fill(nextLogits < minTopK, float("-inf"))

            probs = torch.softmax(nextLogits, dim=-1)
            nextToken = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, nextToken], dim=1)

            if nextToken.item() == tokenizer.eos_id():
                break

    # --- Decode ---
    generatedIds = tokens[0].tolist()
    # Filter out any token IDs outside the valid vocab range
    validIds = [id for id in generatedIds if 0 <= id < vocabSize]

    return tokenizer.Decode(validIds)