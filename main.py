import argparse
import globalSettings
import tokenizer.trainTokenizer as trainTokenizer
import tokenizer.tokenizer as tokenizeData

import train.train as train
import generate.generate as generate

def fileArgumentsParse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainTokenizer", action="store_true", help="Train the tokenizer"
    )
    parser.add_argument(
        "--trainModel", action="store_true", help="Format tokens and train the PyTorch Transformer from scratch"
    )
    parser.add_argument(
        "--generate", type=str, help="Generate tokens from a given text prompt"
    )
    args = parser.parse_args()

    return args

def main():
    args = fileArgumentsParse()

    if args.trainTokenizer:
        trainTokenizer.trainTokenizer(
            globalSettings.DATASET_LOCATION,
            globalSettings.TOKENIZER_PREFIX,
            globalSettings.VOCAB_SIZE,
        )
        tokenizeData.tokenizeDataset()
        
    if args.trainModel:
        print("Config used:")
        print("vocabSize:", globalSettings.VOCAB_SIZE)
        print("dModel:", globalSettings.D_MODEL)
        print("numHeads:", globalSettings.NUM_HEADS)
        print("dff:", globalSettings.DFF)
        print("numLayers:", globalSettings.NUM_LAYERS)
        
        train.StartTrainingLoop(
            tokenFilePath="data/tokenIds.npy",
            vocabSize=globalSettings.VOCAB_SIZE,
            dModel=globalSettings.D_MODEL,
            numHeads=globalSettings.NUM_HEADS,
            dff=globalSettings.DFF,
            maxContextLength=globalSettings.MAX_CONTEXT_LENGTH,
            numLayers=globalSettings.NUM_LAYERS,
            dropoutRate=globalSettings.DROPOUT_RATE,
            batchSize=32,
            numEpochs=100,
            learningRate=3e-4,
            evalInterval=500
        )
        
    if args.generate:
        prompt = args.generate
        model, sp, device = generate.loadModelAndTokenizer()
        generate.generateText(prompt, model, sp, device, maxNewTokens=200, temperature=0.3)

if __name__ == "__main__":
    main()
