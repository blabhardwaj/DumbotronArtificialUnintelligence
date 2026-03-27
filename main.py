import argparse
import globalSettings
import tokenizer.trainTokenizer as trainTokenizer
import tokenizer.tokenizer as tokenizerModule
import data.prepareDataset as prepareDataset
import model.train as trainModule
import generate as generateModule
import countParameter


def fileArgumentsParse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Train the tokenizer and tokenize the dataset",
    )
    parser.add_argument(
        "--prepareDataset", action="store_true", help="Prepare the dataset for training"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the transformer model"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate text from a prompt"
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Prompt to generate from"
    )
    parser.add_argument(
        "--maxNewTokens", type=int, default=200, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--topK", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--countParameter",
        action="store_true",
        help="Count the number of parameters in the model",
    )

    return parser.parse_args()


def main():
    args = fileArgumentsParse()

    if args.tokenize:
        trainTokenizer.trainTokenizer(
            globalSettings.DATASET_LOCATION,
            globalSettings.TOKENIZER_PREFIX,
            globalSettings.VOCAB_SIZE,
        )
        tokenizerModule.tokenizer()

    if args.prepareDataset:
        prepareDataset.prepareDataset()

    if args.train:
        trainModule.train(
            vocabSize=globalSettings.VOCAB_SIZE,
            embedDim=globalSettings.EMBED_DIM,
            numHeads=globalSettings.NUM_HEADS,
            ffDim=globalSettings.FF_DIM,
            numLayers=globalSettings.NUM_LAYERS,
            encodingBase=globalSettings.ENCODING_BASE,
            contextWindow=globalSettings.CONTEXT_WINDOW,
            dropout=globalSettings.DROPOUT,
            batchSize=globalSettings.BATCH_SIZE,
            learningRate=globalSettings.LEARNING_RATE,
            epochs=globalSettings.EPOCHS,
            modelSavePath=globalSettings.MODEL_SAVE_PATH,
        )

    if args.generate:
        output = generateModule.generate(
            prompt=args.prompt,
            vocabSize=globalSettings.VOCAB_SIZE,
            embedDim=globalSettings.EMBED_DIM,
            numHeads=globalSettings.NUM_HEADS,
            ffDim=globalSettings.FF_DIM,
            numLayers=globalSettings.NUM_LAYERS,
            encodingBase=globalSettings.ENCODING_BASE,
            contextWindow=globalSettings.CONTEXT_WINDOW,
            modelSavePath=globalSettings.MODEL_SAVE_PATH,
            tokenizerPath=globalSettings.TOKENIZER_PREFIX + ".model",
            maxNewTokens=args.maxNewTokens,
            temperature=args.temperature,
            topK=args.topK,
        )
        print(output)
    
    if args.countParameter:
        countParameter.countParameter(globalSettings.MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
