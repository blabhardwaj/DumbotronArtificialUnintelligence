import argparse
import globalSettings
import tokenizer.trainTokenizer as trainTokenizer


# Reading arguments
def fileArgumentsParse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainTokenizer", action="store_true", help="Train the tokenizer"
    )
    args = parser.parse_args()

    return args


def main():
    # Reading flags while running the program
    args = fileArgumentsParse()

    if args.trainTokenizer:
        trainTokenizer.trainTokenizer(
            globalSettings.DATASET_LOCATION,
            globalSettings.TOKENIZER_PREFIX,
            globalSettings.VOCAB_SIZE,
        )


if __name__ == "__main__":
    main()
