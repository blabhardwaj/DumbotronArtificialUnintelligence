# --- Tokenizer ---
DATASET_LOCATION = "data/dataset.txt"
TOKENIZER_PREFIX = "tokenizer/data/tokenizer"
VOCAB_SIZE = 1180

# --- Model ---
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 4
ENCODING_BASE = 10000.0
CONTEXT_WINDOW = 128
DROPOUT = 0.1

# --- Training ---
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 10
MODEL_SAVE_PATH = "model/saved/transformer.pt"
