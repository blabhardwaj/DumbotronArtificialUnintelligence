import torch
import numpy as np
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import globalSettings


def countParameter(MODEL_SAVE_PATH=None):
    if MODEL_SAVE_PATH is None:
        MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "model_final.pt")
    print("=" * 55)
    print("           MODEL & TRAINING STATISTICS")
    print("=" * 55)
    
    # --- Model Parameters ---
    if os.path.exists(MODEL_SAVE_PATH):
        state_dict = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=True)
        total_params = sum(p.numel() for p in state_dict.values())
    else:
        print(f"  [!] Checkpoint not found: {MODEL_SAVE_PATH}")
        total_params = None
    
    # --- Dataset Stats ---
    token_file = os.path.join(PROJECT_ROOT, "data", "tokenIds.npy")
    if os.path.exists(token_file):
        token_ids = np.load(token_file)
        total_tokens = len(token_ids)
    else:
        print(f"  [!] Token file not found: {token_file}")
        total_tokens = None
    
    # --- Config from globalSettings ---
    vocab_size = globalSettings.VOCAB_SIZE
    d_model = globalSettings.D_MODEL
    num_heads = globalSettings.NUM_HEADS
    dff = globalSettings.DFF
    num_layers = globalSettings.NUM_LAYERS
    max_context_length = globalSettings.MAX_CONTEXT_LENGTH
    dropout_rate = globalSettings.DROPOUT_RATE
    
    # --- Training config (from main.py) ---
    batch_size = 32
    num_epochs = 100
    learning_rate = 3e-4
    eval_interval = 500
    split_ratio = 0.9
    
    # --- Derived Stats ---
    if total_tokens:
        train_tokens = int(total_tokens * split_ratio)
        val_tokens = total_tokens - train_tokens
    else:
        train_tokens = val_tokens = 0
    
    tokens_per_batch = batch_size * max_context_length
    batches_per_epoch = max(1, train_tokens // tokens_per_batch)
    total_batches = batches_per_epoch * num_epochs
    tokens_per_step = tokens_per_batch  # 1 batch = 1 step
    total_tokens_seen = total_batches * tokens_per_batch
    
    # Embedding table size (from checkpoint if available, else from config)
    if total_params and state_dict:
        # Read actual embedding shape from the checkpoint
        emb_key = "tokenEmbedding.weight"
        if emb_key in state_dict:
            actual_vocab = state_dict[emb_key].shape[0]
            actual_dmodel = state_dict[emb_key].shape[1]
            embedding_params = actual_vocab * actual_dmodel
        else:
            embedding_params = vocab_size * d_model
        embedding_pct = (embedding_params / total_params) * 100
    else:
        embedding_params = vocab_size * d_model
        embedding_pct = 0
    
    # Expected initial loss
    initial_loss = math.log(vocab_size)
    
    # Head dimension
    head_dim = d_model // num_heads
    
    # --- Print Everything ---
    print("\n  MODEL ARCHITECTURE")
    print("-" * 55)
    print(f"  Vocab Size:              {vocab_size:,}")
    print(f"  d_model (embed dim):     {d_model}")
    print(f"  Num Heads:               {num_heads}")
    print(f"  Head Dimension:          {head_dim}")
    print(f"  FFN Dim (dff):           {dff}")
    print(f"  Num Layers:              {num_layers}")
    print(f"  Max Context Length:      {max_context_length}")
    print(f"  Dropout Rate:            {dropout_rate}")
    
    if total_params:
        print(f"\n  Total Parameters:        {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Embedding Params:        {embedding_params:,} ({embedding_pct:.1f}% of total)")
        print(f"  Transformer Params:      {total_params - embedding_params:,}")
    
    print(f"\n  Expected Initial Loss:   {initial_loss:.2f} (ln({vocab_size}))")
    
    print("\n  DATASET")
    print("-" * 55)
    if total_tokens:
        print(f"  Total Tokens:            {total_tokens:,}")
        print(f"  Train Tokens (90%):      {train_tokens:,}")
        print(f"  Val Tokens (10%):        {val_tokens:,}")
    
    print("\n  TRAINING CONFIG")
    print("-" * 55)
    print(f"  Num Epochs:              {num_epochs}")
    print(f"  Batch Size:              {batch_size}")
    print(f"  Learning Rate:           {learning_rate}")
    print(f"  Eval Interval:           every {eval_interval} steps")
    
    print("\n  PER-STEP / PER-EPOCH BREAKDOWN")
    print("-" * 55)
    print(f"  Tokens per Batch (step): {tokens_per_batch:,} ({batch_size} × {max_context_length})")
    print(f"  Batches per Epoch:       {batches_per_epoch:,}")
    print(f"  Total Batches (steps):   {total_batches:,}")
    print(f"  Total Tokens Seen:       {total_tokens_seen:,} ({total_tokens_seen/1e6:.1f}M)")
    if total_tokens:
        passes_over_data = total_tokens_seen / train_tokens
        print(f"  Passes Over Train Data:  {passes_over_data:.1f}x")
    
    print("=" * 55)


if __name__ == "__main__":
    countParameter()
