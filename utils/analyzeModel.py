import torch
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
import math
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from transformer.pytorchModel import BuildPytorchModel
from data.dataloader import TokenDataLoader
import globalSettings


def loadModel(checkpoint_path, device):
    """Load a model from a checkpoint file, auto-detecting dimensions."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Auto-detect dimensions from the checkpoint
    vocab_size = state_dict["tokenEmbedding.weight"].shape[0]
    d_model = state_dict["tokenEmbedding.weight"].shape[1]
    dff = state_dict["blocks.0.ffnLayer.0.weight"].shape[0]
    
    num_layers = 0
    while f"blocks.{num_layers}.layernorm1.weight" in state_dict:
        num_layers += 1
    
    model = BuildPytorchModel(
        vocabSize=vocab_size,
        dModel=d_model,
        numHeads=globalSettings.NUM_HEADS,
        dff=dff,
        maxContextLength=globalSettings.MAX_CONTEXT_LENGTH,
        numLayers=num_layers,
        dropoutRate=globalSettings.DROPOUT_RATE
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluateModel(model, token_file, device, batch_size=32, num_batches=5):
    """Evaluate model on validation data, returns avg loss, avg accuracy, perplexity."""
    model.eval()
    vocab_size = model.tokenEmbedding.weight.shape[0]
    
    # Load tokens and clamp to model's vocab size
    all_tokens = np.load(token_file)
    all_tokens = np.clip(all_tokens, 0, vocab_size - 1)
    
    # Split into val data (last 10%)
    split_idx = int(len(all_tokens) * 0.9)
    val_data = all_tokens[split_idx:]
    
    total_loss = 0
    total_acc = 0
    count = 0
    ctx_len = globalSettings.MAX_CONTEXT_LENGTH
    
    with torch.no_grad():
        for i in range(num_batches):
            indices = np.random.randint(0, len(val_data) - ctx_len, size=batch_size)
            x = torch.tensor(np.stack([val_data[j:j+ctx_len] for j in indices]), dtype=torch.long, device=device)
            y = torch.tensor(np.stack([val_data[j+1:j+ctx_len+1] for j in indices]), dtype=torch.long, device=device)
            
            logits = model(x)
            B, T, C = logits.shape
            
            loss = F.cross_entropy(logits.view(-1, C), y.view(-1))
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            count += 1
    
    if count == 0:
        return float('inf'), 0.0, float('inf')
    
    avg_loss = total_loss / count
    avg_acc = total_acc / count
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return avg_loss, avg_acc, perplexity


def generateSample(model, sp, device, prompt, max_tokens=50, temperature=0.8, top_k=40):
    """Generate text from a prompt."""
    token_ids = sp.encode(prompt)
    vocab_size = model.tokenEmbedding.weight.shape[0]
    token_ids = [t if t < vocab_size else 3 for t in token_ids]
    context = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            cropped = context[:, -model.maxContextLength:]
            logits = model(cropped)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            context = torch.cat((context, next_token), dim=1)
            
            if sp.eos_id() is not None and next_token.item() == sp.eos_id():
                break
    
    return sp.decode(generated)


def findCheckpoints(checkpoint_dir):
    """Find all epoch checkpoints and sort by epoch number."""
    pattern = os.path.join(checkpoint_dir, "model_epoch_*_step_0.pt")
    files = glob.glob(pattern)
    
    epoch_files = []
    for f in files:
        match = re.search(r'model_epoch_(\d+)_step_0\.pt', f)
        if match:
            epoch_num = int(match.group(1))
            epoch_files.append((epoch_num, f))
    
    epoch_files.sort(key=lambda x: x[0])
    return epoch_files


def plotTrainingCurves(epochs, losses, accuracies, perplexities, save_path):
    """Plot loss, accuracy, and perplexity curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Analysis", fontsize=16, fontweight='bold')
    
    # Loss curve
    axes[0].plot(epochs, losses, 'b-o', markersize=3, linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Val Loss Over Epochs')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=math.log(globalSettings.VOCAB_SIZE), color='r', 
                     linestyle='--', alpha=0.5, label=f'Random baseline (ln({globalSettings.VOCAB_SIZE})={math.log(globalSettings.VOCAB_SIZE):.1f})')
    axes[0].legend()
    
    # Accuracy curve
    axes[1].plot(epochs, [a * 100 for a in accuracies], 'g-o', markersize=3, linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('Val Accuracy Over Epochs')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=100/globalSettings.VOCAB_SIZE, color='r',
                     linestyle='--', alpha=0.5, label=f'Random baseline ({100/globalSettings.VOCAB_SIZE:.3f}%)')
    axes[1].legend()
    
    # Perplexity curve
    axes[2].plot(epochs, perplexities, 'r-o', markersize=3, linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Perplexity Over Epochs')
    axes[2].grid(True, alpha=0.3)
    if max(perplexities) > 1000:
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved training curves to: {save_path}")
    plt.close()


def main():
    # Force CPU to avoid conflicts with running training on GPU
    device = "cpu"
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    token_file = os.path.join(PROJECT_ROOT, "data", "tokenIds.npy")
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer", "data", "tokenizer.model")
    output_dir = os.path.join(PROJECT_ROOT, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("          MODEL ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    # --- Load tokenizer ---
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    # --- Find checkpoints ---
    epoch_files = findCheckpoints(checkpoint_dir)
    if not epoch_files:
        print("  [!] No epoch checkpoints found!")
        return
    
    print(f"\n  Found {len(epoch_files)} epoch checkpoints")
    
    # --- Evaluate each checkpoint ---
    # Sample evenly (evaluate every Nth checkpoint to keep it fast)
    if len(epoch_files) > 20:
        step = max(1, len(epoch_files) // 20)
        sampled = epoch_files[::step]
        # Always include first and last
        if epoch_files[0] not in sampled:
            sampled.insert(0, epoch_files[0])
        if epoch_files[-1] not in sampled:
            sampled.append(epoch_files[-1])
    else:
        sampled = epoch_files
    
    print(f"  Evaluating {len(sampled)} checkpoints...\n")
    
    epochs_list = []
    losses_list = []
    accs_list = []
    perps_list = []
    
    for epoch_num, ckpt_path in sampled:
        try:
            model = loadModel(ckpt_path, device)
            
            val_loss, val_acc, perplexity = evaluateModel(model, token_file, device)
            
            epochs_list.append(epoch_num)
            losses_list.append(val_loss)
            accs_list.append(val_acc)
            perps_list.append(perplexity)
            
            print(f"  Epoch {epoch_num:3d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Perplexity: {perplexity:.1f}")
            
            del model
            torch.cuda.empty_cache() if device == "cuda" else None
        except Exception as e:
            print(f"  Epoch {epoch_num:3d} | ERROR: {e}")
    
    # --- Plot curves ---
    if len(epochs_list) >= 2:
        print("\n  Generating plots...")
        plot_path = os.path.join(output_dir, "training_curves.png")
        plotTrainingCurves(epochs_list, losses_list, accs_list, perps_list, plot_path)
    
    # --- Test best model (last checkpoint) ---
    print("\n" + "=" * 60)
    print("          MODEL GENERATION TEST")
    print("=" * 60)
    
    best_epoch, best_path = epoch_files[-1]
    model = loadModel(best_path, device)
    
    test_prompts = [
        "hi",
        "hello",
        "startup",
        "good morning",
        "what",
        "the most important thing",
    ]
    
    print(f"\n  Using checkpoint: epoch {best_epoch}")
    print(f"  Temperature: 0.8 | Top-K: 40 | Max tokens: 50\n")
    
    for prompt in test_prompts:
        try:
            output = generateSample(model, sp, device, prompt, max_tokens=50, temperature=0.8)
            # Handle unicode for Windows console
            output = output.encode('ascii', errors='replace').decode('ascii')
            print(f"  Prompt: \"{prompt}\"")
            print(f"  Output: {output}")
            print(f"  {'-' * 50}")
        except Exception as e:
            print(f"  Prompt: \"{prompt}\"")
            print(f"  Output: [Error: {e}]")
            print(f"  {'-' * 50}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("          SUMMARY")
    print("=" * 60)
    if epochs_list:
        print(f"\n  Best Val Loss:      {min(losses_list):.4f} (Epoch {epochs_list[losses_list.index(min(losses_list))]})")
        print(f"  Best Val Accuracy:  {max(accs_list)*100:.2f}% (Epoch {epochs_list[accs_list.index(max(accs_list))]})")
        print(f"  Best Perplexity:    {min(perps_list):.1f} (Epoch {epochs_list[perps_list.index(min(perps_list))]})")
        print(f"  Final Val Loss:     {losses_list[-1]:.4f}")
        print(f"  Loss Improvement:   {losses_list[0]:.4f} -> {losses_list[-1]:.4f} ({((losses_list[0]-losses_list[-1])/losses_list[0])*100:.1f}% reduction)")
    
    print(f"\n  Analysis output saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
