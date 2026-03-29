import sys
import os
import subprocess

def printInstructions():
    print("=" * 60)
    print("         GGUF CONVERSION INSTRUCTIONS")
    print("=" * 60)
    print("To convert our PyTorch model to GGUF format for Ollama, we will use")
    print("the official llama.cpp conversion tools.\n")
    
    print("STEP 1: Clone llama.cpp")
    print("  Run: git clone https://github.com/ggerganov/llama.cpp.git\n")
    
    print("STEP 2: Install the conversion requirements")
    print("  Run: cd llama.cpp")
    print("  Run: pip install -r requirements.txt\n")
    
    print("STEP 3: Run the conversion script")
    print("  Run: python convert.py ../checkpoints/ --outfile ../export/dumbotron.gguf --outtype q8_0\n")
    
    print("STEP 4: Create an Ollama Modelfile")
    print("  Create a file named `Modelfile` containing:")
    print("      FROM ./export/dumbotron.gguf")
    print("      TEMPLATE \"{{ .Prompt }}\"")
    print("  Run: ollama create dumbotron -f Modelfile\n")
    
    print("STEP 5: Run your Tiny LLM!")
    print("  Run: ollama run dumbotron")
    print("=" * 60)

if __name__ == "__main__":
    printInstructions()
