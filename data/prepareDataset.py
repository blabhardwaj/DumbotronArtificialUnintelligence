import numpy as np

def prepareDataset():
    # Load the saved token IDs
    tokens = np.load("data/tokenIds.npy")
    
    # Split into input and target (next-token prediction)
    input_data  = tokens[:-1]   # feed this to the model
    target_data = tokens[1:]    # model predicts this
    
    print("Input shape:", input_data.shape)
    print("Target shape:", target_data.shape)
    print("First 5 input tokens:", input_data[:5])
    print("First 5 target tokens:", target_data[:5])
    
    # Save them
    np.save("data/input.npy", input_data)
    np.save("data/target.npy", target_data)
    
    print("Saved input.npy and target.npy")

if __name__ == "__main__":
    prepareDataset()
