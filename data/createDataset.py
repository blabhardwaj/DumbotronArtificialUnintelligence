import random

def generateConversationalData(filepath, num_samples=1000):
    """
    Generates a simple synthetic dataset of conversational greetings.
    By repeating these patterns, the tiny SLM will learn to associate 
    greetings with appropriate responses.
    """
    
    inputs = [
        "hi", "hello", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "hi there", "hello there"
    ]
    
    outputs = [
        "Hello! How can I help you today?",
        "Hi there! What's on your mind?",
        "Greetings! I am ready to assist you.",
        "Hello! I am Dumbotron, your loyal AI.",
        "Hey! How are you doing?",
        "Good day to you! How may I serve you?",
        "Hi! It is a pleasure to meet you."
    ]
    
    with open(filepath, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            # Just putting the input on one line and the response on the next line
            q = random.choice(inputs)
            a = random.choice(outputs)
            
            conversation = f"{q}\n{a}</s>\n"
            f.write(conversation)
            
    print(f"Successfully generated {num_samples} conversational pairs at {filepath}")

if __name__ == "__main__":
    generateConversationalData("data/greetings.txt", 2000)
    
    # We will also append this to the Shakespeare file so the model learns both
    print("Reading greetings...")
    with open("data/greetings.txt", "r", encoding="utf-8") as f:
        greetings_text = f.read()
        
    print("Appending to shakespear.txt...")
    with open("data/shakespear.txt", "a", encoding="utf-8") as f:
        f.write("\n" + greetings_text)
        
    print("Done! The dataset now contains conversational examples.")
