from llama_cpp import Llama
import os

# Configuration
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Updated to Mistral 7B
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

def download_model():
    """Download the model if it doesn't exist."""
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found. Please download the Mistral 7B model from:\n{MODEL_URL}")
        print("\nAnd place it in the 'models' directory as 'mistral-7b-instruct-v0.1.Q4_K_M.gguf'")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

def initialize_model():
    download_model()
    
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,        # Large context window
        n_batch=512,       # Increased batch size for better performance
        n_threads=4,       # Adjust based on your CPU
        n_gpu_layers=0     # CPU only, increase if you have GPU
    )

def generate_response(llm, prompt: str, max_tokens: int = 256) -> str:
    # Mistral specific instruction format
    formatted_prompt = f"""<s>[INST] You are Mistral, a highly knowledgeable AI assistant. Please provide accurate and concise answers.

{prompt} [/INST]"""
    
    output = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=0.1,    # Low temperature for factual responses
        top_p=0.9,
        repeat_penalty=1.2,
        echo=False,
        stop=["</s>", "[INST]"]  # Mistral's stop tokens
    )
    
    response = output["choices"][0]["text"].strip()
    return response

def main():
    print("\nInitializing Mistral-7B model... (this may take a moment)")
    try:
        llm = initialize_model()
        print("\nModel loaded successfully!")
        
        user_input = input("\nEnter your question: ").strip()
        if user_input:
            print("\nMistral: ", end='', flush=True)
            response = generate_response(llm, user_input)
            print(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
