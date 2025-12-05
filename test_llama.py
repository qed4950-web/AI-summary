import os
from llama_cpp import Llama

# 모델 경로 설정 (사용자 환경에 존재하는 파일)
model_path = "models/gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

print(f"Loading model from {model_path}...")

try:
    # n_gpu_layers=-1로 설정하여 가능한 모든 레이어를 GPU로 로드 (Metal 가속 확인)
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=True
    )
    
    print("Model loaded successfully!")
    
    prompt = "Hello! How are you today?"
    print(f"\nTesting generation with prompt: '{prompt}'\n")
    
    output = llm(
        f"<|system|>\nYou are a helpful assistant.\n</s>\n<|user|>\n{prompt}\n</s>\n<|assistant|>\n", 
        max_tokens=50, 
        stop=["</s>"], 
        echo=True
    )
    
    print("\nGeneration Result:")
    print(output['choices'][0]['text'])
    print("\nTest completed successfully.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
