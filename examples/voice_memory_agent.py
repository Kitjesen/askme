import asyncio
import os
import sys
from openai import AsyncOpenAI

# ==========================================
# Configuration
# ==========================================
DEEPSEEK_API_KEY = "sk-030f66a1f75540fc8328cbdddaffc5ec"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Initialize Client
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ==========================================
# 1. Voice Input (Sherpa-ONNX ASR)
# ==========================================
def audio_to_text(audio_data):
    """Simulates Sherpa-ONNX Speech-to-Text."""
    print(f"[Ear] Hearing audio...")
    # TODO: Implement Sherpa-ONNX ASR here
    # Mock return for demo
    return "Hi, tell me a short joke."

# ==========================================
# 2. Reasoning (DeepSeek Chat Only)
# ==========================================
async def think(user_text):
    print(f"[Brain] Thinking about: '{user_text}'")
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_text}
            ],
            stream=True  # Enable streaming for real-time response
        )
        
        full_response = ""
        thinking_content = ""
        is_thinking = False
        
        print("\n--- Response Stream ---")
        async for chunk in response:
            delta = chunk.choices[0].delta
            
            # Handle thinking/reasoning content
            r_content = getattr(delta, 'reasoning_content', None)
            if r_content:
                if not is_thinking:
                    sys.stdout.write("\n[Thinking]:\n")
                    is_thinking = True
                sys.stdout.write(r_content)
                sys.stdout.flush()
                thinking_content += r_content
                
            # Handle final answer content
            elif delta.content:
                if is_thinking:
                    sys.stdout.write("\n\n[Answer]:\n")
                    is_thinking = False
                
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                full_response += delta.content
                
        print("\n-----------------------")
        return full_response

    except Exception as e:
        return f"Error communicating with DeepSeek: {str(e)}"

# ==========================================
# 3. Voice Output (Sherpa-ONNX TTS)
# ==========================================
def text_to_audio(text):
    """Simulates Sherpa-ONNX Text-to-Speech."""
    print(f"\n[Mouth] Speaking: '{text}'")
    # TODO: Implement Sherpa-ONNX TTS here

# ==========================================
# Main Loop
# ==========================================
async def main():
    print("--- Voice Agent (DeepSeek Integrated) ---")
    
    # 1. Simulate Input
    user_audio = b"..." 
    user_text = audio_to_text(user_audio)
    print(f"[User Said]: {user_text}")
    
    # 2. Process (Chat Only)
    response_text = await think(user_text)
    
    # 3. Simulate Output
    if response_text:
        text_to_audio(response_text)

if __name__ == "__main__":
    asyncio.run(main())
