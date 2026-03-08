import os
import asyncio
import sys
from openai import AsyncOpenAI

# ==========================================
# Configuration
# ==========================================
DEEPSEEK_API_KEY = "sk-030f66a1f75540fc8328cbdddaffc5ec"

# ⚠️ 探测结果：标准接口 "https://api.deepseek.com" 是通的 (返回了 200 OK)
# 而那个特殊的 "...v3.2_speciale..." 返回 404。
# 这可能是因为特殊 endpoint 不支持 OpenAI 的 SDK 自动路径拼接，或者它其实是一个别名。
# 既然 https://api.deepseek.com 能用，我们就用它！
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Initialize Client
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

async def main():
    print("Initializing DeepSeek Stream Chat Test...")
    print(f"Base URL: {DEEPSEEK_BASE_URL}")
    print(f"Model: deepseek-reasoner")
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Please think carefully about who you are."}
    ]
    
    try:
        print("-" * 50)
        print("Sending request (Streaming)...")
        
        # 使用流式输出
        response = await client.chat.completions.create(
            model="deepseek-reasoner", 
            messages=messages,
            stream=True 
        )
        
        thinking_content = ""
        is_thinking = False
        
        async for chunk in response:
            delta = chunk.choices[0].delta
            
            # 尝试获取思考内容 (reasoning_content)
            r_content = getattr(delta, 'reasoning_content', None)
            
            if r_content:
                if not is_thinking:
                    sys.stdout.write("\n[Thinking Process]:\n")
                    is_thinking = True
                sys.stdout.write(r_content)
                sys.stdout.flush()
                thinking_content += r_content
            
            # 处理最终回答
            elif delta.content:
                if is_thinking:
                    sys.stdout.write("\n\n[Final Answer]:\n")
                    is_thinking = False
                
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                
        print("\n" + "-" * 50)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
