"""
Azure OpenAI Model Test Script
==============================
Test your GPT-4o and text-embedding-3-small deployments
"""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

# ============================================
# Configuration is read from your .env file
# (copy .env.example to .env and fill it in).
# ============================================
load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def test_gpt4o():
    """Test GPT-4o deployment."""
    print("\n" + "=" * 50)
    print("Testing GPT-4o...")
    print("=" * 50)
    
    try:
        client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'Hello, I am working!' in exactly 5 words."}],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ GPT-4o SUCCESS!")
        print(f"   Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ GPT-4o FAILED!")
        print(f"   Error: {e}")
        return False


def test_embedding():
    """Test text-embedding-3-small deployment."""
    print("\n" + "=" * 50)
    print("Testing text-embedding-3-small...")
    print("=" * 50)
    
    try:
        client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION
        )
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="Hello world test"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ Embedding SUCCESS!")
        print(f"   Vector length: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding FAILED!")
        print(f"   Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AZURE OPENAI MODEL TEST")
    print("=" * 50)
    
    gpt_ok = test_gpt4o()
    embed_ok = test_embedding()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"GPT-4o:                 {'✅ WORKING' if gpt_ok else '❌ NOT WORKING'}")
    print(f"text-embedding-3-small: {'✅ WORKING' if embed_ok else '❌ NOT WORKING'}")
    print("=" * 50)
    
    if gpt_ok and embed_ok:
        print("\n🎉 Both models working! Update your .env and run main.py")
        print("\nYour .env should be:")
        print("-" * 50)
        print(f'AZURE_OPENAI_ENDPOINT={ENDPOINT}')
        print('AZURE_OPENAI_API_KEY=<your-key>')
        print('AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o')
        print('AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small')
        print('AZURE_OPENAI_API_VERSION=2024-12-01-preview')
    else:
        print("\n⚠️  Fix the failing model(s) before running main.py")