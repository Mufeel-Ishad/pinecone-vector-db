"""
Helper script to set up environment variables.
This script helps users create a .env file with their API keys.
"""

import os

def setup_env():
    """Create .env file with API keys."""
    env_file = ".env"
    
    if os.path.exists(env_file):
        response = input(".env file already exists. Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
            print("Setup cancelled.")
            return
    
    print("\n" + "="*60)
    print("Pinecone Vector Database - Environment Setup")
    print("="*60)
    print("\nPlease provide your API keys:")
    print("(You can get these from https://www.pinecone.io/ and https://platform.openai.com/)\n")
    
    pinecone_key = input("Enter your Pinecone API Key: ").strip()
    openai_key = input("Enter your OpenAI API Key: ").strip()
    
    if not pinecone_key or not openai_key:
        print("\nError: Both API keys are required!")
        return
    
    # Write to .env file
    with open(env_file, "w") as f:
        f.write(f"# Pinecone API Key\n")
        f.write(f"PINECONE_API_KEY={pinecone_key}\n\n")
        f.write(f"# OpenAI API Key\n")
        f.write(f"OPENAI_API_KEY={openai_key}\n")
    
    print(f"\n✅ Environment file created successfully at {env_file}")
    print("You can now run the main script or Streamlit app!")


if __name__ == "__main__":
    setup_env()

