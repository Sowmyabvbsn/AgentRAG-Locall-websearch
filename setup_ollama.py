#!/usr/bin/env python3
"""
Setup script to ensure Ollama is properly configured with required models
"""

import subprocess
import sys
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Start Ollama service"""
    try:
        logger.info("Starting Ollama service...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Give it time to start
        return check_ollama_running()
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False

def pull_model(model_name):
    """Pull a model using Ollama"""
    try:
        logger.info(f"Pulling {model_name} model...")
        # Fix Windows encoding issues
        result = subprocess.run(
            ["ollama", "pull", model_name], 
            capture_output=True, 
            text=True, 
            timeout=600,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters
        )
        if result.returncode == 0:
            logger.info(f"✅ {model_name} model ready")
            return True
        else:
            logger.error(f"Failed to pull {model_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout pulling {model_name}")
        return False
    except Exception as e:
        logger.error(f"Error pulling {model_name}: {e}")
        return False

def setup_ollama():
    """Main setup function"""
    logger.info("🚀 Setting up Ollama for Enhanced RAG System")
    
    # Check if Ollama is installed
    try:
        subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            check=True,
            encoding='utf-8',
            errors='replace'
        )
        logger.info("✅ Ollama is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ Ollama is not installed. Please install it from https://ollama.ai")
        return False
    
    # Check if Ollama is running, start if not
    if not check_ollama_running():
        logger.info("Ollama is not running, attempting to start...")
        if not start_ollama():
            logger.error("❌ Could not start Ollama. Please run 'ollama serve' manually")
            return False
    else:
        logger.info("✅ Ollama is running")
    
    # Pull required models
    models_to_pull = ["llama2", "nomic-embed-text"]
    
    for model in models_to_pull:
        if not pull_model(model):
            logger.error(f"❌ Failed to pull {model}")
            return False
    
    logger.info("🎉 Ollama setup complete! You can now run the RAG system.")
    return True

if __name__ == "__main__":
    success = setup_ollama()
    sys.exit(0 if success else 1)