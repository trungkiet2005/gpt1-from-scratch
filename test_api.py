import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL for your Hugging Face Space
BASE_URL = "https://huynhtrungkiet09032005-gpt1.hf.space"

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        logger.info(f"\nHealth Check Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")

def test_api_info():
    """Test the API info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api")
        data = response.json()
        logger.info(f"\nAPI Info Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Error in API info: {str(e)}")

def test_generate_poetry():
    """Test the poetry generation endpoint"""
    try:
        test_prompt = "mặt trời mọc"
        payload = {
            "prompt": test_prompt,
            "max_length": 100,
            "temperature": 1.0
        }
        
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        data = response.json()
        logger.info(f"\nGenerate Poetry Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Error in generate poetry: {str(e)}")

def test_generate_creative_poetry():
    """Test the creative poetry generation endpoint"""
    try:
        test_prompt = "mặt trăng lặn"
        payload = {
            "prompt": test_prompt,
            "max_length": 150,
            "temperature": 1.5
        }
        
        response = requests.post(
            f"{BASE_URL}/generate_creative",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        data = response.json()
        logger.info(f"\nGenerate Creative Poetry Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Error in generate creative poetry: {str(e)}")

if __name__ == "__main__":
    print("Running tests...")
    test_health_check()
    test_api_info()
    test_generate_poetry()
    test_generate_creative_poetry()
    print("Tests completed!") 