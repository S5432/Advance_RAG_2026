import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = "put gemini api key"

MODEL_NAME = "gemini-3.5-flash-lite"

API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL_NAME}:generateContent"
)

def test_gemini_api():

    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Hello Gemini! "
                            "Please introduce yourself and "
                            "confirm that you are working."
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200
        }
    }

    try:
        print("Connecting to Gemini API...")

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            generated_text = (
                result["candidates"][0]["content"]["parts"][0]["text"]
            )

            print("\nSUCCESS: Gemini API is working!")
            print(f"Model: {MODEL_NAME}")
            print("\nGemini Response:")
            print(generated_text)

        else:
            print("\nFAILED: Gemini API request unsuccessful.")
            print(f"Status Code: {response.status_code}")
            print(f"Error: {response.text}")

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out.")

    except requests.exceptions.RequestException as error:
        print(f"NETWORK ERROR: {error}")

    except (KeyError, IndexError, TypeError):
        print("ERROR: Unexpected response format.")
        print(response.text)


if __name__ == "__main__":
    test_gemini_api()