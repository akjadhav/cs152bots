import json
import math
import pathlib
import requests
import logging
from pydantic import BaseModel, Field


# --------- Schema for the response format ---------
class SextortionClassification(BaseModel):
    is_sextortion: bool = Field(
        description="True if the message is classified as sextortion, False otherwise."
    )


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
TOGETHER_API_URL = "https://api.together.ai/v1/completions"

# Together API endpoint and model specification
# Replace with the correct endpoint if Together.ai updates their API
# ────────────────────── TOKEN & LOGGING ──────────────────────────────
TOKENS_FILE = pathlib.Path(__file__).with_name("tokens.json")
try:
    TOKEN: str = json.loads(TOKENS_FILE.read_text())["discord"]
    TOGETHER_TOKEN: str = json.loads(TOKENS_FILE.read_text())["together"]
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        f"❌ Could not read Discord or Together token from {TOKENS_FILE}"
    ) from exc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mod-bot")

# -----------------------------
# Few-Shot Prompt Template
# -----------------------------

FEW_SHOT_PROMPT = """
You are a content-safety classifier. Given a user message, you must decide whether it describes or attempts sextortion.
Sextortion is when someone threatens to share private sexual/explicit images/videos unless the victim does something (e.g., pay money, send more images).

Classify the message as SEXTORTION or SAFE. Respond with a JSON object with a single key `is_sextortion`:

Example 1:
Message: "Hey, if you don’t send me nudes, I’ll post those pictures of you I saved."
{{is_sextortion: true}}

Example 2:
Message: "Can I see your new profile pic?"
{{is_sextortion: false}}

Example 3:
Message: "Give me your password or I’ll leak your private photos."
{{is_sextortion: true}}

Example 4:
Message: "Let’s trade memes and funny videos!"
{{is_sextortion: false}}

Now classify the following message:
>> "{message}"
"""

# -----------------------------
# Classification Function
# -----------------------------


def is_sextortion(message: str) -> (bool, float):
    """
    Returns True if the message is likely sextortion, False otherwise.
    Uses a few-shot prompt on Together.ai Llama-2-7B.
    """
    # Fill in the prompt
    message = message.replace('"', '\\"')  # Escape quotes for JSON
    prompt = FEW_SHOT_PROMPT.format(message=message)

    headers = {
        "Authorization": f"Bearer {TOGETHER_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 20,  # Increased to ensure complete JSON
        "temperature": 0.0,  # deterministic output
        "logprobs": 2,  # Return top‐2 token logprobs
        "response_format": {
            "type": "json_object",
            "schema": SextortionClassification.model_json_schema(),
        },
        # Removed stop parameter to avoid truncation
    }

    try:
        resp = requests.post(
            TOGETHER_API_URL, headers=headers, json=payload, timeout=15
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return (False, 0.0)

    data = resp.json()

    try:
        completion = data["choices"][0]["text"].strip()
        
        # Try to parse JSON
        try:
            parsed_json = json.loads(completion)
            is_sextortion = parsed_json["is_sextortion"]
        except json.JSONDecodeError as e:
            # Check if it's just missing the closing brace
            if completion.startswith('{"is_sextortion":') and not completion.endswith('}'):
                try:
                    fixed_completion = completion + '}'
                    parsed_json = json.loads(fixed_completion)
                    is_sextortion = parsed_json["is_sextortion"]
                except:
                    # Fallback to text parsing
                    completion_lower = completion.lower()
                    if "true" in completion_lower:
                        is_sextortion = True
                    elif "false" in completion_lower:
                        is_sextortion = False
                    else:
                        return (False, 0.0)
            else:
                # Try to extract boolean from text
                completion_lower = completion.lower()
                if "true" in completion_lower:
                    is_sextortion = True
                elif "false" in completion_lower:
                    is_sextortion = False
                else:
                    return (False, 0.0)

        # Extract confidence from logprobs
        confidence = 0.0
        confidence_false = 0.0
        
        try:
            log_probs = data["choices"][0]["logprobs"]["top_logprobs"]
            for log_key, value in log_probs[-2].items():
                if "true" in log_key:
                    confidence = get_probability(value)
                elif "false" in log_key:
                    confidence_false = get_probability(value)

            confidence = max(confidence, confidence_false) if confidence_false > 0 else confidence
        except (KeyError, IndexError) as e:
            confidence = 50.0  # Default confidence

    except (KeyError, IndexError) as e:
        return (False, 0.0)

    return (is_sextortion, confidence)


# ------- Helper Functions -------
def get_probability(logprob: float) -> float:
    return round(math.exp(logprob) * 100, 2)


# -----------------------------
# Quick CLI for Testing
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a given message as SEXTORTION or SAFE using Together.ai + Llama-2-7B."
    )
    parser.add_argument(
        "message",
        type=str,
        help="The user message to classify (wrap in quotes if it has spaces).",
    )
    args = parser.parse_args()

    is_sextortion_result = is_sextortion(args.message)
    print(
        f"Classification: {is_sextortion_result[0]}, Confidence: {is_sextortion_result[1]}%"
    )
