import os
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Get API tokens from environment
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

MAX_CHUNK_CHARS = 1000

# Language code mapping for gpt-oss-model-120b-1:0
LANGUAGE_CODES = {
    "English": "en",
    "Polish": "pl",
    "Bulgarian": "bg",
    "Portuguese": "pt",
    "Czech": "cs",
    "Romanian": "ro",
    "Danish": "da",
    "Russian": "ru",
    "German": "de",
    "Slovak": "sk",
    "Greek": "el",
    "Slovenian": "sl",
    "Spanish": "es",
    "Serbian": "sr",
    "Estonian": "et",
    "Swedish": "sv",
    "Finnish": "fi",
    "Turkish": "tr",
    "French": "fr",
    "Vietnamese": "vi",
    "Croatian": "hr",
    "Irish": "ga",
    "Hungarian": "hu",
    "Maltese": "mt",
    "Indonesian": "id",
    "Italian": "it",
    "Icelandic": "is",
    "Chinese": "zh",
    "Kazakh": "kk",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Japanese": "ja",
    "Dutch": "nl",
    "Korean": "ko",
    "Norwegian": "no",
    "Thai": "th",
    "Arabic": "ar",
    "Malay": "ms",
}

# AWS Bedrock model configuration - gpt-oss-model-120b only
AWS_MODEL_ID = "openai.gpt-oss-120b-1:0"


def get_language_code(language_name: str) -> str:
    """Get the language code for a given language name."""
    return LANGUAGE_CODES.get(language_name, "fr")


def _translate_via_aws_bedrock(text: str, target_lang: str = "French") -> str:
    """
    Translate text using AWS Bedrock.
    
    Args:
        text: English text to translate
        target_lang: Target language name (e.g., "French", "German")
    
    Returns:
        Translated text
    """
    try:
        # Create Bedrock Runtime client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        # Get language code for the target language
        lang_code = get_language_code(target_lang)
        
        # Improved translation prompt - very explicit
        prompt = f"""Task: Translate the following medical document text from English to {target_lang}.

IMPORTANT: 
- You MUST translate the text, do NOT copy it
- Output ONLY the {target_lang} translation, nothing else
- Do NOT include any explanations, notes, or original text

Text to translate:
{text}

{target_lang} translation:"""
        
        print(f"=== AWS TRANSLATION DEBUG ===")
        print(f"Model: {AWS_MODEL_ID}")
        print(f"Region: {AWS_REGION}")
        print(f"Target: {target_lang} ({lang_code})")
        print(f"Input text: {text[:100]}...")

        # Request payload for OpenAI model on Bedrock (chat format)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.9
        }

        # Invoke the model
        print("Invoking AWS Bedrock...")
        response = bedrock_runtime.invoke_model(
            modelId=AWS_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        
        print(f"Response received, status: OK")

        # Parse the response
        response_body = json.loads(response['body'].read())
        
        print(f"=== AWS RESPONSE ===", flush=True)
        print(f"Response keys: {response_body.keys()}", flush=True)
        print(f"Full response: {json.dumps(response_body)[:1000]}", flush=True)
        print(f"Full response: {json.dumps(response_body)[:500]}...")
        
        # Extract the translated text from the response
        # OpenAI format returns 'choices' array with 'message' object
        if 'choices' in response_body and len(response_body['choices']) > 0:
            translated_text = response_body['choices'][0].get('message', {}).get('content', '')
            translated_text = translated_text.strip()
        elif 'completion' in response_body:
            # Fallback: check other possible response formats
            translated_text = response_body.get('completion', '').strip()
        elif 'text' in response_body:
            translated_text = response_body.get('text', '').strip()
        else:
            print(f"Unexpected response format: {response_body}")
            return None
        
        print(f"Raw translation: {translated_text[:200]}...")
        
        # Clean up reasoning/thinking tags if present
        import re
        # Remove <reasoning>...</reasoning> tags
        translated_text = re.sub(r'<reasoning>.*?</reasoning>', '', translated_text, flags=re.DOTALL)
        # Remove <thinking>...</thinking> tags
        translated_text = re.sub(r'<thinking>.*?</thinking>', '', translated_text, flags=re.DOTALL)
        # Remove any other XML-like tags
        translated_text = re.sub(r'<[^>]+>', '', translated_text)
        translated_text = translated_text.strip()
        
        # Check if the response is just the original English text (model didn't translate)
        if translated_text.lower() == text.lower():
            print("ERROR: Model returned original text without translation!")
            return None
        
        # If the response starts with the input text, remove it
        if text.lower() in translated_text.lower() and len(translated_text) > len(text) * 1.5:
            # Find where the actual translation starts
            lines = translated_text.split('\n')
            if len(lines) > 1:
                # Skip the first line if it contains English
                for i, line in enumerate(lines):
                    if line.strip().lower() != text.lower():
                        translated_text = '\n'.join(lines[i:])
                        break
        
        # Remove the target language label if present at the start
        for prefix in [f"{target_lang}:", f"{target_lang} translation:", f"{lang_code}:"]:
            if translated_text.lower().startswith(prefix.lower()):
                translated_text = translated_text[len(prefix):].strip()
                break
              
        print(f"Final translation: {translated_text[:200]}...")
        return translated_text if translated_text else None

    except ClientError as e:
        print(f"AWS Bedrock error: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        print(f"Translation exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def _translate_with_fallback(text: str, target_lang: str = "French") -> str:
    """
    Translate using AWS Bedrock gpt-oss-120b model.
    Returns the translated text or raises an exception on failure.
    """
    import sys    
    # Try AWS Bedrock
    print(f"=== AWS TRANSLATION DEBUG ===", flush=True)
    print(f"Translating to {target_lang}...", flush=True)
    sys.stdout.flush()
    
    translated = _translate_via_aws_bedrock(text, target_lang)
    
    if translated:
        print(f"AWS translation successful: {translated[:100]}...", flush=True)
        return translated
    
    # Translation failed - raise exception so we can see what's happening
    print(f"AWS translation FAILED for text: {text[:50]}...", flush=True)
    raise Exception(f"AWS translation failed! Target: {target_lang}, Input: {text[:100]}...")


def _chunk_text(text: str) -> list[str]:
    """Split text into chunks at sentence boundaries within char limit."""
    sentences = text.replace("\n", " \n ").split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 2 <= MAX_CHUNK_CHARS:
            current += sentence + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def translate_chunks(text: str, target_lang: str = "French"):
    """Generator yielding (translated_chunk, step, total) for each chunk."""
    chunks = _chunk_text(text)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        translated = _translate_with_fallback(chunk, target_lang)
        
        # If translation failed, raise an error so the user knows
        if translated is None:
            raise Exception(f"Translation to {target_lang} failed. Please check AWS credentials and model availability.")
        
        yield translated, i + 1, total


def translate_text(text: str, target_lang: str = "French") -> str:
    """Translate English text to target language using available translation API."""
    result_chunks = []
    for chunk, _, _ in translate_chunks(text, target_lang):
        result_chunks.append(chunk)
    return " ".join(result_chunks)


def translate_segments(segments: list[dict], target_lang: str = "French"):
    """
    Translate document segments to target language using available translation API.
    
    Args:
        segments: List of dicts with 'id' and 'text' keys
        target_lang: Target language name (e.g., "French", "German")
    
    Yields:
        dicts with 'id', 'type', 'text', and 'translated_text' keys
    """
    for seg in segments:
        text = seg.get("text", "")
        if not text:
            yield {"id": seg.get("id"), "translated_text": "", "type": seg.get("type", "p")}
            continue
        
        # Translate the text in chunks
        chunks = _chunk_text(text)
        translated_parts = []
        
        for chunk in chunks:
            translated = _translate_with_fallback(chunk, target_lang)
            
            # If translation failed, raise an exception
            if translated is None:
                raise Exception(f"Translation to {target_lang} failed. Please check AWS credentials and model availability.")
            
            translated_parts.append(translated)
        
        yield {
            "id": seg.get("id"),
            "type": seg.get("type", "p"),
            "text": text,
            "translated_text": " ".join(translated_parts)
        }
