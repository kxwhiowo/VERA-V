import json
import re
from typing import List, Tuple, Union, Any

def sanitize_string_output(value: Any, fallback: str = "") -> str:
    """
    Ensures the output is always a clean string, handling various input types.
    
    Args:
        value: Input value of any type
        fallback: Fallback string to use if conversion fails
        
    Returns:
        A clean string representation
    """
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, dict):
        # If it's a dict, try to extract meaningful content
        if "image prompt" in value:
            return str(value["image prompt"]).strip()
        elif "text prompt" in value:
            return str(value["text prompt"]).strip()
        else:
            # Convert dict to string as last resort
            return str(value).strip()
    elif isinstance(value, list):
        # If it's a list, join elements or take first element
        if value:
            return sanitize_string_output(value[0], fallback)
        else:
            return fallback
    elif value is None:
        return fallback
    else:
        # Convert any other type to string
        return str(value).strip()

def clean_json_string(json_str: str) -> str:
    """
    Cleans and fixes common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Fix missing closing braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    # Fix missing closing brackets
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    # Fix unescaped quotes in strings (basic attempt)
    # This is a simple fix - more sophisticated parsing might be needed
    json_str = re.sub(r'(?<!\\)"(?=.*".*:)', '\\"', json_str)
    
    return json_str

def extract_json_content(text: str) -> dict:
    """
    Extracts JSON content from text using multiple strategies.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dictionary or empty dict if parsing fails
    """
    # Strategy 1: Look for complete JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            cleaned_json = clean_json_string(json_str)
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Manual extraction using regex
    result = {}
    
    # Extract image prompt
    img_patterns = [
        r'"image prompt"\s*:\s*"(.*?)"(?:\s*,|\s*})',
        r'"image prompt"\s*:\s*"(.*)',
        r'image prompt["\']?\s*:\s*["\']?(.*?)["\']?(?:\s*,|\s*})'
    ]
    
    for pattern in img_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["image prompt"] = match.group(1).strip().rstrip('",}')
            break
    
    # Extract text prompt
    txt_patterns = [
        r'"text prompt"\s*:\s*"(.*?)"(?:\s*})',
        r'"text prompt"\s*:\s*"(.*)',
        r'text prompt["\']?\s*:\s*["\']?(.*?)["\']?(?:\s*})'
    ]
    
    for pattern in txt_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["text prompt"] = match.group(1).strip().rstrip('",}')
            break
    
    return result

def parse_adversarial_json_output(
    adversarial_queries: List[Union[str, List[str], dict, Any]], 
    original_prompt: str
) -> Tuple[List[str], List[str]]:
    """
    Robustly parses adversarial queries, handling various input formats and malformed JSON.
    
    Args:
        adversarial_queries: A list that may contain strings, lists, dicts, or other formats
        original_prompt: The original prompt for fallback purposes
        
    Returns:
        A tuple of (image_prompts, text_prompts) where both are guaranteed to be lists of strings
    """
    image_prompts = []
    text_prompts = []
    
    if not adversarial_queries:
        print("Warning: Empty adversarial_queries provided")
        return [original_prompt], ["Please analyze the image and provide detailed information."]
    
    for i, query_item in enumerate(adversarial_queries):
        try:
            # Step 1: Normalize input to string
            if isinstance(query_item, dict):
                # Handle dict input directly
                image_prompt = sanitize_string_output(
                    query_item.get("image prompt", original_prompt), 
                    original_prompt
                )
                text_prompt = sanitize_string_output(
                    query_item.get("text prompt", "Please analyze the image and provide detailed information."), 
                    "Please analyze the image and provide detailed information."
                )
                
                image_prompts.append(image_prompt)
                text_prompts.append(text_prompt)
                continue
                
            elif isinstance(query_item, list):
                if not query_item:
                    query_str = original_prompt
                else:
                    query_str = sanitize_string_output(query_item[0], original_prompt)
            else:
                query_str = sanitize_string_output(query_item, original_prompt)
            
            # Step 2: Extract JSON content
            parsed_content = extract_json_content(query_str)
            
            if parsed_content:
                image_prompt = sanitize_string_output(
                    parsed_content.get("image prompt", original_prompt), 
                    original_prompt
                )
                text_prompt = sanitize_string_output(
                    parsed_content.get("text prompt", query_str), 
                    "Please analyze the image and provide detailed information."
                )
            else:
                # Fallback: treat entire string as content
                print(f"Warning: No JSON structure found in query {i}, using as fallback")
                image_prompt = original_prompt
                text_prompt = query_str if query_str else "Please analyze the image and provide detailed information."
            
            # Step 3: Final validation and sanitization
            image_prompt = sanitize_string_output(image_prompt, original_prompt)
            text_prompt = sanitize_string_output(text_prompt, "Please analyze the image and provide detailed information.")
            
            # Ensure prompts are not empty
            if not image_prompt:
                image_prompt = original_prompt
            if not text_prompt:
                text_prompt = "Please analyze the image and provide detailed information."
                
            image_prompts.append(image_prompt)
            text_prompts.append(text_prompt)
            
        except Exception as e:
            print(f"Warning: Error processing query item {i}: {e}")
            print(f"Query item type: {type(query_item)}, content: {str(query_item)[:100]}...")
            
            # Emergency fallback
            image_prompts.append(original_prompt)
            text_prompts.append("Please analyze the image and provide detailed information.")
    
    # Final safety check
    if not image_prompts:
        image_prompts = [original_prompt]
    if not text_prompts:
        text_prompts = ["Please analyze the image and provide detailed information."]
    
    print(f"Parsed {len(image_prompts)} image prompts and {len(text_prompts)} text prompts")
    return image_prompts, text_prompts

def validate_prompts(image_prompts: List[str], text_prompts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Additional validation layer to ensure all prompts are valid strings.
    
    Args:
        image_prompts: List of image prompts
        text_prompts: List of text prompts
        
    Returns:
        Validated and cleaned prompt lists
    """
    validated_image = []
    validated_text = []
    
    for i, (img_prompt, txt_prompt) in enumerate(zip(image_prompts, text_prompts)):
        # Ensure both are strings
        img_clean = sanitize_string_output(img_prompt, f"Generate an image for prompt {i}")
        txt_clean = sanitize_string_output(txt_prompt, f"Please analyze this image {i}")
        
        validated_image.append(img_clean)
        validated_text.append(txt_clean)
    
    return validated_image, validated_text

def clean_vlm_response(response: str) -> str:
    """
    Cleans VLM responses by removing unwanted prefixes and artifacts.
    
    Args:
        response: Raw VLM response that may contain unwanted prefixes
        
    Returns:
        Cleaned response with prefixes removed
    """
    if not response or not isinstance(response, str):
        return str(response) if response else ""
    
    # Only remove the instruction part before "ASSISTANT:" but keep the actual response
    prefixes_to_remove = [
        # Remove everything before "ASSISTANT:" including the instruction
        r"^.*?ASSISTANT:\s*",
        r"^.*?Assistant:\s*",
    ]
    
    cleaned_response = response.strip()
    
    # Remove each prefix pattern
    for prefix_pattern in prefixes_to_remove:
        cleaned_response = re.sub(prefix_pattern, "", cleaned_response, flags=re.IGNORECASE)
        cleaned_response = cleaned_response.strip()
    
    # Remove any leading colons, dashes, or other artifacts
    cleaned_response = re.sub(r"^[:\-\s]+", "", cleaned_response)
    
    # If the response becomes empty or too short after cleaning, return original
    if len(cleaned_response.strip()) < 10:
        return response
    
    return cleaned_response.strip()

def clean_vlm_responses(responses: List[str]) -> List[str]:
    """
    Cleans a batch of VLM responses.
    
    Args:
        responses: List of raw VLM responses
        
    Returns:
        List of cleaned responses
    """
    return [clean_vlm_response(response) for response in responses]