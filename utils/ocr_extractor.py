import easyocr
import re
import numpy as np
import cv2

# Initialize the reader once globally (this downloads the models on first run if needed)
reader = easyocr.Reader(['en'], gpu=False)

def parse_engagement_number(text):
    """
    Parses strings like '1.2K', '500', '2M' into integers.
    """
    text = text.upper()
    text = text.upper().strip()
    
    # Extract only the leading numerical part and an optional K/M/B suffix
    match = re.match(r'^([\d\.]+)([KMB]?)', text)
    if not match:
        return 0
        
    num_str, suffix = match.groups()
    try:
        val = float(num_str)
        if suffix == 'K': return int(val * 1000)
        elif suffix == 'M': return int(val * 1000000)
        elif suffix == 'B': return int(val * 1000000000)
        else: return int(val)
    except:
        return 0

def extract_features_from_image(image_bytes: bytes):
    """
    Runs OCR on the given image bytes and attempts to extract:
    - Post text
    - Hashtags
    - Likes
    - Watch time (estimated from views)
    """
    # Convert bytes to numpy array for cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run OCR
    result = reader.readtext(img, detail=0)
    
    # Sanitize the result to remove non-ascii characters (like emojis or block chars \u2588)
    # This prevents UnicodeEncodeErrors downstream when FastAPI/Pydantic processes the string
    sanitized_result = [text.encode('ascii', 'ignore').decode('ascii') for text in result]
    print(f"DEBUG OCR TOKENS: {sanitized_result}", file=__import__('sys').stderr)
    
    full_text = " ".join(sanitized_result)
    
    # 1. Extract hashtags
    # Sometimes OCR merges hashtags or drops the #. We'll look for #\w+ 
    hashtags = re.findall(r'#\w+', full_text)
    
    # Remove hashtags from the main text body to clean up the caption
    caption_text = full_text
    for tag in hashtags:
        caption_text = caption_text.replace(tag, '')
    caption_text = caption_text.strip()
    
    print(f"DEBUG FULL TEXT FOR LIKES: {full_text}", file=__import__('sys').stderr)
    
    # 2. Extract Likes
    likes = 0
    # First, try to find ALL explicit "likes" labels and take the maximum
    all_explicit_likes = re.findall(r'([\d\.\,KMBkmb]+)\s*likes?', full_text, flags=re.IGNORECASE)
    if all_explicit_likes:
        parsed_expl = [parse_engagement_number(x) for x in all_explicit_likes]
        likes = max(parsed_expl) if parsed_expl else 0
    
    # If no explicit label, try a heuristic: in many UIs (like the screenshot), the likes are the *first* number in the top left.
    if likes == 0:
        # EasyOCR sometimes returns multiple words in one token (e.g. "2.9k 82 yourf1guy")
        # Split tokens by space and find the very first valid engagement number.
        for token in sanitized_result:
            words = token.split()
            for word in words:
                # Force strictly strip out any attached punctuation/garbage from image artifacts
                clean_word = re.sub(r'[^\d\.KMBkmb]', '', word)
                if clean_word and len(clean_word) < 10:
                    parsed_val = parse_engagement_number(clean_word)
                    # Ensure it's not a tiny number like '1' from a hashtag #f1
                    if parsed_val > 10: 
                        likes = parsed_val
                        break
            if likes > 0:
                break
            
    return {
        "text": caption_text,
        "hashtags": hashtags,
        "likes": likes,
        "raw_ocr": full_text
    }
