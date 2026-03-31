import re

def parse_engagement_number(text):
    """
    Parses strings like '1.2K', '500', '2M' into integers.
    """
    text = text.upper().strip()
    
    # Extract only the leading numerical part and an optional K/M/B suffix
    match = re.match(r'^([\\d\\.]+)([KMB]?)', text)
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

sanitized_result = ['2.9k', '82', 'yourfIguy', 'Comment your predictions']
likes = 0
for token in sanitized_result:
    words = token.split()
    for word in words:
        # Force strictly strip out any attached punctuation/garbage from image artifacts
        clean_word = re.sub(r'[^\\d\\.KMBkmb]', '', word)
        print(f"Token: '{token}', Word: '{word}', Clean: '{clean_word}'")
        if clean_word and len(clean_word) < 10:
            parsed_val = parse_engagement_number(clean_word)
            print(f"  Parsed Val: {parsed_val}")
            # Ensure it's not a tiny number like '1' from a hashtag #f1
            if parsed_val > 10: 
                likes = parsed_val
                break
    if likes > 0:
        break

print("Final Likes:", likes)
