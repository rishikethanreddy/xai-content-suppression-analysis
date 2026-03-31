import re

text = "18+ sensitive content hub that's SO dangerous 34w carlitobonito2.0 Holy crap he's fortunate not to be in pieces 34w 7 likes Reply 744 likes 5, 2025 July #explosion #fireworks #buildings"

# Test explicit regex
likes_match = re.search(r'([\d\.\,KMBkmb]+)\s*likes?', text, flags=re.IGNORECASE)
if likes_match:
    print("Explicit match:", likes_match.group(1))
else:
    print("Explicit match FAILED")
    
# Find all explicit explicit likes and take the max
all_explicit = re.findall(r'([\d\.\,KMBkmb]+)\s*likes?', text, flags=re.IGNORECASE)
print("All explicit matches:", all_explicit)

def parse_engagement_number(text):
    text = text.upper().strip()
    match = re.match(r'^([\d\.]+)([KMB]?)', text)
    if not match: return 0
    num_str, suffix = match.groups()
    try:
        val = float(num_str)
        if suffix == 'K': return int(val * 1000)
        elif suffix == 'M': return int(val * 1000000)
        elif suffix == 'B': return int(val * 1000000000)
        else: return int(val)
    except: return 0

likes = 0
if all_explicit:
    parsed_expl = [parse_engagement_number(x) for x in all_explicit]
    likes = max(parsed_expl)
    
print("Max Likes Parsed:", likes)
