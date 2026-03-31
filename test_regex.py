import re

def parse_engagement_number(text):
    text = text.upper().strip()
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

tests = ["2.9k", "82", "yourfIguy", "f1", "1.5M", "likes", "2026"]
for t in tests:
    print(f"'{t}' -> {parse_engagement_number(t)}")
