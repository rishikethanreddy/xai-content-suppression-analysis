import requests
import json

try:
    with open("test_img.png", "rb") as f:
        image_bytes = f.read()

    files = {"file": ("upload.png", image_bytes, "image/png")}
    print("Testing local API extraction directly...")
    
    resp = requests.post("http://localhost:8000/predict-post-image", files=files)
    
    print("Status Code:", resp.status_code)
    try:
        data = resp.json()
        print("API Response extracted_data likes:", data.get('extracted_data', {}).get('likes'))
        print("API Response extracted_data hashtags:", data.get('extracted_data', {}).get('hashtags'))
    except:
        print("Raw text:", resp.text)
except Exception as e:
    print("Test failed:", str(e))
