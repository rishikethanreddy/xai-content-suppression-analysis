from utils.ocr_extractor import extract_features_from_image
import traceback

try:
    with open("test_img.png", "rb") as f:
        image_bytes = f.read()

    print("Running real code...")
    res = extract_features_from_image(image_bytes)
    print("FINALLY Extracted dict:", res)
except Exception as e:
    traceback.print_exc()
