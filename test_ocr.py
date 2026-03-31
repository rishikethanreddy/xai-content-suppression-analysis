import cv2
import numpy as np

# Create an image with text
img = np.zeros((200, 800, 3), dtype=np.uint8)
img.fill(255) # White background

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "2.9K  82  yourf1guy", (10, 40), font, 1, (0, 0, 0), 2)
cv2.putText(img, "Comment your predictions and follow for more 2026 updates!", (10, 90), font, 0.7, (0, 0, 0), 2)
cv2.putText(img, "#f1 #formula1 #lewishamilton #maxverstappen", (10, 140), font, 1, (0, 0, 0), 2)

cv2.imwrite('test_img.png', img)

from utils.ocr_extractor import extract_features_from_image
with open('test_img.png', 'rb') as f:
    res = extract_features_from_image(f.read())
    
print("EXTRACTED RESULTS:")
print("Text:", res['text'])
print("Hashtags:", res['hashtags'])
print("Likes:", res['likes'])
print("Raw:", res['raw_ocr'])
