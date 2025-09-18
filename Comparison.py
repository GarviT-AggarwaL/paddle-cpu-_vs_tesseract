import os
import time
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image

image_folder = "./new_dataset"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]


ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

start_time = time.time()
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    _ = ocr.predict(image_path)  # run OCR
paddle_time = time.time() - start_time
print(f"PaddleOCR processed {len(image_files)} images in {paddle_time:.2f} seconds")


start_time = time.time()
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    _ = pytesseract.image_to_string(Image.open(image_path))  # run OCR
tesseract_time = time.time() - start_time
print(f"Tesseract processed {len(image_files)} images in {tesseract_time:.2f} seconds")


print("\nTime Comparison:")
print(f"PaddleOCR: {paddle_time:.2f} sec")
print(f"Tesseract: {tesseract_time:.2f} sec")
