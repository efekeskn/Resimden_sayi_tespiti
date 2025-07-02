import cv2
import numpy as np
import easyocr
import re
from PIL import Image, ImageEnhance, ImageFilter
import warnings

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

reader = easyocr.Reader(['en'], gpu=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(2.0)

    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.1)

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]
    if width < 500 or height < 500:
        scale_factor = max(500/width, 500/height, 2.5)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 80, 80)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    denoised = cv2.medianBlur(cleaned, 3)
    return denoised

def detect_number_easyocr(image_path):
    try:
        preprocessing_methods = [
            preprocess_image,
            preprocess_for_dark_background,
            preprocess_minimal
        ]

        all_results = []

        for preprocess_func in preprocessing_methods:
            try:
                processed_img = preprocess_func(image_path)
                results = reader.readtext(processed_img)

                for (bbox, text, confidence) in results:
                    number = re.sub(r'[^0-9]', '', text)
                    if number and 3 <= len(number) <= 5 and confidence > 0.1:
                        all_results.append((number, confidence, text, preprocess_func.__name__))

                inverted = cv2.bitwise_not(processed_img)
                inv_results = reader.readtext(inverted)
                for (bbox, text, confidence) in inv_results:
                    number = re.sub(r'[^0-9]', '', text)
                    if number and 3 <= len(number) <= 5 and confidence > 0.1:
                        all_results.append((number, confidence, text, f"{preprocess_func.__name__}_inverted"))
            except:
                continue

        try:
            original_results = reader.readtext(image_path)
            for (bbox, text, confidence) in original_results:
                number = re.sub(r'[^0-9]', '', text)
                if number and 3 <= len(number) <= 5 and confidence > 0.1:
                    all_results.append((number, confidence, text, "original"))
        except:
            pass

        if not all_results:
            return "Unclear"

        all_results.sort(key=lambda x: x[1], reverse=True)

        four_digit_results = [r for r in all_results if len(r[0]) == 4 and r[1] > 0.2]
        if four_digit_results:
            return four_digit_results[0][0]

        three_digit_results = [r for r in all_results if len(r[0]) == 3 and r[1] > 0.25]
        if three_digit_results:
            return three_digit_results[0][0]

        for number, confidence, original_text, method in all_results:
            if confidence > 0.15:
                if len(number) == 5 and number.startswith('16'):
                    corrected = '4' + number[2:]
                    if len(corrected) == 4:
                        return corrected

                if 3 <= len(number) <= 5:
                    return number

        return "Unclear"
    except Exception as e:
        print(f"Error with EasyOCR: {e}")
        return "Unclear"

def preprocess_for_dark_background(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.8)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.5)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    if width < 400:
        scale = 400 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

def preprocess_minimal(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    height, width = img.shape[:2]
    if width < 300:
        scale = 300 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def detect_number_tesseract_fallback(image_path):
    try:
        import pytesseract
        processed_img = preprocess_image(image_path)

        configs = [
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
        ]

        for config in configs:
            text = pytesseract.image_to_string(processed_img, config=config)
            number = re.sub(r'[^0-9]', '', text.strip())
            if number and len(number) >= 3:
                return number

        return "Unclear"
    except:
        return "Unclear"

def main():
    image_paths = [
        
        "resim2.png",
        "resim3.png",
        "resim4.png",
        "resim5.png",
        "resim7.png",
    ]

    print("\n Toplam Görsel Sayısı:", len(image_paths))
    print("=" * 50)

    success_count = 0

    for idx, path in enumerate(image_paths, 1):
        print(f"\n [{idx}] Görsel: {path}")
        print("-" * 50)

        detected_number = detect_number_easyocr(path)

        if detected_number == "Unclear":
            print(" EasyOCR yetersiz kaldı, Tesseract deneniyor...")
            detected_number = detect_number_tesseract_fallback(path)

        if detected_number == "Unclear":
            print(" Tespit Edilen Sayı: Belirlenemedi")
        else:
            print(f" Tespit Edilen Sayı: {detected_number}")
            success_count += 1

        print("=" * 50)

    print(f"\n Başarıyla Tespit Edilen Sayı: {success_count}/{len(image_paths)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detected_number = detect_number_easyocr(image_path)
        if detected_number == "Unclear":
            detected_number = detect_number_tesseract_fallback(image_path)
        print(detected_number)
    else:
        main()
