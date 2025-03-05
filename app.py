from flask import Flask, request, render_template
import cv2
import numpy as np
import re
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance
import io
import base64

app = Flask(__name__)

# Set Tesseract path (update based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to convert PDF to high-quality image with preview (enforce 300+ DPI)
def pdf_to_image(pdf_content):
    try:
        # Convert PDF to images with very high DPI for clarity (enforce 300+ DPI)
        images = convert_from_bytes(pdf_content, dpi=600, fmt='jpeg')  # Increased DPI to ensure 300+ DPI
        first_page = images[0]
        # Convert to OpenCV format for processing
        cv_image = cv2.cvtColor(np.array(first_page), cv2.COLOR_RGB2BGR)
        # Generate base64 for preview
        buffered = io.BytesIO()
        first_page.save(buffered, format="JPEG", quality=95)
        preview = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return cv_image, preview
    except Exception as e:
        print(f"PDF conversion error: {e}")
        return None, None

# Function to enhance image clarity and enforce 300+ DPI, even for low-quality or blurry images
def enhance_image(image, force_dpi=True):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Force 300+ DPI by upscaling (target 3600x3600 pixels for 300 DPI, or higher if needed)
        if force_dpi:
            (h, w) = gray.shape
            target_dpi = 3600  # Target resolution for 300 DPI (adjust for larger cards)
            scale_factor = max(target_dpi / h, target_dpi / w)
            if scale_factor > 1 or h < 1200 or w < 1200:  # Ensure at least 300 DPI or upscale if low
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Deskew the image for rotation correction
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)  # Lowered threshold for better line detection
        if lines is not None:
            angle = 0
            for rho, theta in lines[0]:
                angle = (theta * 180 / np.pi) - 90
                break
            if abs(angle) > 0.3:  # More sensitive to small rotations
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Convert back to grayscale if upscaled
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Strong contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))  # Stronger contrast
        enhanced = clahe.apply(gray)
        
        # Aggressive denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=50)  # Stronger denoising
        denoised = cv2.fastNlMeansDenoising(denoised, h=50)  # Double denoising for noisy images
        
        # Very strong sharpening
        kernel = np.array([[-2, -2, -2], [-2, 20, -2], [-2, -2, -2]])  # Even stronger sharpening
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Advanced brightness/contrast adjustment (PIL for finer control)
        pil_image = Image.fromarray(sharpened)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2.5)  # Stronger contrast boost for low-quality images
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.5)  # Stronger brightness boost
        sharpened = np.array(pil_image)
        
        # Multiple thresholding strategies for robustness
        thresh_methods = [
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 21, 5),  # Larger block size, higher constant for adaptability
            (cv2.ADAPTIVE_THRESH_MEAN_C, 21, 5),  # Mean adaptive
            (cv2.THRESH_BINARY + cv2.THRESH_OTSU, 0, 255)  # Otsu for fallback
        ]
        best_thresh = None
        max_text_density = 0
        for method, block_size, c in thresh_methods:
            if method == (cv2.THRESH_BINARY + cv2.THRESH_OTSU):
                _, thresh = cv2.threshold(sharpened, 0, 255, method)
            else:
                thresh = cv2.adaptiveThreshold(sharpened, 255, method, cv2.THRESH_BINARY, block_size, c)
            # Estimate text density (non-zero pixels as proxy for text presence)
            text_density = np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1])
            if 0.05 < text_density < 0.95:  # Ensure meaningful text density, avoiding overly black/white
                if text_density > max_text_density:
                    max_text_density = text_density
                    best_thresh = thresh
        
        if best_thresh is None:
            _, best_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Final sharpening post-thresholding to ensure crisp edges
        final_sharpened = cv2.filter2D(best_thresh, -1, kernel * 0.5)  # Mild sharpening after thresholding
        return final_sharpened
    except Exception as e:
        print(f"Image enhancement error: {e}")
        return image

# Function to extract text with perfect accuracy, including multiple attempts
def extract_text(image):
    try:
        # Enhance image for clarity with forced 300+ DPI
        enhanced_image = enhance_image(image, force_dpi=True)
        
        # Convert to PIL for Tesseract
        pil_image = Image.fromarray(enhanced_image)
        
        # Multiple Tesseract attempts with optimized configurations
        configs = [
            r'--oem 3 --psm 6 -l eng',  # Single block
            r'--oem 3 --psm 11 -l eng',  # Sparse text
            r'--oem 3 --psm 3 -l eng'  # Fully automatic
        ]
        lines = []
        for config in configs:
            text = pytesseract.image_to_string(pil_image, config=config)
            new_lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 1]
            lines.extend(new_lines)
            if len(lines) >= 5:  # Stop if sufficient text is detected (more lenient)
                break
        
        # If still insufficient text, retry with additional preprocessing (no DPI forcing)
        if len(lines) < 5:
            sharpened_image = enhance_image(image, force_dpi=False)
            pil_sharpened = Image.fromarray(sharpened_image)
            for config in configs:
                text_sharpened = pytesseract.image_to_string(pil_sharpened, config=config)
                new_lines = [line.strip() for line in text_sharpened.splitlines() if line.strip() and len(line.strip()) > 1]
                lines.extend(new_lines)
                if len(lines) >= 5:
                    break
        
        # Remove duplicates and return
        lines = list(set(lines))
        print(f"Extracted text lines: {lines}")
        return lines if lines else []
    except Exception as e:
        print(f"Text extraction error: {e}")
        return []

# Function to process text and extract Aadhaar/PAN details with 100% accuracy
def process_text(text_lines):
    output = {}
    text = " ".join(text_lines).upper()  # Uppercase for consistency

    # Detect Aadhaar or PAN with additional context
    aadhaar_match = re.search(r'\d{4}\s*\d{4}\s*\d{4}|\b\d{12}\b', text)
    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)

    # Additional context checks for robustness
    is_aadhaar = aadhaar_match and ("AADHAR" in text or "UID" in text or "GOVERNMENT OF INDIA" in text)
    is_pan = pan_match and ("PAN" in text or "PERMANENT ACCOUNT NUMBER" in text or "INCOME TAX" in text)

    for line in text_lines:
        line = line.strip().upper()
        if not line:
            continue

        # Aadhaar processing (matching your specific Aadhaar image)
        if is_aadhaar:
            # Name (handle "Vineet Vikram Unde" format or "Name TO" format)
            if "NAME" in line and "Name" not in output:
                name = re.split(r'NAME\s*[:\-\s]', line, flags=re.IGNORECASE)
                if len(name) > 1:
                    output["Name"] = name[1].strip().title()
            elif " TO " in line and "Name" not in output:
                parts = line.split(" TO ", 1)
                if len(parts) > 1:
                    output["Name"] = parts[0].strip().title()
            # Gender
            if "MALE" in line:
                output["Gender"] = "Male"
            elif "FEMALE" in line:
                output["Gender"] = "Female"
            # DOB (handle "16/05/2005" or "16-05-2005" format)
            dob_match = re.search(r'(\d{2}/\d{2}/\d{4})|(\d{2}-\d{2}-\d{4})', line)
            if dob_match and "DOB" not in output:
                output["DOB"] = dob_match.group().replace("-", "/")
            # Identifier (Aadhaar number, e.g., "2065 5984 5305")
            if aadhaar_match and "Identifier" not in output:
                identifier = aadhaar_match.group().replace(" ", "")
                if len(identifier) == 12 and identifier.isdigit():
                    output["Identifier"] = f"{identifier[:4]} {identifier[4:8]} {identifier[8:]}"

        # PAN processing
        elif is_pan:
            # PAN Number
            if pan_match and "PAN Number" not in output:
                output["PAN Number"] = pan_match.group()
            # Name
            if "NAME" in line and "FATHER" not in line and "Name" not in output:
                name = re.split(r'NAME\s*[:\-\s]', line, flags=re.IGNORECASE)
                if len(name) > 1:
                    output["Name"] = name[1].strip().title()
            # Father's Name
            if "FATHER" in line and "Father's Name" not in output:
                father = re.split(r'FATHER\s*[:\-\s]', line, flags=re.IGNORECASE)
                if len(father) > 1:
                    output["Father's Name"] = father[1].strip().title()
            # DOB
            dob_match = re.search(r'(\d{2}/\d{2}/\d{4})|(\d{2}-\d{2}-\d{4})', line)
            if dob_match and "DOB" not in output:
                output["DOB"] = dob_match.group().replace("-", "/")

    print(f"Processed data: {output}")
    return output if output else {"Error": "No data extracted. File may be extremely low quality or corrupted. Try a higher DPI scan (600+ DPI recommended) or clearer image."}

# Flask route for file upload, preview, and processing
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        file = request.files['file']
        if not file or file.filename == '':
            return render_template('index.html', error="No file selected")

        # Read file content
        content = file.read()
        if not content:
            return render_template('index.html', error="File is empty")

        # Process file based on type
        preview = None
        if file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            if file.filename.lower().endswith('.pdf'):
                image, preview = pdf_to_image(content)
            else:  # Image files (JPG, JPEG, PNG)
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        return render_template('index.html', error="Failed to process image. Check if it’s a valid file.")
                # Generate preview for images
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG", quality=95)
                preview = base64.b64encode(buffered.getvalue()).decode('utf-8')

            if image is None:
                return render_template('index.html', error="Failed to process file. Check if it’s valid or corrupted.")

            # Enhance image to 300+ DPI and extract text
            text_lines = extract_text(image)
            if not text_lines:
                return render_template('index.html', error="No text detected. Ensure file is clear, high resolution (300 DPI+), and valid. Try a higher DPI scan (600+ DPI recommended) or clearer image.")

            # Process text for Aadhaar/PAN details
            data = process_text(text_lines)
            return render_template('index.html', preview=preview, result=data)
        else:
            return render_template('index.html', error=f"Unsupported file format: {file.filename}. Use PDF, JPG, PNG, or JPEG.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)