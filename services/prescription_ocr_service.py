import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

HEADER_WORD = "Medicine"
FOOTER_WORDS = ["Next", "Visit"]

NOISE_KEYWORDS = [
    "clinic", "timing", "timings", "residence", "consultation", "mobile",
    "diagnosis", "complaints", "next visit", "prescription is valid",
    "not for medico-legal", "consultant", "physician", "doctor", "address",
    "powered by", "identity is not verified"
]

def _find_table_bounds(image):
    """
    Tries to find the medicine table area using OCR word positions.
    Returns (x1, y1, x2, y2).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slightly sharpen text for better OCR detection
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DATAFRAME,
        config="--oem 3 --psm 6"
    )
    data = data.dropna(subset=["text"])
    data["text_clean"] = data["text"].astype(str).str.strip()

    # Find the table header
    header_rows = data[data["text_clean"].str.contains(HEADER_WORD, case=False, na=False)]
    if not header_rows.empty:
        header_y = int(header_rows["top"].min())
    else:
        header_y = int(h * 0.42)

    # Find the footer area
    next_rows = data[
        data["text_clean"].str.contains("Next", case=False, na=False)
        | data["text_clean"].str.contains("Visit", case=False, na=False)
    ]
    if not next_rows.empty:
        footer_y = int(next_rows["top"].min())
    else:
        footer_y = int(h * 0.78)

    # Crop width around the medicine table
    x1 = int(w * 0.08)
    x2 = int(w * 0.94)

    # Slightly expand the vertical crop
    y1 = max(0, header_y - 25)
    y2 = min(h, footer_y - 10)

    # Fallback if bounds look wrong
    if y2 <= y1:
        y1 = int(h * 0.40)
        y2 = int(h * 0.78)
        x1 = int(w * 0.06)
        x2 = int(w * 0.96)

    return x1, y1, x2, y2

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return None

    x1, y1, x2, y2 = _find_table_bounds(image)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        crop = image

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale for better OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Noise reduction
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Improve contrast / binarize
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return gray

def extract_text_from_image(uploaded_file):
    image = preprocess_image(uploaded_file)
    if image is None:
        return "", []

    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image, config="--oem 3 --psm 6")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return text.strip(), lines

def parse_medicines(text: str):
    """
    Keep only numbered medicine rows like:
    1) CEFYCLAV 1-0-1 After Food - Daily - 2 Days
    2) ZERODOL P 1-0-1 After Food - Daily - 2 Days
    """
    medicines = []

    if not text.strip():
        return medicines

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    numbered_row = re.compile(r"^\s*(\d+)\s*[\)\.\-]?\s*(.+)$")

    for line in lines:
        lower = line.lower()

        # Skip obvious non-medicine lines
        if any(k in lower for k in NOISE_KEYWORDS):
            continue

        m = numbered_row.match(line)
        if not m:
            continue

        content = m.group(2).strip()

        # Skip composition lines
        if content.lower().startswith("composition"):
            continue

        # Extract dosage pattern like 1-0-1 / 0-0-1 / 1-1-1
        dosage_match = re.search(r"\b\d[-–—]\d[-–—]\d\b", content)
        dosage = dosage_match.group(0).replace("–", "-").replace("—", "-") if dosage_match else ""

        # Frequency / timing
        frequency = ""
        for word in ["after food", "before food", "daily", "morning", "night", "twice daily", "thrice daily"]:
            if word in lower:
                frequency = word
                break

        # Duration
        duration_match = re.search(r"\b\d+\s?(day|days|week|weeks)\b", lower)
        duration = duration_match.group(0) if duration_match else ""

        # Medicine name = text before dosage/timing words
        medicine_name = re.split(
            r"\b\d[-–—]\d[-–—]\d\b|after food|before food|daily|morning|night|twice daily|thrice daily",
            content,
            flags=re.IGNORECASE
        )[0].strip(" -,:;.")

        # Reject junk names
        if not medicine_name or len(medicine_name) < 2:
            continue

        medicines.append(
            {
                "medicine_name": medicine_name,
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration,
                "raw_line": line,
            }
        )

    return medicines