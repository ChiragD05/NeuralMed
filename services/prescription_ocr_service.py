import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

NOISE_KEYWORDS = [
    "clinic", "timing", "timings", "residence", "consultation", "mobile",
    "diagnosis", "complaints", "next visit", "prescription is valid",
    "not for medico-legal", "consultant", "physician", "doctor", "address",
    "powered by", "identity is not verified", "mr.", "mrs.", "male", "female",
    "date", "phone", "mobile no", "m.b.b.s", "mbbs", "md", "signature"
]

ROW_RE = re.compile(r"^\s*(\d{1,2})\s*[\)\.\-]?\s*(.+)$")
DOSAGE_RE = re.compile(r"\b(?:[01]\s*[-–—]\s*){2}[01]\b")
DURATION_RE = re.compile(r"\b\d+\s?(day|days|week|weeks)\b", re.I)
PHONE_RE = re.compile(r"\b\d{7,}\b")
DATE_RE = re.compile(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b", re.I)

def _find_table_bounds(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DATAFRAME,
        config="--oem 3 --psm 6"
    )

    data = data.dropna(subset=["text"])
    data["text_clean"] = data["text"].astype(str).str.strip()

    header_rows = data[data["text_clean"].str.contains("Medicine", case=False, na=False)]
    footer_rows = data[
        data["text_clean"].str.contains("Next Visit", case=False, na=False)
        | data["text_clean"].str.contains("Prescription is valid", case=False, na=False)
    ]

    if not header_rows.empty:
        y1 = int(header_rows["top"].min()) - 20
    else:
        y1 = int(h * 0.38)

    if not footer_rows.empty:
        y2 = int(footer_rows["top"].min()) - 15
    else:
        y2 = int(h * 0.78)

    x1 = int(w * 0.06)
    x2 = int(w * 0.96)

    y1 = max(0, y1)
    y2 = min(h, y2)

    if y2 <= y1:
        y1 = int(h * 0.38)
        y2 = int(h * 0.78)

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
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
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
    medicines = []

    if not text.strip():
        return medicines

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        lower = line.lower()

        # skip obvious noise
        if any(k in lower for k in NOISE_KEYWORDS):
            continue
        if PHONE_RE.search(line) or DATE_RE.search(line):
            continue

        # only numbered medicine rows
        m = ROW_RE.match(line)
        if not m:
            continue

        content = m.group(2).strip()

        # skip non-medicine table/header lines
        if content.lower().startswith("composition"):
            continue
        if len(content) < 3:
            continue

        has_dosage = bool(DOSAGE_RE.search(content))
        has_duration = bool(DURATION_RE.search(content))
        has_timing = any(
            word in lower
            for word in [
                "after food", "before food", "daily",
                "morning", "night", "twice daily", "thrice daily"
            ]
        )

        # keep only rows that look like prescription entries
        if not (has_dosage or has_timing or has_duration):
            continue

        dosage_match = DOSAGE_RE.search(content)
        dosage = dosage_match.group(0).replace("–", "-").replace("—", "-") if dosage_match else ""

        frequency = ""
        for word in ["after food", "before food", "daily", "morning", "night", "twice daily", "thrice daily"]:
            if word in lower:
                frequency = word
                break

        duration_match = DURATION_RE.search(content)
        duration = duration_match.group(0).lower() if duration_match else ""

        medicine_name = re.split(
            r"\b(?:[01]\s*[-–—]\s*){2}[01]\b|after food|before food|daily|morning|night|twice daily|thrice daily",
            content,
            flags=re.IGNORECASE
        )[0].strip(" -,:;.")

        # final safety cleanup
        medicine_name = re.sub(r"\s{2,}", " ", medicine_name)
        medicine_name = re.sub(r"\s+[cC]\s*[-–—]+\s*[-–—]+\s*\d*$", "", medicine_name).strip()

        if not medicine_name:
            continue

        medicines.append({
            "medicine_name": medicine_name,
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "raw_line": line,
        })

    return medicines