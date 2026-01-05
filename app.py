
import tempfile
import traceback
import cv2
import numpy as np
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import streamlit as st
import pdfplumber
import json
from openai import OpenAI
import os
from docx import Document
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract



# adjust path if different on your server
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- PACKING CODE DETECTION (literal regex search) ----------------
import re

PACKING_CODES = [
"1A1","1A2","1B1","1B2","1D","1G","1H1","1H2","1N1","1N2",
"3A1","3A2","3B1","3B2","3H1","3H2",
"4A","4B","4C1","4C2","4D","4F","4G","4H1","4H2",
"5H1","5H2","5H3","5H4","5L1","5L2","5L3","5M1","5M2",
"6HA1","6HG1","6PG1"
]

PACKING_REGEX = re.compile(r"\b(" + "|".join(re.escape(x) for x in PACKING_CODES) + r")\b", flags=re.IGNORECASE)

def find_packing_codes(text):
    """
    Returns a unique list of packing codes found in the given text (uppercased).
    """
    if not text:
        return []
    matches = PACKING_REGEX.findall(text)
    # unique while preserving order
    seen = set()
    unique = []
    for m in matches:
        up = m.upper()
        if up not in seen:
            seen.add(up)
            unique.append(up)
    return unique


# ---------------- Load SOP DOCX ----------------
def load_sop_text():
    sop_path = "/home/ubuntu/jupyter_workspace/test/DGD MSDS/data/DGD AND MSDS CHECKING SOP.docx"
    doc = Document(sop_path)
    return "\n".join([p.text for p in doc.paragraphs])

sop_text = load_sop_text()

def extract_pdf_text_and_image_info(uploaded_file, dpi=300):
    """
    Extracts:
    - text: pdfplumber-extracted text ONLY
    - ocr_pages: OCR text per page
    - images_info: deterministic image/text-based findings (signature label + presence)
    """

    images_info = []
    ocr_pages = []
    text = ""

    try:
        # ------------------------------------------------------------
        # Write PDF to temp file
        # ------------------------------------------------------------
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            uploaded_file.seek(0)
            tmp.write(uploaded_file.read())
            tmp.flush()

        # ------------------------------------------------------------
        # TEXT EXTRACTION (PDFPLUMBER) — DO NOT TOUCH
        # ------------------------------------------------------------
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    ptext = page.extract_text() or ""
                    if ptext.strip():
                        text += ptext + "\n"
        except Exception:
            pass  # fallback to OCR only if needed

        # ------------------------------------------------------------
        # OCR + IMAGE PROCESSING (PAGE-WISE)
        # ------------------------------------------------------------
        pil_pages = convert_from_path(tmp_path, dpi=dpi)

        for idx, pil in enumerate(pil_pages, start=1):
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            # -------- OCR PREPROCESSING --------
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if max(h, w) < 1500:
                scale = 2.0
                gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

            gray = cv2.GaussianBlur(gray, (3,3), 0)
            th = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 11
            )

            ocr_text = pytesseract.image_to_string(th, config="--oem 3 --psm 6")
            ocr_pages.append(ocr_text)


            # --------------------------------------------------------
            # 🔴 REQUIRED: define signature bbox FIRST
            # --------------------------------------------------------
            sig_bbox_pdf = find_signature_region_from_pdf(tmp_path, page_no=idx-1)

            # --------------------------------------------------------
            # SIGNATURE CHECK — STRICT, NO FALSE POSITIVES
            # --------------------------------------------------------

            label_present = False
            signature_text_present = False
            signature_ink_present = False
            sig_crop = None

            # 1️⃣ Try to find signature label via PDF text
            sig_bbox_pdf = find_signature_region_from_pdf(tmp_path, page_no=idx-1)

            if sig_bbox_pdf:
                label_present = True
                pixel_box = pdf_bbox_to_image_pixels(sig_bbox_pdf, pil)

                if pixel_box:
                    x0, y0, x1, y1 = pixel_box

                    # 🔥 EXPAND SEARCH AREA AROUND LABEL (IMPORTANT)
                    pad_x = int((x1 - x0) * 0.4)
                    pad_y = int((y1 - y0) * 1.2)

                    x0 = max(0, x0 - pad_x)
                    x1 = min(img.shape[1], x1 + pad_x)
                    y0 = max(0, y0)
                    y1 = min(img.shape[0], y1 + pad_y)

                    sig_crop = img[y0:y1, x0:x1]

            # 2️⃣ Fallback ONLY if label NOT found → bottom narrow strip
            if sig_crop is None:
                h_img, w_img = img.shape[:2]

                # ⚠️ VERY NARROW FOOTER ZONE (not whole bottom)
                y0 = int(h_img * 0.83)
                y1 = int(h_img * 0.93)
                x0 = int(w_img * 0.15)
                x1 = int(w_img * 0.85)

                sig_crop = img[y0:y1, x0:x1]
                label_present = False  # fallback ≠ label

                # 🔍 DEBUG — save signature crop image
                cv2.imwrite(f"/tmp/sig_debug_page_{idx}.png", sig_crop)


            # 3️⃣ OCR TEXT CHECK (WEAK SIGNAL ONLY)
            gray_crop = cv2.cvtColor(sig_crop, cv2.COLOR_BGR2GRAY)
            ocr_crop_text = pytesseract.image_to_string(
                gray_crop,
                config="--oem 3 --psm 6"
            )

            cleaned = re.sub(r"[^\w\s]", " ", ocr_crop_text).lower()
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            cleaned = re.sub(
                r"\b(signature|signat|declarant|déclarant|driver|conducteur)\b",
                "",
                cleaned
            ).strip()

            if len(cleaned) >= 6:
                signature_text_present = True

            # 4️⃣ INK CHECK — STRICT CONFIDENCE
            ink_flag, ink_conf = detect_signature_image(sig_crop)

            if ink_flag and ink_conf >= 0.65:
                signature_ink_present = True

            # 5️⃣ FINAL DECISION (NO GUESSING)
            signature_present = (
                (label_present and signature_ink_present) or
                (not label_present and signature_ink_present and ink_conf >= 0.75)
            )

            images_info.append({
                "page": idx,
                "label_present": label_present,
                "signature_present": signature_present,
                "signature_text_present": signature_text_present,
                "signature_ink_present": signature_ink_present,
                "red_dots_count": int(detect_red_dots(img))
            })





    except Exception:
        st.error("Error during PDF extraction. Check logs.")
        st.text(traceback.format_exc())

    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return {
        "text": text,
        "ocr_pages": ocr_pages,
        "images_info": images_info
    }


def detect_signature_image(cv2_img, debug=False):
    """
    Stricter signature detector (replace your old function with this).
    - Designed to reduce false positives from tables/logos/text.
    - Works best on a cropped region around the 'Signature' label.
    Returns (flag_bool, confidence).
    """
    img = cv2_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equalize to improve contrast
    gray = cv2.equalizeHist(gray)

    # adaptive threshold: strokes -> white on black
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 9)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    dil = cv2.dilate(closed, kernel, iterations=1)

    # detect fine strokes too
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, kernel, iterations=1)

    mask = cv2.bitwise_or(dil, edges)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0.0

    h, w = mask.shape[:2]
    page_area = max(1, w * h)

    stroke_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8:   # ignore tiny noise
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        bbox_area = max(1, ww * hh)
        fill_ratio = area / float(bbox_area)
        aspect = ww / float(hh + 1)

        # filter out large filled boxes (tables/logos) and keep smaller, thin strokes
        if area < (page_area * 0.0015) and fill_ratio < 0.65 and hh < (h * 0.25):
            stroke_contours.append((area, x, y, ww, hh, fill_ratio, aspect))

    n_stroke = len(stroke_contours)
    stroke_area = sum(s[0] for s in stroke_contours)
    stroke_density = stroke_area / float(page_area)

    small_height_count = sum(1 for s in stroke_contours if s[4] < 60)
    avg_fill = (sum(s[5] for s in stroke_contours) / n_stroke) if n_stroke else 1.0

    # Conservative confidence mapping
    conf = min(1.0, stroke_density * 200 + (small_height_count * 0.05))

    # Decision rule: require multiple thin-stroke contours and not-too-high fill_ratio,
    # or very high confidence.
    # stricter: require either a strong multi-stroke signature OR high confidence
    flag = (n_stroke >= 5 and small_height_count >= 4 and avg_fill < 0.55 and conf > 0.18) or (conf > 0.45)


    if debug:
        overlay = img.copy()
        cv2.drawContours(overlay, contours, -1, (0,255,0), 1)
        debug_p = "/tmp/sig_debug.png"
        cv2.imwrite(debug_p, overlay)
        print("Wrote debug overlay to", debug_p)

    return bool(flag), float(conf)



def find_signature_region_from_pdf(tmp_path, page_no=0):
    """
    Look only for exact SOP labels and return a NORMALIZED bbox:
    (x0_norm, y0_norm, x1_norm, y1_norm, label_text)
    where coords are in [0..1] relative to the pdf page width/height.

    Matching phrases (case-insensitive):
      - 'Signature du déclarant' / 'Signature du declarant'
      - 'Signature of declarant'
      - 'Signature pour le compte du conducteur'
      - 'Signature on behalf of the driver'
    """
    labels = [
        "signature du déclarant",
        "signature du declarant",
        "signature of declarant",
        "signature pour le compte du conducteur",
        "signature on behalf of the driver"
    ]
    try:
        with pdfplumber.open(tmp_path) as pdf:
            if page_no < 0 or page_no >= len(pdf.pages):
                return None
            page = pdf.pages[page_no]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)  # slightly tolerant
            # make a list of normalized tokens
            tokens = [( (w.get("text") or "").strip(), float(w.get("x0",0)), float(w.get("x1",0)),
                        float(w.get("top",0)), float(w.get("bottom",0)), w ) for w in words]
            lower_tokens = [t[0].lower() for t in tokens]

            # sliding window search for each multi-word label
            for kw in labels:
                kw_parts = kw.split()
                L = len(kw_parts)
                for i in range(0, len(lower_tokens)-L+1):
                    window = " ".join(lower_tokens[i:i+L])
                    if window == kw:
                        # compute bbox that spans the matched words
                        matched_words = tokens[i:i+L]
                        x0 = min(w[1] for w in matched_words) - 8.0
                        x1 = max(w[2] for w in matched_words) + 40.0   # smaller horizontal extension
                        y0 = min(w[3] for w in matched_words) - 8.0
                        y1 = max(w[4] for w in matched_words) + 40.0   # smaller vertical extension
                        # normalize
                        w_pdf = float(page.width) or 1.0
                        h_pdf = float(page.height) or 1.0
                        return (max(0.0, x0/w_pdf), max(0.0, y0/h_pdf),
                                min(1.0, x1/w_pdf), min(1.0, y1/h_pdf), kw)
            # fallback: if full text contains the label, try to find first word occurrence
            page_text = (page.extract_text() or "").lower()
            for kw in labels:
                if kw in page_text:
                    # find approximate first occurrence as before
                    for w in words:
                        if kw.split()[0] in (w.get("text") or "").lower():
                            x0 = max(0.0, float(w.get("x0",0)) - 10.0)
                            x1 = min(float(page.width), float(w.get("x1",0)) + 300.0)
                            y0 = max(0.0, float(w.get("top", w.get("y0", 0))) - 10.0)
                            y1 = min(float(page.height), float(w.get("bottom", w.get("y1", page.height))) + 120.0)
                            w_pdf = float(page.width) or 1.0
                            h_pdf = float(page.height) or 1.0
                            return (max(0.0, x0/w_pdf), max(0.0, y0/h_pdf),
                                    min(1.0, x1/w_pdf), min(1.0, y1/h_pdf), kw)
    except Exception:
        pass
    return None




def pdf_bbox_to_image_pixels(sig_bbox, pil_img):
    """
    Convert normalized PDF bbox (0..1) to image pixel coordinates.
    Handles PDF origin mismatch safely.
    """
    if not sig_bbox:
        return None

    x0n, y0n, x1n, y1n, _ = sig_bbox
    img_w, img_h = pil_img.size

    # PDF origin is bottom-left, image is top-left → invert Y
    x0 = int(x0n * img_w)
    x1 = int(x1n * img_w)
    y0 = int((1.0 - y1n) * img_h)
    y1 = int((1.0 - y0n) * img_h)

    # Clamp
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)

    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1, y1



def detect_red_dots(cv2_img):
    """
    Count small red-colored blobs (red dots).
    Works by converting to HSV and thresholding red ranges.
    Returns integer count of blobs above small area threshold.
    """
    img = cv2_img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red in HSV wraps around 0: two ranges
    lower1 = np.array([0, 120, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 120, 80])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    h, w = mask.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        # ignore large red areas (logos) — focus on dot-sized marks
        if 5 < area < 2000:
            # optional: check circularity
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter*perimeter))
            # accept approximate circular/small blobs or small elongated marks
            if circularity > 0.1 or area < 500:
                count += 1
    return count



# ---------------- JSON TEMPLATE ----------------
JSON_TEMPLATE = """
Return ONLY valid JSON in this exact structure:
{
 "presence_of_documents": { "status": "", "details": "" },
 "language_check": { "status": "", "languages_detected": "" },
 "signature_check": { "status": "", "details": "" },
 "emergency_contact_check": { "status": "", "details": "" },
 "weight_check": {
    "status": "", "gross_weight_kg": "", "net_weight_kg": "",
    "weight_unit_used": "", "details": ""
 },
 "mandatory_basic_data": {
    "un_number": "", "proper_shipping_name": "", "class": "",
    "packing_group": "", "marine_pollutant_mentioned": "",
    "details": "", "status": ""
 },
 "packing_code_check": {
    "status": "", "packing_code_found": "",
    "packing_code_value": "", "is_valid_imo_code": "",
    "details": "", "raw_text_detected": ""
 },
 "inner_packing_check": {
    "status": "", "inner_packing_required": "",
    "inner_packing_found": "", "inner_packing_material_specified": "",
    "inner_packing_material_value": "",
    "inner_packing_types_detected": [],
    "inner_packing_quantities": [],
    "inner_packing_volumes": [],
    "details": ""
 },
 "flash_point_check": {
    "status": "", "flash_point_value": "",
    "flash_point_unit": "", "flash_point_valid_format": "",
    "details": ""
 },
 "segregation_group_check": {
    "status": "", "segregation_group_present": "",
    "segregation_group_value": "", "details": ""
 },
 "msds_match_check": {
    "status": "", "un_numbers_in_dgd": [],
    "un_numbers_in_msds": [], "missing_msds_for_un": [],
    "extra_msds_un_not_in_dgd": [], "details": ""
 },
 "msds_validity": {
    "status": "", "msds_dates_found": [],
    "invalid_msds_files": [], "details": ""
 },
 "marine_pollutant_check": {
    "status": "", "marine_pollutant_detected": "",
    "details": ""
 },
 "un_restriction_check": {
    "status": "", "un_number_found": "",
    "restriction_level": "", "details": ""
 }
 "addtional_document_check":{
    "status": "", "details": ""
 }
}
"""





def sop_llm_extraction(dgd_text, msds_texts,country, loe_text, api_text, image_findings_summary=""):
    msds_blocks = ""
    for i, msds in enumerate(msds_texts, start=1):
        msds_blocks += "\n------ MSDS DOCUMENT #{} ------\n{}\n".format(i, msds)
    
    Restricted_UN=[
    "UN 0004","UN 0005","UN 0006","UN 0007","UN 0009","UN 0010","UN 0012","UN 0014","UN 0015","UN 0016","UN 0018","UN 0019","UN 0020","UN 0021",
    "UN 0027","UN 0028","UN 0029","UN 0030","UN 0033","UN 0034","UN 0035","UN 0037","UN 0038","UN 0039","UN 0042","UN 0043","UN 0044","UN 0048",
    "UN 0049","UN 0050","UN 0054","UN 0055","UN 0056","UN 0059","UN 0060","UN 0065","UN 0066","UN 0070","UN 0072","UN 0073","UN 0074","UN 0075",
    "UN 0076","UN 0077","UN 0078","UN 0079","UN 0081","UN 0082","UN 0083","UN 0084","UN 0092","UN 0093","UN 0094","UN 0099","UN 0101","UN 0102",
    "UN 0103","UN 0104","UN 0105","UN 0106","UN 0107","UN 0110","UN 0113","UN 0114","UN 0118","UN 0121","UN 0124","UN 0129","UN 0130","UN 0131",
    "UN 0132","UN 0133","UN 0135","UN 0136","UN 0137","UN 0138","UN 0143","UN 0144","UN 0146","UN 0147","UN 0150","UN 0150","UN 0151","UN 0153",
    "UN 0154","UN 0155","UN 0159","UN 0160","UN 0161","UN 0167","UN 0168","UN 0169","UN 0171","UN 0173","UN 0174","UN 0180","UN 0181","UN 0182",
    "UN 0183","UN 0186","UN 0190","UN 0191","UN 0192","UN 0193","UN 0194","UN 0195","UN 0196","UN 0197","UN 0204","UN 0207","UN 0208","UN 0209",
    "UN 0212","UN 0213","UN 0214","UN 0215","UN 0216","UN 0217","UN 0218","UN0 219","UN 0220","UN 0221","UN 0222","UN 0224","UN 0225","UN 0226",
    "UN 0234","UN 0235","UN 0236","UN 0237","UN 0238","UN 0240","UN 0241","UN 0242","UN 0243","UN 0244","UN 0245","UN 0246","UN 0247","UN 0248",
    "UN 0249","UN 0250","UN 0254","UN 0255","UN 0257","UN 0266","UN 0267","UN 0268","UN 0271","UN 0272","UN 0275","UN 0276","UN 0277","UN 0278",
    "UN 0279","UN 0280","UN 0281","UN 0282","UN 0283","UN 0284","UN 0285","UN 0286","UN 0287","UN 0288","UN 0289","UN 0290","UN 0291","UN 0292",
    "UN 0293","UN 0294","UN 0295","UN 0296","UN 0297","UN 0299","UN 0300","UN 0301","UN 0303","UN 0303","UN 0305","UN 0306","UN 0312","UN 0313",
    "UN 0314","UN 0315","UN 0316","UN 0317","UN 0318","UN 0319","UN 0320","UN 0321","UN 0322","UN 0323","UN 0324","UN 0325","UN 0326","UN 0327",
    "UN 0327","UN 0328","UN 0329","UN 0330","UN 0331","UN 0332","UN 0333","UN 0334","UN 0335","UN 0336","UN 0337","UN 0338","UN 0338","UN 0339",
    "UN 0339","UN 0340","UN 0341","UN 0342","UN 0343","UN 0344","UN 0345","UN 0346","UN 0347","UN 0348","UN 0349","UN 0350","UN 0351","UN 0352",
    "UN 0353","UN 0354","UN 0355","UN 0356","UN 0357","UN 0358","UN 0359","UN 0360","UN 0361","UN 0362","UN 0363","UN 0364","UN 0365","UN 0366",
    "UN 0367","UN 0368","UN 0369","UN 0370","UN 0371","UN 0372","UN 0373","UN 0374","UN 0375","UN 0376","UN 0377","UN 0378","UN 0379","UN 0380",
    "UN 0381","UN 0382","UN 0383","UN 0384","UN 0385","UN 0386","UN 0387","UN 0388","UN 0389","UN 0390","UN 0391","UN 0392","UN 0393","U N0394",
    "UN 0395","UN 0396","UN 0397","UN 0398","UN 0399","UN 0400","UN 0401","UN 0402","UN 0403","UN 0404","UN 0405","UN 0406","UN 0407","UN 0408",
    "UN 0409","UN 0410","UN 0411","UN 0412","UN 0413","UN 0414","UN 0415","UN 0417","UN 0418","UN 0419","UN 0420","UN 0421","UN 0424","UN 0425",
    "UN 0426","UN 0427","UN 0428","UN 0429","UN 0430","UN 0431","UN 0432","UN 0433","UN 0434","UN 0435","UN 0436","UN 0437","UN 0438","UN 0439",
    "UN 0440","UN 0441","UN 0442","UN 0443","UN 0444","UN 0445","UN 0446","UN 0447","UN 0448","UN 0449","UN 0450","UN 0451","UN 0452","UN 0453",
    "UN 0454","UN 0455","UN 0456","UN 0457","UN 0458","UN 0459","UN 0460","UN 0461","UN 0462","UN 0463","UN 0464","UN 0465","UN 0466","UN 0467",
    "UN 0468","UN 0469","UN 0470","UN 0471","UN 0472","UN 0473","UN 0474","UN 0475","UN 0476","UN 0477","UN 0478","UN 0479","UN 0480","UN 0481",
    "UN 0482","UN 0483","UN 0484","UN 0485","UN 0486","UN 0487","UN 0488","UN 0489","UN 0490","UN 0491","UN 0492","UN 0493","UN 0494","UN 0495",
    "UN 0496","UN 0497","UN 0498","UN 0499","UN 0500","UN 0501","UN 0502","UN 0503","UN 0504","UN 0505","UN 0506","UN 0507","UN 0508","UN 0509",
    "UN 1008","UN 1017","UN 1026","UN 1045","UN 1048","UN 1048","UN 1053","UN 1053","UN 1067","UN 1067","UN 1069","UN 1076","UN 1079","UN 1204",
    "UN 1310","UN 1320","UN 1321","UN 1322","UN 1336","UN 1337","UN 1344","UN 1347","UN 1348","UN 1349","UN 1354","UN 1355","UN 1356","UN 1357",
    "UN 1512","UN 1517","UN 1571","UN 1581","UN 1582","UN 1589","UN 1612","UN 1660","UN 1749","UN 1798","UN 1859","UN 1911","UN 1942","UN 1966",
    "UN 1975","UN 2059","UN 2059","UN 2067","UN 2071","UN 2186","UN 2188","UN 2189","UN 2190","UN 2191","UN 2192","UN 2194","UN 2195","UN 2196",
    "UN 2198","UN 2199","UN 2202","UN 2204","UN 2249","UN 2417","UN 2418","UN 2420","UN 2421","UN 2455","UN 2534","UN 2548","UN 2555","UN 2556",
    "UN 2557","UN 2676","UN 2814","UN 2852","UN 2900","UN 2901","UN 2907","UN 2908","UN 2909","UN 2910","UN 2911","UN 2912","UN 2913","UN 2915",
    "UN 2916","UN 2917","UN 2919","UN 2977","UN 2978","UN 3064","UN 3083","UN 3097","UN 3100","UN 3101","UN 3102","UN 3111","UN 3112","UN 3113",
    "UN 3114","UN 3115","UN 3116","UN 3117","UN 3118","UN 3119","UN 3120","UN 3121","UN 3127","UN 3133","UN 3137","UN 3255","UN 3291","UN 3317",
    "UN 3319","UN 3321","UN 3322","UN 3323","UN 3324","UN 3325","UN 3326","UN 3327","UN 3328","UN 3329","UN 3330","UN 3331","UN 3332","UN 3333",
    "UN 3343","UN 3344","UN 3357","UN 3364","UN 3365","UN 3366","UN 3367","UN 3368","UN 3369","UN 3370","UN 3373","UN 3375","UN 3376","UN 3379",
    "UN 3380","UN 3474","UN 3507","UN 3509","UN 3519","UN 3520","UN 3521","UN 3522","UN 3523","UN 3524","UN 3525","UN 3526"
    ]


    # ---------------- SAFE PROMPT (NO PYTHON CODE INSIDE THE PROMPT) ----------------
    prompt = """
You are an expert in Dangerous Goods (DGD) & MSDS validation. You have a task to extract data from the documents that will be provided, 
and provide output as specified in the rules and terms ahead.

------ SOP RULES ------
{SOP_TEXT}

------ DGD DOCUMENT ------
{DGD_TEXT}

------ MSDS DOCUMENTS ------
{MSDS_BLOCKS}

------ IMAGE FINDINGS ------
{IMAGE_FINDINGS}

------ DESTINATION COUNTRY ------
{DESTINATION_COUNTRY}

------ LOE DOCUMENT ------
{LOE_TEXT}

------ API DOCUMENT ------
{API_TEXT}

------ RESTRICTED UN NUMBERS ------
{RESTICTED_UN}


------ GLOBAL EXTRACTION SAFETY RULES ------
These rules override all other logic. Do NOT hallucinate, infer, assume, or guess ANY data not explicitly written in the DGD/MSDS text.

IMPORTANT — ZERO ASSUMPTION RULE
- If a field does NOT exist in the actual PDF text, return it as empty.
- DO NOT fabricate values based on logic, expectations, IMDG rules, or typical wording.
- DO NOT correct user data.
- DO NOT infer packing codes, signatures, flash points, or emergency numbers.
- Only return EXACT values visible in the raw PDF text.
- If the item is not literally present → mark status = "not found".

──────────────────────────────────────────
1) PRESENCE OF DOCUMENTS
──────────────────────────────────────────
- Count how many DGD and MSDS documents are detected in the provided input.
- Do NOT assume additional documents.
- If only 1 MSDS is present, do NOT generate separate MSDS for multiple UNs.
- Number of MSDS should be equal to the number of *UNIQUE* UN numbers present in one DGD. If all the times that the UN number is present in 
the DGD are the same number, then number of UN numbers will be 1. If more than 1 different UN numbers are present in the DGD document, then
that will be the number of UN numbers present in the DGD.
-If number of *UNIQUE* UN numbers != number of MSDS →  "MSDS not provided for all UNs".
-If number of *UNIQUE* UN numbers = number of MSDS →  "Presence of documents checked".
──────────────────────────────────────────
2) LANGUAGE DETECTION RULES
──────────────────────────────────────────
- Detect languages ONLY from real extracted text.
- If DGD contains a mixture (ex: French + English), return all detected languages.
- Do NOT hallucinate translation or assume document language.

──────────────────────────────────────────
3) SIGNATURE VALIDATION RULES
──────────────────────────────────────────
Signature should be present in the DGD.
Following are the rules to check for the signature in the DGD:

A valid signature exists ONLY IF:
- It is next to or under a signature field OR
- Sewing, stamp, or actual sign marks appear.
- Name of field is SIGNATURE DU DECLARENT OR SIGNATURE OF DECLARANT OR Any translation of this phrase
Do NOT treat company names as signatures.

If no valid sign:
- status = "Signature missing"
- details = "Declarant signature field is not signed."

If signature is an image or unreadable text:
status = "Signature present (image)"
details = "Signature appears visually but no text extracted"

──────────────────────────────────────────
4) EMERGENCY CONTACT EXTRACTION RULES
──────────────────────────────────────────
A valid emergency phone number has the following:
- MUST contain minimum 7 digits and only digits, with the exception of 
- inclusion of “+”, “-”, spaces, parentheses

INVALID phone indicators:
- “CARECHEM”, “NCEC”
- Strings like “CARECHEM 29003/NCEC”
- Internal codes, ticket numbers, hotline references

If only CARECHEM/NCEC appears → treat as CONTACT NAME only.

──────────────────────────────────────────
5) WEIGHT CHECK RULES
──────────────────────────────────────────
- Extract weight ONLY if written in the DGD.
- Units must be kg.
- Check the column/row heading before fetching value of net weight/gross weight. Remember to show net mass/net weight as that and gross 
mass/gross weight as that. Do not mix the two. Especially, if only one is present. 
- Gross weight MUST be > net weight.
If invalid → status = "Weight incorrect".
If both are missing → status = "Weight missing"
If any one is missing → status = "Weight incomplete"

──────────────────────────────────────────
6) MANDATORY BASIC DATA RULES
──────────────────────────────────────────
Required order:
UN NUMBER → Proper Shipping Name → Technical Name (if NOS or NSA or marine pollutant) → Class → PG.

Technical name rules:
- If NOS or NSA or Marine Pollutant is mentioned → technical name MUST appear in parenthesis.
- If technical name missing while NOS or NSA or marine pollutant present → RED FAIL.

──────────────────────────────────────────
7) PACKING CODE RULES *(VERY STRICT)*
──────────────────────────────────────────
Valid packing codes MUST appear EXACTLY and LITERALLY in the DGD text.
list of Existing ONU Packing code (Outturn packing) This is MANDATORY on ALL DGD 
Do NOT hallucinate a packing code if it is not literally mentioned in the DGD.
The packing code will be present in or near the line containing the UN number, or in the field that contains the line with the UN number.


These codes are defined in the IMDG Code (Chapter 6.1) and indicate: 

First digit = Type of packaging 

Letter = Material of construction 

Additional digit = Category (e.g., removable head or not) 

Shape 

1) Drums 

1A1: Steel, non-removable head 

1A2: Steel, removable head 

1B1 / 1B2: Aluminum 

1D: Plywood 

1G: Fiber 

1H1 / 1H2: Plastic 

1N1 / 1N2: Metal other than steel or aluminum 

3 )Jerricans 

3A1 / 3A2: Steel 

3B1 / 3B2: Aluminum 

3H1 / 3H2: Plastic 

4 ) Boxes 

4A: Steel 

4B: Aluminum 

4C1 / 4C2: Natural wood 

4D: Plywood 

4F: Reconstituted wood 

4G: Fiberboard 

4H1 / 4H2: Plastic 

5 )Bags 

5H1 / 5H2 / 5H3 / 5H4: Plastic (woven, film) 

5L1 / 5L2 / 5L3: Textile 

5M1 / 5M2: Paper multi-wall 

6 )Composite Packagings 

Examples:  

6HA1: Plastic receptacle in steel drum 

6HG1: Plastic receptacle in fiberboard box 

6PG1: Glass receptacle in fiberboard box 

Read carefully and do not mix it with any other type of code provided in the DGD.
Do NOT infer packing code from packaging description.
The packing code must be present within 450 characters before or after the UN Number
Return packing_code_found = "no" unless literal code appears.
Return packing_code_found = "no" if the code is found more than 450 characters before or after the UN Number.


──────────────────────────────────────────
8) INNER PACKING RULES (Appendix 6)
──────────────────────────────────────────
- REQUIRED only when OUTTURN PACKING CODE begins with “4…”
- NEVER hallucinate inner packing.
- Extract it ONLY if explicitly written.
- If you have extracted inner packing information, make sure that the specific material is also precised.
- If outer packing starts with “4…” and NONE detected → status = "Inner packing missing".
- If no material is detected → status = "Inner packing material missing".

VALID inner packing types:
- Bottles (glass/plastic/metal)
- Jars (glass/plastic)
- Cans (metal/plastic)
- Jerricans (small metal/plastic)
- Flasks
- Ampoules
- Plastic receptacles
- Metal receptacles
- Porcelain/Stoneware/Earthenware containers
- Composite inner packaging
- Bags
- Tubes

──────────────────────────────────────────
9) FLASH POINT RULES
──────────────────────────────────────────
Applies ONLY to Class 3 specifically.

VALID FORMATS OF FLASH POINT VALUE:
- A single °C value (e.g., 23 °C)
- °C value followed by °F in parenthesis (e.g., 10 °C (50 °F))
- Characters such as '+' and '-' are also permitted but they might not be present in all documents

Flash point is INVALID ONLY IF:
- °C numeric value is missing
- °C value is non-numeric
- Only Fahrenheit value is present without Celsius
- Contains “>” or “<”

──────────────────────────────────────────
10) SEGREGATION GROUP RULES
──────────────────────────────────────────
Applies only to Class 8:
- If present → "mentioned"
- If missing → "might be requested"
- If class != 8 → "No message"

──────────────────────────────────────────
11) MSDS–DGD MATCH
──────────────────────────────────────────
Strict match:
- UN Number
- Proper Shipping Name
- Class
If MSDS does not match DGD, return status → "MSDS-DGD match fail"

──────────────────────────────────────────
12) MSDS VALIDITY
──────────────────────────────────────────
- Extract actual MSDS date.
- Must be < 5 years old.
- If MSDS is > 5 years old, status→ "invalid"

──────────────────────────────────────────
13) MARINE POLLUTANT RULES
──────────────────────────────────────────
If Marine Pollutant YES:
- Technical name MUST be present.
- If missing → RED

Else → GREEN if handled correctly.
If GREEN Status→ "No Marine Pollutant found"

──────────────────────────────────────────
14) ADDITIONAL DOCUMENT RULES
──────────────────────────────────────────
These rules apply ONLY when the Destination Country is Algeria OR Ivory Coast.
For any other destination country, these rules are not applicable and return status → "No additional documents required".

If the destination country is Algeria:
- LOE(Letter of Engagement) is required as additional document.
- Do not assume presence of document. It should be explicitly uploaded and provided.
- The document should have the words "Letter of Engagement" or "Lettre d'Engagement" on the top of the page.
- The document must have 3 stamps and the bottom of the page.
- Sample LOE for Algeria is given in Appendix 1 on page 7 of the SOP rules. Refer to that sample document and check that the provided LOE is valid. Valid 
if the document uploaded is similar but the format of the document should be exactly the same as the sample provided.

If valid, return status → "LOE present and valid".
if invalid, return status → "invalid".



If the destination country is Ivory Coast:
- API(Autorisation Prealable d'Importation) and LOE(Letter of Engagement) are required as additional documents.
- Do not assume presence of documents. They should be explicitly uploaded and provided.
- Sample API for Ivory Coast is given in Appendix 2 on page 8 of the SOP rules. Refer to that document and check that the provided API is 
valid. Valid if the document uploaded is similar but the format of the document should be exactly the same as the sample provided.
- Sample LOE for Ivory Coast is given in Appendix 3(mentioned in the SOP rules as Abidjan) on page 9 of the SOP rules. Refer to that sample 
document and check that the provided LOE is valid. Valid if the document uploaded is similar but the format of the document should be exactly
the same as the sample provided.

If valid, return status → "API and LOE present and valid".
If LOE is invalid, return status → "invalid LOE".
If API is invalid, return status → "invalid API
If both are invalid, return status → "invalid API and LOE".

──────────────────────────────────────────
14) UN RESTRICTION RULES
──────────────────────────────────────────
-A list of Restricted UN Numbers has been provided. Adhere strictly to the list provided. Do no update the list, or change it in any way.
-If the any of the UN numbers found in the DGD are present in the provided list of Restricted UN Numbers, return status in the form of "UN XXXX
is restricted". Here XXXX should be replaced by the actual number of the restricted UN Number that was found in the DGD.

-If none of the UN Numbers found in the DGD are in the provided list of Restricted UN Numbers, return status → "All UN Numbers clear" and 
return details → "UN Number(s) not present in list".


------ VALIDATION FORMAT ------
{JSON_TEMPLATE}
""".format(
        SOP_TEXT=sop_text,
        DGD_TEXT=dgd_text,
        MSDS_BLOCKS=msds_blocks,
        IMAGE_FINDINGS=image_findings_summary,
        JSON_TEMPLATE=JSON_TEMPLATE,
        DESTINATION_COUNTRY=country,
        LOE_TEXT=loe_text,
        API_TEXT=api_text,
        RESTICTED_UN=Restricted_UN
    )

    # ---------------- SEND TO OPENAI ----------------
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# ------------------ STREAMLIT VALIDATION PORTAL UI ------------------
st.set_page_config(page_title="DGD & MSDS Validation Portal", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #d5e1eb, #add6f7);}
.main-title { font-size: 32px; font-weight: 700; margin-bottom: -10px; }
.subtitle { font-size: 16px; color: #444; margin-bottom: -40px}
.stFileUploader {padding: 1rem; border-radius: 10px; background-color: #8cc6f5; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 20px;}
div[data-testid="stFileUploader"] section {background-color: #f9fafb;}
.stButton>button {width: 100%; border-radius: 8px; height: 3em; background-color: #2574f5; color: white; font-weight: bold; border: none; transition: 0.3s;}
.stButton>button:hover {background-color: #3295e6; color: white; border: none;}
div[data-testid="stFileUploader"] button {color: #0e266b; background-color: #ffffff; font-weight: 600 ;}
.upload-box { text-align: center; margin-bottom: -50px; margin-top: 0px; }
.result-row { padding: 12px 0; border-bottom: 1px solid #eee; }
.label { font-weight: 600; }
div[data-testid="stAlert"] {background-color: #ffffff;border: 1px solid #e5e7eb;border-radius:12px; box-shadow:none; overflow:hidden;}
div[data-testid="stFileUploader"] button{background-color: transparent !important; box-shadow:none;}
div[data-testid="stFileUploader"] button:hover{background-color:transparent !important;}
hr {margin: 10px !important; padding: 5px !important; height: 10px;}
div[data-testid="stPills"] button[data-selected="true"] {background: #ffffff}
</style>

""", unsafe_allow_html=True)
    
st.markdown("<div class='main-title'>DGD & MSDS Validation Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your DGD & MSDS files for instant SOP validation ⚡</div>", unsafe_allow_html=True)


# ------------------ Upload Panel ------------------
left, right = st.columns([2.57, 1], gap="large")


with left:
    st.divider()
    st.markdown("<div class='upload-box'>Upload DGD & MSDS PDFs below</div>", unsafe_allow_html=True)
    st.divider()
    cola, colb= st.columns(2)
    with cola:
        uploaded_dgd = st.file_uploader("Upload DGD PDF", type=["pdf"], key="dgd")
    with colb:
        uploaded_msds = st.file_uploader("Upload MSDS PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True, key="msds")
    st.write("### 🌍 Logistics Details")
    country = st.pills("Select Destination Country",["France", "Belgium", "Netherlands", "Ivory Coast","Algeria", "Brazil", "Mexico", "Peru", 
                        "Chile"], selection_mode="single")
    if country == "Algeria":
        st.write("Letter of Engagement/ Lettre d'Engagement is required for destination: Algeria. Please upload it below")
        uploaded_loe= st.file_uploader("Upload Letter of Engagement here", type=["pdf"], key="loe")
    if country == "Ivory Coast":
        st.write("API and Letter of Engagement is required for destination: Ivory Coast. Please upload them below")
        ad_1, ad_2= st.columns(2)
        with ad_1:
            uploaded_api= st.file_uploader("Upload API here", type=["pdf"], key="api")
        with ad_2:
            uploaded_loe= st.file_uploader("Upload Letter of Engagement here", type=["pdf"], key="loe")

    run_validation = st.button("Validate Now", use_container_width=True)

with right:
    st.markdown("### Recommendations")
    if not uploaded_dgd or not uploaded_msds:
        st.info("Upload DGD and MSDS document to see AI-driven Recommendations")



# ------------------ VALIDATION PROCESS ------------------
if run_validation:

    if not uploaded_dgd:
        st.error("Please upload a DGD PDF before validating.")
        st.stop()
    if country == "Algeria":
        if not uploaded_loe:
            st.error("Please upload a LOE PDF before validating.")
            st.stop()
    if country == "Ivory Coast":
        if not uploaded_api or not uploaded_loe:
            st.error("Please upload a API and Letter of Engagement PDF before validating.")
            st.stop()



    with st.spinner("Extracting & validating..."):
        api_text=''
        loe_text=''

        # Use the enhanced extractor (image + OCR) if you added it; otherwise use existing extractor
        dgd_result = extract_pdf_text_and_image_info(uploaded_dgd, dpi=300)  # if you implemented this helper
        # ALWAYS define img_info explicitly
        img_info = dgd_result.get("images_info", []) if isinstance(dgd_result, dict) else []
        pdf_text = (dgd_result.get("text", "") or "").strip()
        ocr_text = "\n".join(dgd_result.get("ocr_pages", [])).strip()

        # -------- OCR MODE DETECTION (CORRECTED) --------
        PDF_TEXT_MIN_CHARS = 200   # threshold to consider document text-based

        is_ocr_only = (len(pdf_text) < PDF_TEXT_MIN_CHARS and len(ocr_text) > 300)



        # Use OCR only if PDF text is missing
        if pdf_text:
            combined_dgd_text = pdf_text
            text_source = "pdf"
        else:
            combined_dgd_text = ocr_text
            text_source = "ocr"

        # DEBUG (temporary)
        st.write("TEXT SOURCE:", text_source)



        if country == "Algeria":
            loe_result = extract_pdf_text_and_image_info(uploaded_loe, dpi=300)  
            loe_text = loe_result.get("text", "") if isinstance(loe_result, dict) else loe_result
        
        if country == "Ivory Coast":

            api_result = extract_pdf_text_and_image_info(uploaded_api, dpi=300)  
            api_text = api_result.get("text", "") if isinstance(api_result, dict) else api_result

            loe_result = extract_pdf_text_and_image_info(uploaded_loe, dpi=300)  
            loe_text = loe_result.get("text", "") if isinstance(loe_result, dict) else loe_result

        # ---------------- DEBUG OUTPUT ----------------
        # st.write("DEBUG images_info:", json.dumps(dgd_result.get("images_info", []), indent=2))
        # st.write("DEBUG ocr_pages (first 3):", [p[:300] for p in dgd_result.get("ocr_pages", [])[:3]])

        # detect packing codes deterministically in the DGD text
        detected_packing = find_packing_codes(combined_dgd_text)

        image_summary = "\n".join([
            f"Page {it.get('page')}: "
            f"label_present={it.get('label_present', False)}, "
            f"signature_detected={it.get('signature_detected', False)}, "
            f"red_dots={it.get('red_dots_count', 0)}"
            for it in img_info
        ])



        # append packing codes to the summary so LLM sees literal tokens
        if detected_packing:
            image_summary += "\n[AUTOMATED-DETECTED-PACKING-CODES]: " + ", ".join(detected_packing)

        # Process MSDS files similarly (text only or with image info)
        msds_texts = []
        if uploaded_msds:
            for msds in uploaded_msds:
                r = extract_pdf_text_and_image_info(msds, dpi=300)
                msds_texts.append(r.get("text","") if isinstance(r, dict) else r)
                # optionally append msds image findings into image_summary:
                if isinstance(r, dict):
                    image_summary += "\n" + "\n".join([
                        f"MSDS Page {it.get('page')}: "
                        f"red_dots={it.get('red_dots_count', 0)}, "
                        f"signature_detected={it.get('signature_detected', False)}"
                        for it in img_info
                    ])





        llm_output = sop_llm_extraction(combined_dgd_text, msds_texts, country, loe_text, api_text, image_findings_summary=image_summary)
        



        cleaned = llm_output.strip().replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
        except:
            st.error("LLM returned invalid JSON.")
            st.code(llm_output)
            st.stop()

        # ================= DETERMINISTIC OVERRIDES =================

        # ---- DGD PRESENCE ----
        if is_ocr_only:
            parsed["presence_of_documents"] = {
                "status": "Presence of documents checked",
                "details": f"DGD uploaded. Extraction mode used: {text_source.upper()}."
            }

        else:
            parsed["presence_of_documents"] = {
                "status": "Presence of documents checked",
                "details": "DGD and MSDS documents were uploaded."
            }

        # ---- MANDATORY BASIC DATA (OCR-ONLY SAFE OVERRIDE) ----
        if is_ocr_only and not parsed.get("mandatory_basic_data", {}).get("un_number"):
            parsed["mandatory_basic_data"] = {
                "status": "not found",
                "un_number": "",
                "proper_shipping_name": "",
                "class": "",
                "packing_group": "",
                "marine_pollutant_mentioned": "",
                "details": "DGD is image-based. Mandatory data could not be reliably extracted via OCR."
            }




# ----------------- Deterministic signature override (TEXT / OCR BASED) -----------------
try:
    dgd_img_info = dgd_result.get("images_info", []) if isinstance(dgd_result, dict) else []

    # Pages where signature label exists
    label_pages = [it["page"] for it in dgd_img_info if it.get("label_present")]

    # Pages where actual signature content (OCR/text) exists under label
    signed_pages = [it["page"] for it in dgd_img_info if it.get("signature_present")]

    if signed_pages:
        parsed["signature_check"] = {
            "status": "Signature present (image)",
            "details": (
                "Signature content detected near declarant signature field "
                f"on page(s): {', '.join(map(str, signed_pages))}."
            )
        }

    elif label_pages:
        parsed["signature_check"] = {
            "status": "Signature missing",
            "details": (
                "Signature field found on page(s): "
                f"{', '.join(map(str, label_pages))}, but no signature content detected."
            )
        }

    else:
        parsed["signature_check"] = {
            "status": "Signature missing",
            "details": "No declarant signature field found in the DGD."
        }

except Exception:
    pass
# ----------------- end deterministic override -----------------






# ---------------- SAFETY CHECK ----------------
if run_validation and 'parsed' not in locals():
    st.stop()

if not run_validation:
    st.stop()


with right:

    st.markdown("### ⚠️ Issues Summary")

    ERROR_WORDS = [
        "missing", "fail", "incorrect", "invalid","restricted",
        "not found", "inner packing missing", "may request validity "
    ]

    WARNING_WORDS = [
        "might", "warning", "incomplete",
        "not specified", "material not specified"
    ]

    def norm(text):
        return text.lower().replace("-", " ").replace("_", " ").strip()

    errors = []
    warnings = []

    # ----------- CLASSIFY ISSUES -----------
    for key, data in parsed.items():
        status = norm(data.get("status", ""))
        details = data.get("details", "").strip()
        key_name = key.replace("_", " ").title()

        # 🔴 ERRORS
        if any(w in status for w in ERROR_WORDS) or any(w in details.lower() for w in ERROR_WORDS):
            errors.append(f"**{key_name}** — {details}")

        # 🟡 WARNINGS
        elif any(w in status for w in WARNING_WORDS) or any(w in details.lower() for w in WARNING_WORDS):
            warnings.append(f"**{key_name}** — {details}")

    # ----------- RENDER CLEAN OUTPUT -----------

    # Errors First
    if errors or warnings:
        if errors: 
            st.markdown("#### 🔴 Errors")
            for item in errors:
                st.markdown(f"- {item}")
        if warnings: 
            st.markdown("#### 🟡 Warnings")
            for item in warnings:
                st.markdown(f"- {item}")
    else:
        st.markdown("### 🟢 *No errors detected.*")











# ------------------ RESULT PANEL ------------------

with left:
    st.markdown("### Validation Results")
    def icon(status, key=None, item=None):
        if not status:
            return "⚪"

        # Normalize EVERYTHING once
        s = status.lower().replace("-", " ").replace("_", " ").replace("  ", " ")
        d = item.get("details", "").lower().replace("-", " ").replace("_", " ").replace("  ", " ")

        # ----------------------------------------
        # SPECIAL CASE — SEGREGATION GROUP
        # ----------------------------------------
        if key == "segregation_group_check":
            if "no message" in s or "not class 8" in s:
                return "⚪"
            if "might" in s or "requested" in s:
                return "🟡"
            if "mentioned" in s or "present" in s:
                return "🟢"
            return "⚪"

        # ----------------------------------------
        # SPECIAL CASE — MSDS VALIDITY (> 5 years old)
        # ----------------------------------------
        if key == "msds_validity":
            dates = item.get("msds_dates_found", [])
            if dates:
                try:
                    # extract first date
                    from datetime import datetime
                    msds_year = int(str(dates[0])[:4])  # assume "YYYY-MM-DD" or similar
                    current_year = datetime.now().year

                    if current_year - msds_year > 5:
                        return "🔴"  # older than 5 years -> red
                except:
                    pass
            # If within 5 years and status is OK
            if "valid" in s or "ok" in s or "checked" in s:
                return "🟢"
            # If LLM still said "older than 5 years" in text
            if "older" in d or "5 years" in d:
                return "🔴"

        # ----------------------------------------
        # SPECIAL CASE — INNER PACKING
        # ----------------------------------------
        if key == "inner_packing_check":

            # Case 1: Outturn packing does NOT start with "4" → NOT REQUIRED → always GREY ⚪
            if item.get("inner_packing_required", "").lower() == "no":
                return "⚪"

            # Case 2: Required but NOT FOUND → RED
            if item.get("inner_packing_required", "").lower() == "yes" and \
               item.get("inner_packing_found", "").lower() == "no":
                return "🔴"

            # Case 3: Required AND found → GREEN
            if item.get("inner_packing_required", "").lower() == "yes" and \
               item.get("inner_packing_found", "").lower() == "yes":
                return "🟢"

            # Otherwise fallback to grey
            return "⚪"


        # ----------------------------------------
        # SPECIAL CASE — FLASH POINT (Class 3 only)
        # ----------------------------------------
        if key == "flash_point_check":

            status_fp = s  # normalized status
            valid_format = (item.get("flash_point_valid_format", "") or "").lower()

            # Case 1: Not required (class is not 3)
            if "not required" in status_fp:
                return "⚪"

            # Case 2: Required & invalid format → RED
            if "invalid" in status_fp or valid_format == "no":
                return "🔴"

            # Case 3: Required & valid format → GREEN
            if valid_format == "yes" or "valid" in status_fp:
                return "🟢"

            # Fallback
            return "⚪"

        # ----------------------------------------
        # SPECIAL CASE — MARINE POLLUTANT
        # ----------------------------------------
        if key == "marine_pollutant_check":
            mp = item.get("marine_pollutant_detected", "").lower()
            dt = d

            if "yes" in mp:
                if any(x in dt for x in ["technical name missing", "not specified", "n.o.s", "nos"]):
                    return "🔴"
                return "🟢"
            return "🟢"

        # ----------------------------------------
        # SPECIAL CASE — UNRESTRICTION
        # ----------------------------------------
        if key == "un_restriction_check":
            if "restricted" in s:
                return "🔴"
            if "clear" in s:
                return "🟢"

        # ----------------------------------------
        # RED FAIL CONDITIONS
        # ----------------------------------------
        if any(w in s for w in ["missing", "fail", "incorrect", "not found", "invalid"]):
            return "🔴"

        # Packing code missing
        if key == "packing_code_check":
            if not item.get("packing_code_value"):
                return "🔴"

        # Flash point wrong
        if key == "flash_point_check":
            if any(w in s for w in [">", "<", "incorrect", "missing"]):
                return "🔴"

        # ----------------------------------------
        # YELLOW WARNING CONDITIONS
        # ----------------------------------------
        if any(w in s for w in ["warning", "might", "incomplete"]) or \
        any(w in d for w in ["not specified", "material not specified"]):
            return "🟡"

        # ----------------------------------------
        # GREEN PASS
        # ----------------------------------------
        if any(w in s for w in ["checked", "found", "present", "valid", "match", "ok", "correct"]):
            return "🟢"

        return "⚪"



    # ---------------- DISPLAY ALL VALIDATION LINES ----------------
    for key, label in [
        ("presence_of_documents", "Presence of DGD + MSDS"),
        ("language_check", "Language Check"),
        ("signature_check", "Signature"),
        ("emergency_contact_check", "Emergency Contact"),
        ("weight_check", "Weight Check"),   
        ("mandatory_basic_data", "Mandatory Basic Data"),
        ("packing_code_check", "Packing Code"),
        ("inner_packing_check", "Inner Packing"),
        ("flash_point_check", "Flash Point"),
        ("segregation_group_check", "Segregation Group"),
        ("msds_match_check", "UN Match"),
        ("msds_validity", "MSDS Validity"),
        ("marine_pollutant_check", "Marine Pollutant"),
        ("un_restriction_check", "UN Restriction Check"),
        ("addtional_document_check", "Additional Documents") # ✅ newly added
    ]:

        item = parsed.get(key, {})
        status = item.get("status", "")
        details = item.get("details", "")

        # SPECIAL: SHOW LANGUAGES CLEARLY
        if key == "language_check":
            langs = item.get("languages_detected", "")
            details = f"Languages detected: {langs}"

        st.markdown(f"""
            <div class='result-row'>
                <span style='font-size:22px'>{icon(status, key, item)}</span>
                <span class='label'> {label} — {status}</span>
                <div style='margin-left:40px; color:#2c3e50;'>{details}</div>
            </div>
        """, unsafe_allow_html=True)


# ---------------- SHOW RAW + PRETTY JSON ----------------
st.markdown("### 📦 Extracted JSON (Raw Output)")
st.code(cleaned, language="json")

try:
    pretty_json = json.dumps(parsed, indent=4)
    st.markdown("### 📘 Extracted JSON (Formatted)")
    st.code(pretty_json, language="json")
except:
    st.warning("Could not format JSON.")




