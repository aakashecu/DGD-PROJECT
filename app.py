
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

from pdf2image import convert_from_path
import pytesseract



# adjust path if different on your server
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"



# Initialize OpenAI client


import streamlit as st
from openai import OpenAI

api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit secrets")
    st.stop()

client = OpenAI(api_key=api_key)


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
    # SOP file must be inside your repo, e.g. ./data/
    sop_path = os.path.join(os.path.dirname(__file__), "data", "DGD AND MSDS CHECKING SOP.docx")

    if not os.path.exists(sop_path):
        st.error(f"SOP file not found at: {sop_path}")
        st.stop()

    doc = Document(sop_path)
    return "\n".join(p.text for p in doc.paragraphs)

sop_text = load_sop_text()


def extract_pdf_text_and_image_info(uploaded_file, dpi=300):
    """
    Safer extraction: writes to a NamedTemporaryFile, converts pages to images,
    runs OCR and image analysis. Returns dict {text, images_info, ocr_pages}.
    """
    images_info = []
    ocr_pages = []
    text = ""

    try:
        # write to a unique temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            uploaded_file.seek(0)
            tmp.write(uploaded_file.read())
            tmp.flush()

        # Try page-wise text extraction via pdfplumber
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    ptext = page.extract_text() or ""
                    if ptext.strip():
                        text += ptext + "\n"
        except Exception:
            # keep going to OCR if pdfplumber fails
            pass

        # Convert pages to images
        pil_pages = convert_from_path(tmp_path, dpi=dpi)
        for idx, pil in enumerate(pil_pages, start=1):
            # convert to cv2 BGR
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            # OCR preprocessing & OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if max(h, w) < 1500:
                scale = 2.0
                gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

            gray = cv2.GaussianBlur(gray, (3,3), 0)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 11)

            ocr_text = pytesseract.image_to_string(th, config="--oem 3 --psm 6")
            ocr_pages.append(ocr_text)
            if ocr_text and len(ocr_text.strip())>5:
                text += ocr_text + "\n"

            # ---------- IMAGE-BASED DETECTIONS (label-aware) ----------
            red_dots = detect_red_dots(img)

            # find label bbox (function already looks for the exact phrases)
            sig_bbox_pdf = find_signature_region_from_pdf(tmp_path, page_no=idx-1)

            # defaults
            final_sig_flag = False
            final_sig_conf = 0.0
            signature_expected_but_missing = False
            crop_ocr = ""
            crop_ocr_clean = ""
            ink_frac = None
            max_area = None

            # Helper: detect a local ink/blob in a crop — returns (has_ink, ink_fraction, max_contour_area)
            def crop_has_local_ink(cv2_img, min_frac=0.0015, min_contour_area=800):
                try:
                    gray_local = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                    gray_local = cv2.GaussianBlur(gray_local, (3,3), 0)
                    # invert so ink becomes white on black
                    _, th_local = cv2.threshold(gray_local, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    kernel_local = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                    th_clean = cv2.morphologyEx(th_local, cv2.MORPH_OPEN, kernel_local, iterations=1)
                    th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_DILATE, kernel_local, iterations=1)

                    contours_local, _ = cv2.findContours(th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    total = float(cv2_img.shape[0] * cv2_img.shape[1])
                    total_ink = 0.0
                    max_area_local = 0.0
                    big_count = 0
                    for c in contours_local:
                        a = float(cv2.contourArea(c))
                        total_ink += a
                        if a > max_area_local:
                            max_area_local = a
                        if a >= min_contour_area:
                            big_count += 1

                    ink_fraction_local = total_ink / total if total else 0.0
                    # require at least one reasonably large connected ink blob AND decent ink fraction
                    has_ink_local = (big_count >= 1) and (ink_fraction_local >= min_frac)
                    return bool(has_ink_local), float(ink_fraction_local), float(max_area_local)
                except Exception:
                    return False, 0.0, 0.0

            # Helper: OCR check that looks for real name vs label text
            def ocr_has_meaningful_text(cv2_img):
                try:
                    gray_tmp = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                    _, th_tmp = cv2.threshold(gray_tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    txt = pytesseract.image_to_string(th_tmp, config="--oem 3 --psm 6")
                    cleaned = re.sub(r'[^\w\s\-\']+', ' ', (txt or "")).strip()
                    return txt, cleaned
                except Exception:
                    return "", ""

            if sig_bbox_pdf:
                # Map normalized bbox -> pixel crop
                pixel_box = pdf_bbox_to_image_pixels(sig_bbox_pdf, pil)
                if pixel_box:
                    x0, y0, x1, y1 = pixel_box
                    try:
                        crop = img[y0:y1, x0:x1]
                        # save crop for manual inspection
                        try:
                            cv2.imwrite(f"/tmp/sig_crop_page{idx}.png", crop)
                        except Exception:
                            pass

                        # OCR the crop for debug and label detection
                        raw_txt, crop_ocr_clean = ocr_has_meaningful_text(crop)
                        crop_ocr = raw_txt

                        # If crop OCR contains the label word "signature" (in any language),
                        # we don't accept OCR-only as signature. We require a real ink blob below.
                        contains_label_word = False
                        if re.search(r'\bsignat|signature|signatura\b', (crop_ocr_clean or "").lower()):
                            contains_label_word = True

                        # Count words (if it's long printed paragraph -> definitely not signature)
                        words = re.findall(r'\w+', crop_ocr_clean)
                        word_count = len(words)

                        # If crop looks like long printed text, treat as missing
                        if word_count > 10:
                            final_sig_flag = False
                            final_sig_conf = 0.0
                            signature_expected_but_missing = True
                            ink_frac = None
                            max_area = None
                        else:
                            # run stricter ink check: require large connected blob + min ink fraction
                            has_ink, ink_frac, max_area = crop_has_local_ink(
                                crop,
                                min_frac=0.0015,      # stricter
                                min_contour_area=800  # require a bigger connected component
                            )

                            # Accept signature only if we have real ink blob AND crop is not only the label text
                            if has_ink and (not contains_label_word):
                                final_sig_flag = True
                                final_sig_conf = 0.95
                            else:
                                # if contains label word but we also have a strong ink blob below label -> accept
                                if has_ink and contains_label_word and max_area and max_area >= 1200:
                                    final_sig_flag = True
                                    final_sig_conf = 0.93
                                else:
                                    final_sig_flag = False
                                    final_sig_conf = 0.0

                            signature_expected_but_missing = not final_sig_flag

                    except Exception:
                        final_sig_flag = False
                        final_sig_conf = 0.0
                        signature_expected_but_missing = True
                        ink_frac = None
                        max_area = None
                else:
                    final_sig_flag = False
                    final_sig_conf = 0.0
                    signature_expected_but_missing = True
                    ink_frac = None
                    max_area = None
            else:
                # If no label present, do NOT scan whole page — mark missing
                final_sig_flag = False
                final_sig_conf = 0.0
                signature_expected_but_missing = False
                crop_ocr = ""
                crop_ocr_clean = ""
                ink_frac = None
                max_area = None

            # save safe debug values (guard round against None)
            debug_ink_frac = round(ink_frac, 6) if (ink_frac is not None) else None
            debug_max_area = round(max_area, 2) if (max_area is not None) else None

            images_info.append({
                "page": idx,
                "has_signature_image": bool(final_sig_flag),
                "signature_confidence": float(final_sig_conf),
                "red_dots_count": int(red_dots),
                "label_present": bool(sig_bbox_pdf),
                "signature_detected": bool(final_sig_flag),
                "signature_expected_but_missing": bool(signature_expected_but_missing),
                "debug": {
                    "crop_ocr": crop_ocr,
                    "crop_ocr_clean": crop_ocr_clean,
                    "ink_fraction": debug_ink_frac,
                    "max_ink_contour_area": debug_max_area
                }
            })










    except Exception as e:
        # log for debugging (Streamlit-friendly)
        st.error("Error during PDF extraction. Check logs.")
        st.text(traceback.format_exc())
    finally:
        # optionally remove tmp file - safe cleanup
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return {"text": text, "images_info": images_info, "ocr_pages": ocr_pages}

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




def pdf_bbox_to_image_pixels(bbox_norm_or_abs, pil_image, pdf_width=None, pdf_height=None):
    img_w, img_h = pil_image.size

    if not bbox_norm_or_abs:
        return None

    x0, y0, x1, y1 = bbox_norm_or_abs[0], bbox_norm_or_abs[1], bbox_norm_or_abs[2], bbox_norm_or_abs[3]

    # treat as normalized only if all coords are in [0.0, 1.0]
    is_normalized = all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1))

    if is_normalized:
        x0_px = int(max(0, round(x0 * img_w)))
        x1_px = int(min(img_w, round(x1 * img_w)))
        y0_px = int(max(0, round(y0 * img_h)))
        y1_px = int(min(img_h, round(y1 * img_h)))
    else:
        if not pdf_width or not pdf_height:
            return None
        scale_x = img_w / float(pdf_width)
        scale_y = img_h / float(pdf_height)
        x0_px = int(max(0, round(x0 * scale_x)))
        x1_px = int(min(img_w, round(x1 * scale_x)))
        y0_px = int(max(0, round(y0 * scale_y)))
        y1_px = int(min(img_h, round(y1 * scale_y)))

    # sanity: if the crop covers most of the page, treat as invalid (avoid whole-page false positives)
    crop_w = max(0, x1_px - x0_px)
    crop_h = max(0, y1_px - y0_px)
    if crop_w <= 0 or crop_h <= 0:
        return None
    crop_area = crop_w * crop_h
    page_area = img_w * img_h
    if crop_area >= 0.85 * page_area:
        # crop is almost full page — don't trust it as a "label crop"
        return None

    return (x0_px, y0_px, x1_px, y1_px)



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
}
"""





def sop_llm_extraction(dgd_text, msds_texts, image_findings_summary=""):
    msds_blocks = ""
    for i, msds in enumerate(msds_texts, start=1):
        msds_blocks += "\n------ MSDS DOCUMENT #{} ------\n{}\n".format(i, msds)

    # ---------------- SAFE PROMPT (NO PYTHON CODE INSIDE THE PROMPT) ----------------
    prompt = """
You are an expert in Dangerous Goods (DGD) & MSDS validation.

------ SOP RULES ------
{SOP_TEXT}

------ DGD DOCUMENT ------
{DGD_TEXT}

------ MSDS DOCUMENTS ------
{MSDS_BLOCKS}

------ IMAGE FINDINGS ------
{IMAGE_FINDINGS}

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

──────────────────────────────────────────
2) LANGUAGE DETECTION RULES
──────────────────────────────────────────
- Detect languages ONLY from real extracted text.
- If DGD contains a mixture (ex: French + English), return all detected languages.
- Do NOT hallucinate translation or assume document language.

──────────────────────────────────────────
3) SIGNATURE VALIDATION RULES
──────────────────────────────────────────
A valid signature exists ONLY IF:
- A person's name appears AND
- It is next to or under a signature field OR
- Sewing, stamp, or actual sign marks appear.
- SIGNATURE DU DELARENT 
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
A valid emergency phone number MUST:
- Contain minimum 7 digits
- May include “+”, “-”, spaces, parentheses

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
- Gross weight MUST be > net weight.
If invalid → status = "Weight incorrect".

──────────────────────────────────────────
6) MANDATORY BASIC DATA RULES
──────────────────────────────────────────
Required order:
UN NUMBER → Proper Shipping Name → Technical Name (if NOS or marine pollutant) → Class → PG.

Technical name rules:
- If NOS or Marine Pollutant is mentioned → technical name MUST appear in parenthesis.
- If technical name missing while NOS or marine pollutant present → RED FAIL.

──────────────────────────────────────────
7) PACKING CODE RULES (VERY STRICT)
──────────────────────────────────────────
Valid packing codes MUST appear EXACTLY and LITERALLY in the DGD text.
list of Existing ONU Packing code (Outturn packing) This is MANDATORY on ALL DGD 

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

Do NOT infer packing code from packaging description.
Return packing_code_found = "no" unless literal code appears.



──────────────────────────────────────────
8) INNER PACKING RULES (Appendix 6)
──────────────────────────────────────────
- REQUIRED only when OUTTURN PACKING CODE begins with “4…”
- NEVER hallucinate inner packing.
- Extract it ONLY if explicitly written.
- If outer packing starts with “4…” and NONE detected → status = "Inner packing missing".

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
Applies ONLY to Class 3.

VALID FORMATS:
- A single °C value (e.g., 23 °C)
- °C value followed by °F in parenthesis (e.g., 10 °C (50 °F))

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

──────────────────────────────────────────
12) MSDS VALIDITY
──────────────────────────────────────────
- Extract actual MSDS date.
- Must be < 5 years old.

──────────────────────────────────────────
13) MARINE POLLUTANT RULES
──────────────────────────────────────────
If Marine Pollutant YES:
- Technical name MUST be present.
- If missing → RED

Else → GREEN if handled correctly.

------ VALIDATION FORMAT ------
{JSON_TEMPLATE}
""".format(
        SOP_TEXT=sop_text,
        DGD_TEXT=dgd_text,
        MSDS_BLOCKS=msds_blocks,
        IMAGE_FINDINGS=image_findings_summary,
        JSON_TEMPLATE=JSON_TEMPLATE
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
.main-title { font-size: 32px; font-weight: 700; margin-bottom: -10px; }
.subtitle { font-size: 16px; color: #444; }
.upload-box { border: 2px dashed #c7d3df; padding: 20px; border-radius: 12px; text-align: center; }
.result-row { padding: 12px 0; border-bottom: 1px solid #eee; }
.label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>DGD & MSDS Validation Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your DGD & MSDS files for instant SOP validation ⚡</div>", unsafe_allow_html=True)

# ------------------ Upload Panel ------------------
left, right = st.columns([2.2, 1], gap="large")

with left:
    st.markdown("<div class='upload-box'>Upload DGD & MSDS PDFs below</div>", unsafe_allow_html=True)
    uploaded_dgd = st.file_uploader("Upload DGD PDF", type=["pdf"])
    uploaded_msds = st.file_uploader("Upload MSDS PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True)
    country = st.selectbox("Select Destination Country",
                           ["France", "Belgium", "Netherlands", "Ivory Coast",
                            "Algeria", "Brazil", "Mexico", "Peru", "Chile"])

    run_validation = st.button("Validate Now", use_container_width=True)

with right:
    st.markdown("### Recommendations")




# ------------------ VALIDATION PROCESS ------------------
if run_validation:

    if not uploaded_dgd:
        st.error("Please upload a DGD PDF before validating.")
        st.stop()

    with st.spinner("Extracting & validating..."):

        # Use the enhanced extractor (image + OCR) if you added it; otherwise use existing extractor
        dgd_result = extract_pdf_text_and_image_info(uploaded_dgd, dpi=300)  # if you implemented this helper
        dgd_text = dgd_result.get("text", "") if isinstance(dgd_result, dict) else dgd_result

        # ---------------- DEBUG OUTPUT ----------------
        # st.write("DEBUG images_info:", json.dumps(dgd_result.get("images_info", []), indent=2))
        # st.write("DEBUG ocr_pages (first 3):", [p[:300] for p in dgd_result.get("ocr_pages", [])[:3]])

        # detect packing codes deterministically in the DGD text
        detected_packing = find_packing_codes(dgd_text)

        # build image findings summary (if you have image info)
        image_summary = ""
        if isinstance(dgd_result, dict):
            img_info = dgd_result.get("images_info", [])
            image_summary = "\n".join([
                f"Page {it['page']}: signature_image={it['has_signature_image']}, sig_conf={it['signature_confidence']:.2f}, red_dots={it['red_dots_count']}"
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
                        f"MSDS Page {it['page']}: red_dots={it['red_dots_count']}, sig={it['has_signature_image']}"
                        for it in r.get("images_info", [])
                    ])





        
        # send image_summary into your LLM call (we assume sop_llm_extraction accepts it)
        llm_output = sop_llm_extraction(dgd_text, msds_texts, image_findings_summary=image_summary)



        cleaned = llm_output.strip().replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
        except:
            st.error("LLM returned invalid JSON.")
            st.code(llm_output)
            st.stop()


# ----------------- Deterministic signature override (SAFE GUARDED) -----------------
if run_validation and 'dgd_result' in locals():
    try:
        # get image-info from the DGD extractor
        dgd_img_info = dgd_result.get("images_info", []) if isinstance(dgd_result, dict) else []

        # pages where our extractor thinks there's an image-signature
        pages_with_image_sig = [it["page"] for it in dgd_img_info if it.get("signature_detected")]

        # pages where a label exists but no signature image was detected under it
        label_but_no_image = [
            it["page"] for it in dgd_img_info
            if it.get("label_present") and not it.get("signature_detected")
        ]

        # Decide deterministically (ONLY image under label counts as signature present)
        if pages_with_image_sig:
            pages_str = ", ".join(map(str, pages_with_image_sig))
            parsed["signature_check"] = {
                "status": "Signature present (image)",
                "details": f"Signature image detected on page(s): {pages_str}."
            }

        elif label_but_no_image:
            pages_str = ", ".join(map(str, label_but_no_image))
            parsed["signature_check"] = {
                "status": "Signature missing",
                "details": f"Signature label present but no signature image found under label on page(s): {pages_str}."
            }

        else:
            parsed["signature_check"] = {
                "status": "Signature missing",
                "details": "No signature label or signature image detected in the DGD. (Text-only occurrences are NOT accepted.)"
            }

    except Exception:
        # Never crash the app due to override logic
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
        "missing", "fail", "incorrect", "invalid",
        "not found", "inner packing missing"
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
    if errors:
        st.markdown("#### 🔴 Errors")
        for item in errors:
            st.markdown(f"- {item}")
    else:
        st.markdown("#### 🔴 Errors")
        st.markdown("*No errors detected.*")










# ------------------ RESULT PANEL ------------------
st.markdown("### Validation Results")
res_left, res_right = st.columns([2.3, 1], gap="large")

with res_left:

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
        ("presence_of_documents", "Presence of GGD + MSDS"),
        ("language_check", "Language Check"),
        ("signature_check", "Signature"),
        ("emergency_contact_check", "Emergency Contact"),
        ("weight_check", "Weight Check"),   # ✅ newly added
        ("mandatory_basic_data", "Mandatory Basic Data"),
        ("packing_code_check", "Packing Code"),
        ("inner_packing_check", "Inner Packing"),
        ("flash_point_check", "Flash Point"),
        ("segregation_group_check", "Segregation Group"),
        ("msds_match_check", "UN Match"),
        ("msds_validity", "MSDS Validity"),
        ("marine_pollutant_check", "Marine Pollutant"),
        ("un_restriction_check", "UN Restriction Check")
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
                <div style='margin-left:40px; color:#bbb'>{details}</div>
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





