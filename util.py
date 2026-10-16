import string
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Load the character model shipped with this repository.
char_detector = YOLO(str(Path(__file__).resolve().parent / 'models' / 'character_detector.pt'))

# Letters actually used on Vietnamese license plates.
VN_PLATE_LETTERS = set('ABCDEFGHKLMNPRSTUVXYZ')

OCR_TO_DIGIT = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'A': '4',
    'S': '5',
    'G': '6',
    'B': '8'
}

OCR_TO_ALPHA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}

VN_PLATE_PATTERNS = [
    {'slots': 'DDADDDDDD', 'formatter': lambda text: f'{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}'},
    {'slots': 'DDAADDDDD', 'formatter': lambda text: f'{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}'},
    {'slots': 'DDADDDDD', 'formatter': lambda text: f'{text[:2]}-{text[2:3]} {text[3:6]}.{text[6:]}'},
    {'slots': 'DDAADDDD', 'formatter': lambda text: f'{text[:2]}-{text[2:4]} {text[4:]}'},
    {'slots': 'DDADDDD', 'formatter': lambda text: f'{text[:2]}-{text[2:3]} {text[3:]}'}
]


def normalize_plate_text(text):
    """
    Normalize OCR output before validating Vietnamese plate formats.

    Args:
        text (str): OCR output.

    Returns:
        str: Upper-cased alphanumeric string without separators.
    """
    if not isinstance(text, str):
        return ''
    return ''.join(ch for ch in text.upper() if ch.isalnum())


def coerce_by_slots(text, slots):
    """
    Force a text string to match a pattern of digits ('D') and alphas ('A').
    Uses float penalties to favor highly likely OCR misreads (like Z->2).
    """
    if len(text) != len(slots):
        return None

    candidate = ''
    replacements = 0.0

    for char, slot in zip(text, slots):
        if slot == 'D':
            if char.isdigit():
                candidate += char
            elif char in OCR_TO_DIGIT:
                candidate += OCR_TO_DIGIT[char]
                if char in ('Z', 'B', 'O', 'I', 'L'):
                    replacements += 0.4
                else:
                    replacements += 1.0
            else:
                return None
        else:
            if char in string.ascii_uppercase:
                candidate += char
                if char not in VN_PLATE_LETTERS:
                    replacements += 1.0
                elif char == 'Z':
                    replacements += 0.8
                elif char in ('B', 'O', 'I'):
                    replacements += 0.2
            elif char in OCR_TO_ALPHA:
                candidate += OCR_TO_ALPHA[char]
                if char in ('2', '8', '0', '1'):
                    replacements += 0.4
                else:
                    replacements += 1.0
            else:
                return None

    return candidate, replacements


def parse_vietnamese_plate(text):
    """
    Validate and format OCR output as a Vietnamese license plate candidate.
    """
    normalized = normalize_plate_text(text)

    best_match = None

    for pattern in VN_PLATE_PATTERNS:
        result = coerce_by_slots(normalized, pattern['slots'])
        if result is None:
            continue

        candidate, replacements = result
        if best_match is None or replacements < best_match['replacements']:
            best_match = {
                'formatted': pattern['formatter'](candidate),
                'normalized': candidate,
                'replacements': replacements,
            }

    return best_match


def format_vietnamese_plate(text):
    """
    Validate and format OCR output as a Vietnamese license plate.
    """
    match = parse_vietnamese_plate(text)
    if match is None:
        return None

    return match['formatted']


def extract_text_from_chars(boxes, names):
    """
    Converts a list of YOLO bounding boxes into a structured text string.
    Groups characters into lines and sorts them horizontally.
    """
    # Filter out low confidence noise (screws, background dirt)
    valid_boxes = [b for b in boxes if b[4] > 0.4]
    if not valid_boxes:
        return "", 0.0
    
    # Calculate median height to define a line threshold
    heights = [box[3] - box[1] for box in valid_boxes]
    median_h = float(np.median(heights))
    
    # Sort boxes vertically (by center_y) to group into lines
    valid_boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    
    lines = []
    current_line = []
    
    for box in valid_boxes:
        center_y = (box[1] + box[3]) / 2
        if not current_line:
            current_line.append(box)
        else:
            prev_center_y = (current_line[-1][1] + current_line[-1][3]) / 2
            # If the y difference is small relative to character height, it's the same line
            if abs(center_y - prev_center_y) < median_h * 0.5:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    
    if current_line:
        lines.append(current_line)
        
    text_lines = []
    total_conf = 0.0
    char_count = 0
    
    for line in lines:
        # Sort characters in the line horizontally (by center_x)
        line.sort(key=lambda b: (b[0] + b[2]) / 2)
        line_str = "".join([names[int(b[5])] for b in line])
        text_lines.append(line_str)
        for b in line:
            total_conf += b[4]
            char_count += 1
            
    combined_text = "".join(text_lines)
    avg_conf = total_conf / char_count if char_count > 0 else 0.0
    
    return combined_text, avg_conf


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image using YOLO Character Detector.
    """
    if license_plate_crop is None or license_plate_crop.size == 0 or char_detector is None:
        return None, None

    # Detect characters
    results = char_detector(license_plate_crop, verbose=False)[0]
    boxes = results.boxes.data.tolist()
    
    # Extract assembled raw text
    raw_text, avg_conf = extract_text_from_chars(boxes, char_detector.names)
    if not raw_text:
        return None, None
        
    # Validate and format with existing heuristics
    parsed = parse_vietnamese_plate(raw_text)
    if parsed:
        # Deduct replacement penalty to handle ties gracefully (e.g. B vs 8)
        score = avg_conf - (parsed['replacements'] * 0.15)
        return parsed['formatted'], score

    return None, None
