import base64
import os
import math
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from util import read_license_plate

app = FastAPI(title="AI Detection API", description="API for License Plate Recognition")

# Public demo API; no cookie-based authentication is used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model files relative to this module so the API works from any cwd.
MODEL_DIR = Path(__file__).resolve().parent / "models"
YOLO_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Loading YOLO models on {YOLO_DEVICE}...")
license_plate_detector = YOLO(str(MODEL_DIR / 'license_plate_detector.pt'))
license_plate_detector.to(YOLO_DEVICE)
print("Models loaded successfully!")


@app.get("/health")
def health():
    return {"status": "ok", "device": YOLO_DEVICE}
def _decode_image_bytes(contents):
    """
    Decode raw image bytes into an OpenCV frame.

    Args:
        contents (bytes): Raw image bytes.

    Returns:
        np.ndarray | None: Decoded BGR image.
    """
    if not contents:
        return None

    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _try_decode_base64(value):
    """
    Decode a base64 string or data URL into image bytes.

    Args:
        value (str): Base64 string or data URL.

    Returns:
        bytes | None: Decoded bytes if successful.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    payload = value.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        return None


def _frame_from_json_payload(payload):
    """
    Try to build a frame from common JSON payload conventions.

    Args:
        payload (dict): JSON request body.

    Returns:
        tuple[np.ndarray | None, str | None]: Decoded frame and a source description.
    """
    if not isinstance(payload, dict):
        return None, None

    base64_keys = [
        "imageBase64", "image_base64", "base64", "fileBase64", "file_base64",
        "image", "file", "data",
    ]

    for key in base64_keys:
        if key not in payload:
            continue
        raw_bytes = _try_decode_base64(payload.get(key))
        frame = _decode_image_bytes(raw_bytes)
        if frame is not None:
            return frame, f"json:{key}"

    return None, None


def _prepare_detection_variants(frame):
    """
    Build image variants that make YOLO more robust on low-resolution UI captures.

    Only adds upscale variants when the image is small (< 500px on either side).
    For already-large images (e.g. pre-upscaled by C#), upscaling wastes CPU time
    and can actually hurt detection accuracy.

    Args:
        frame (np.ndarray): Original BGR image.

    Returns:
        list[dict]: Detection variants with transform metadata.
    """
    variants = [{
        "name": "original",
        "image": frame,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }]

    height, width = frame.shape[:2]
    needs_upscale = height < 500 or width < 500

    if needs_upscale:
        upscaled = cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        variants.append({
            "name": "upscale_2x",
            "image": upscaled,
            "scale_x": 2.0,
            "scale_y": 2.0,
        })

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    clahe_bgr = cv2.cvtColor(cv2.merge((clahe_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    variants.append({
        "name": "clahe",
        "image": clahe_bgr,
        "scale_x": 1.0,
        "scale_y": 1.0,
    })

    if needs_upscale:
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)
        variants.append({
            "name": "upscale_2x_sharpen",
            "image": sharpened,
            "scale_x": 2.0,
            "scale_y": 2.0,
        })

    return variants


def _run_detector(model, frame, allowed_classes=None):
    """
    Run YOLO on multiple image variants and keep the strongest result set.

    Args:
        model (YOLO): Loaded model.
        frame (np.ndarray): Original BGR image.
        allowed_classes (set[int] | None): Optional class filter.

    Returns:
        tuple[list, str]: Selected detections and the winning variant name.
    """
    best_boxes = []
    best_variant = "original"
    best_rank = (-1, -1.0)

    for variant in _prepare_detection_variants(frame):
        prediction = model.predict(
            variant["image"],
            conf=0.05,
            imgsz=640,
            device=YOLO_DEVICE,
            verbose=False,
        )[0]
        boxes = prediction.boxes.data.tolist()

        normalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            if allowed_classes is not None and int(class_id) not in allowed_classes:
                continue
            normalized_boxes.append([
                x1 / variant["scale_x"],
                y1 / variant["scale_y"],
                x2 / variant["scale_x"],
                y2 / variant["scale_y"],
                score,
                class_id,
            ])

        variant_rank = (
            len(normalized_boxes),
            max((box[4] for box in normalized_boxes), default=0.0),
        )
        if variant_rank > best_rank:
            best_boxes = normalized_boxes
            best_variant = variant["name"]
            best_rank = variant_rank

    return best_boxes, best_variant


def _fallback_plate_from_frame(frame):
    """
    Try OCR directly on the full image when object detectors miss the plate.

    Args:
        frame (np.ndarray): Original BGR image.

    Returns:
        tuple[str | None, float | None]: Plate text and OCR confidence.
    """
    crops = [frame]

    height, width = frame.shape[:2]
    crops.append(frame[0:height, 0:int(width * 0.8)])
    crops.append(frame[int(height * 0.15):int(height * 0.95), int(width * 0.05):int(width * 0.85)])

    best_text = None
    best_score = None
    for crop in crops:
        text, score = read_license_plate(crop)
        if text is not None and (best_score is None or (score or 0) > best_score):
            best_text = text
            best_score = score

    return best_text, best_score


def _expand_bbox(x1, y1, x2, y2, width, height, pad_x=0.08, pad_y=0.12):
    """
    Expand a bounding box slightly to avoid clipping plate characters.

    Args:
        x1, y1, x2, y2 (float): Bounding box coordinates.
        width (int): Frame width.
        height (int): Frame height.
        pad_x (float): Horizontal padding ratio.
        pad_y (float): Vertical padding ratio.

    Returns:
        tuple[int, int, int, int]: Clamped bounding box.
    """
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    return (
        max(int(x1 - box_w * pad_x), 0),
        max(int(y1 - box_h * pad_y), 0),
        min(int(x2 + box_w * pad_x), width),
        min(int(y2 + box_h * pad_y), height),
    )


def _build_plate_crop_boxes(x1, y1, x2, y2, frame_width, frame_height):
    """
    Build a small set of focused context boxes around a detected plate bbox.

    This avoids expensive full-frame OCR while still giving enough surrounding
    pixels for a two-line motorcycle plate.
    """
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    candidates = []

    # 1. Tight expansion around the detection.
    candidates.append(_expand_bbox(x1, y1, x2, y2, frame_width, frame_height, pad_x=0.15, pad_y=0.25))

    # 2. Moderate context around the plate center to capture full 2-line plates.
    crop_w = min(max(int(box_w * 1.35), 180), int(frame_width * 0.72))
    crop_h = min(max(int(box_h * 1.55), 180), int(frame_height * 0.76))
    context_x1 = max(int(center_x - crop_w / 2), 0)
    context_y1 = max(int(center_y - crop_h / 2), 0)
    context_x2 = min(context_x1 + crop_w, frame_width)
    context_y2 = min(context_y1 + crop_h, frame_height)
    candidates.append((context_x1, context_y1, context_x2, context_y2))

    # 3. Left-side crop, useful for bike-entry layouts where the plate dominates the left pane.
    candidates.append((
        max(int(frame_width * 0.02), 0),
        max(int(frame_height * 0.12), 0),
        min(int(frame_width * 0.78), frame_width),
        min(int(frame_height * 0.95), frame_height),
    ))

    deduped = []
    seen = set()
    for candidate in candidates:
        crop_x1, crop_y1, crop_x2, crop_y2 = candidate
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
        key = (crop_x1, crop_y1, crop_x2, crop_y2)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    return deduped

@app.post("/detect")
@app.post("/api/plate-recognition")
async def detect_license_plate(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    upload = file or image
    frame = None

    if upload is None:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            payload = await request.json()
            frame, _ = _frame_from_json_payload(payload)
            if frame is None:
                return {
                    "success": False,
                    "message": "JSON request did not contain a decodable image.",
                    "source": "python_api",
                }
        else:
            form = await request.form()
            for value in form.values():
                if isinstance(value, UploadFile):
                    upload = value
                    break
            if upload is None:
                return {
                    "success": False,
                    "message": "Missing uploaded image. Use multipart field name 'file' or 'image'.",
                    "source": "python_api",
                }

    if upload is not None:
        contents = await upload.read()
        frame = _decode_image_bytes(contents)
    
    if frame is None:
        return {
            "success": False,
            "message": "Invalid image file.",
            "source": "python_api",
        }

    # Detect license plates directly; vehicle detection is not needed here.
    license_plate_detections, plate_variant = _run_detector(license_plate_detector, frame)
    
    results = []
    frame_height, frame_width = frame.shape[:2]
    
    for license_plate in license_plate_detections:
        x1, y1, x2, y2, score, _class_id = license_plate

        best_plate_text = None
        best_plate_score = None
        best_plate_rank = None
        best_crop_box = None

        # 5. Try several context crops around the detected bbox
        crop_boxes = _build_plate_crop_boxes(x1, y1, x2, y2, frame_width, frame_height)
        for crop_index, (crop_x1, crop_y1, crop_x2, crop_y2) in enumerate(crop_boxes, start=1):
            if crop_index >= 3 and best_plate_text is not None:
                break

            license_plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]

            # Upscale tiny crops so OCR has enough pixel data to work with.
            crop_h, crop_w = license_plate_crop.shape[:2]
            if min(crop_h, crop_w) == 0:
                continue
            if crop_h < 100 or crop_w < 100:
                scale = min(4, max(2, math.ceil(100 / min(crop_h, crop_w))))
                license_plate_crop = cv2.resize(
                    license_plate_crop,
                    (crop_w * scale, crop_h * scale),
                    interpolation=cv2.INTER_CUBIC,
                )

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

            score_value = float(license_plate_text_score or 0.0)
            crop_rank = score_value
            if crop_index == 2:
                crop_rank += 0.12
            elif crop_index >= 3:
                crop_rank -= 0.35

            if license_plate_text is not None and (
                best_plate_rank is None or crop_rank > best_plate_rank
            ):
                best_plate_text = license_plate_text
                best_plate_score = score_value
                best_plate_rank = crop_rank
                best_crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]

            # Early exit: stop trying more crops once we have a good plate.
            if best_plate_text is not None and best_plate_score > 0.4:
                break

        if best_plate_text is not None:
            results.append({
                "car_bbox": None,
                "license_plate": {
                    "text": best_plate_text,
                    "text_score": best_plate_score,
                    "bbox": best_crop_box,
                    "bbox_score": score
                }
            })

    if not results:
        fallback_text, fallback_score = _fallback_plate_from_frame(frame)
        if fallback_text is not None:
            results.append({
                "car_bbox": None,
                "license_plate": {
                    "text": fallback_text,
                    "text_score": fallback_score,
                    "bbox": [0, 0, frame.shape[1], frame.shape[0]],
                    "bbox_score": 0.0,
                }
            })

    if results:
        results.sort(
            key=lambda item: (
                float(item["license_plate"].get("bbox_score") or 0.0),
                float(item["license_plate"].get("text_score") or 0.0)
                + (float(item["license_plate"].get("bbox_score") or 0.0) * 1.5),
                float(item["license_plate"].get("text_score") or 0.0),
                float(item["license_plate"].get("bbox_score") or 0.0),
            ),
            reverse=True,
        )

    best_detection = results[0] if results else None
    best_plate = best_detection["license_plate"]["text"] if best_detection else None
    best_score = best_detection["license_plate"]["text_score"] if best_detection else None

    response = {
        "success": True,
        "source": "python_api",
        "detections": results,
        "detections_count": len(results),
        "plate_detections_count": len(license_plate_detections),
        "plate_detector_variant": plate_variant,
        "plate_text": best_plate,
        "plate_number": best_plate,
        "license_number": best_plate,
        "licensePlateText": best_plate,
        "confidence": best_score,
    }

    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AI_API_PORT", "26813"))
    host = os.getenv("AI_API_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, reload=False)
