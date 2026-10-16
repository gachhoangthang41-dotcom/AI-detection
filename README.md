# Vietnamese License Plate Recognition

A local image API for detecting and reading Vietnamese license plates. The pipeline uses one YOLO model to locate the plate and a second YOLO character detector to read its letters and digits. FastAPI returns the recognized text, bounding box, and confidence scores.

## Pipeline

1. Decode an uploaded image or a base64 JSON image.
2. Detect plate boxes on the original image and selected OpenCV variants. Small images may be enlarged with bicubic interpolation; CLAHE improves contrast.
3. Try several crops around each detected plate. The character model predicts individual letters and digits.
4. Group characters into lines by their coordinates, correct likely digit/letter substitutions, and format candidates against Vietnamese plate patterns.
5. If plate detection finds no readable result, try character recognition on a few larger image regions.

The character detector was trained from `yolov8n.pt` for 30 character classes. Its saved training configuration is in [`training/character_detector_args.yaml`](training/character_detector_args.yaml). The configuration specifies 50 epochs, a 320 px image size, and batch size 16. The repository does not contain validation metrics or a controlled before/after comparison, so no accuracy improvement is claimed.

## Repository contents

| Path | Purpose |
| --- | --- |
| `api.py` | FastAPI endpoints and plate detection pipeline |
| `util.py` | Character ordering and plate-format postprocessing |
| `models/license_plate_detector.pt` | Plate detector weights |
| `models/character_detector.pt` | Character detector weights |
| `training/character_detector_args.yaml` | Saved character training configuration |

Both model weight files are included so the API can run without training. The source datasets, local virtual environment, and temporary request images are excluded.

## Run locally

Python 3.10 or newer is recommended. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python api.py
```

On Windows, `start_server.bat` starts the same API after the virtual environment is created. The server listens on `127.0.0.1:26813` by default. Set `AI_API_PORT` and `AI_API_HOST` to override the address. Ultralytics and PyTorch use a CUDA GPU when available and otherwise run on CPU.

Check `http://127.0.0.1:26813/health` or open `http://127.0.0.1:26813/docs` for the interactive API documentation.

Upload an image:

```powershell
curl.exe -X POST -F "file=@C:\path\to\plate.jpg" http://127.0.0.1:26813/detect
```

Or send a base64 image:

```powershell
$encoded = [Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\path\to\plate.jpg'))
$body = @{ imageBase64 = $encoded } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:26813/detect' -ContentType 'application/json' -Body $body
```

`POST /api/plate-recognition` is an alias for `POST /detect`. A successful response includes `plate_text`, `confidence`, `detections`, and `plate_detector_variant`. When no plate is readable, `detections` is empty and `plate_text` is `null`.

## Datasets and limitations

The local training directory includes plate and character data from Roboflow projects such as [Vietnamese Car License Plate](https://universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate/dataset/1) and [OCR](https://universe.roboflow.com/licence-plate-vn/ocr-stypd/dataset/1). Those images are not redistributed here. Consult the original datasets and their licenses before reusing them for training.

Recognition is heuristic after detection: supported plate formats and character substitutions are defined in `util.py`. Images with severe blur, occlusion, unusual plate layouts, or multiple overlapping plates can produce incorrect text. The returned confidence is a model score, not a measured probability that the complete plate is correct.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## License

The source repository includes the GNU Affero General Public License v3; see [`LICENSE`](LICENSE).
