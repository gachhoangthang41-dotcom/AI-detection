import asyncio
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import UploadFile

import api
from util import char_detector, extract_text_from_chars, parse_vietnamese_plate


class PlateProcessingTests(unittest.TestCase):
    def test_character_labels_come_from_the_saved_model(self):
        self.assertEqual(char_detector.names[0], "0")
        self.assertEqual(char_detector.names[29], "Z")
        boxes = [[0, 0, 10, 10, 0.9, 0], [12, 0, 22, 10, 0.9, 10]]
        text, _ = extract_text_from_chars(boxes, char_detector.names)
        self.assertEqual(text, "0A")

    def test_plate_parser_formats_vietnamese_plate(self):
        result = parse_vietnamese_plate("30A12345")
        self.assertIsNotNone(result)
        self.assertEqual(result["formatted"], "30-A 123.45")

    def test_api_limits_crop_upscaling_and_returns_a_plate(self):
        seen_shapes = []

        def fake_read_license_plate(crop):
            seen_shapes.append(crop.shape)
            return "30-A 123.45", 0.9

        image = np.zeros((80, 80, 3), dtype=np.uint8)
        encoded, payload = cv2.imencode(".png", image)
        self.assertTrue(encoded)
        upload = UploadFile(file=io.BytesIO(payload.tobytes()), filename="plate.png")

        with patch.object(api, "_run_detector", return_value=([[10, 10, 30, 30, 0.9, 0]], "original")), patch.object(api, "read_license_plate", side_effect=fake_read_license_plate):
            response = asyncio.run(api.detect_license_plate(request=None, file=upload, image=None))

        self.assertEqual(response["plate_text"], "30-A 123.45")
        self.assertTrue(seen_shapes)
        self.assertLessEqual(max(max(shape[:2]) for shape in seen_shapes), 320)


if __name__ == "__main__":
    unittest.main()
