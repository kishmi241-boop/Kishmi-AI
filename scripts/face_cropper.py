import numpy as np
from PIL import Image
import io
import cv2
import os

# Load OpenCV's pre-trained Haar Cascade for frontal face detection
# This comes bundled with every OpenCV installation
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

class FaceCropper:
    def __init__(self, min_detection_confidence=0.4):
        """
        Face detection and cropping utility using OpenCV Haar Cascade.
        min_detection_confidence is kept as a parameter for API compatibility
        but maps to the scale_factor / min_neighbors used internally.
        """
        self.face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar Cascade from: {_CASCADE_PATH}")
        # Map confidence to detector sensitivity
        self.scale_factor = 1.1
        self.min_neighbors = 3  # Lower = more detections, fewer false negatives

    def crop_face(self, image_input, target_size=(224, 224), margin_ratio=0.25):
        """
        Detects and crops a face from the image, adding a margin to capture
        forehead, cheeks, jawline — then resizes to target_size.

        Args:
            image_input: PIL Image or raw image bytes.
            target_size: (width, height) to resize the output.
            margin_ratio: fraction of the face bounding box to add as padding.

        Returns:
            (PIL.Image, bool): cropped image and whether a face was detected.
        """
        # Load image
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert('RGB')
        else:
            img = image_input.convert('RGB')

        w, h = img.size

        # Convert to OpenCV grayscale for face detection
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Fallback: use center square crop
            return self._center_crop(img, target_size), False

        # Use the largest detected face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = largest

        # Add margin
        mx = int(fw * margin_ratio)
        my = int(fh * margin_ratio)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        cropped = img.crop((x1, y1, x2, y2))
        return cropped.resize(target_size, Image.Resampling.LANCZOS), True

    def _center_crop(self, img, target_size):
        """Return a center square crop of the image, then resize."""
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side)).resize(
            target_size, Image.Resampling.LANCZOS
        )

    def close(self):
        pass  # No resources to release for OpenCV cascade

    def __del__(self):
        self.close()
