"""Extract structured chat messages from a MapleStory screenshot."""

import argparse
import json
import sys
from pathlib import Path

deps = Path(__file__).resolve().with_name(".deps")
if deps.is_dir():
    sys.path.insert(0, str(deps))

import cv2
import numpy as np


def read_image(path):
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def select_roi(image):
    title = "Select chat area"
    print("Drag over the chat area. Press Enter to confirm or C to cancel.", file=sys.stderr)
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        height, width = image.shape[:2]
        scale = min(1, 1100 / width, 700 / height)
        cv2.resizeWindow(title, max(1, round(width * scale)), max(1, round(height * scale)))
        roi = cv2.selectROI(title, image, printNotice=False)
    finally:
        cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        raise ValueError("Chat area selection cancelled.")
    return roi


def crop_image(image, roi):
    x, y, width, height = roi
    image_height, image_width = image.shape[:2]
    if min(x, y) < 0 or min(width, height) <= 0 or (
        x + width > image_width or y + height > image_height
    ):
        raise ValueError(f"ROI must be a positive-size area within the {image_width}x{image_height} image.")
    return image[y : y + height, x : x + width]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the screenshot")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    selection.add_argument("--select-roi", action="store_true", help="Select the chat area with the mouse")
    parser.add_argument("-o", "--output", type=Path, help="Write UTF-8 JSON to a file instead of stdout")
    args = parser.parse_args()

    try:
        if args.output and args.output.resolve() == args.image.resolve():
            raise ValueError("The output path must be different from the input image.")
        image = read_image(args.image)
        roi = select_roi(image) if args.select_roi else args.roi
        if roi is not None:
            image = crop_image(image, roi)

        from maplescreen_ocr import extract_messages

        payload = json.dumps(extract_messages(image), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8", newline="\n")
        else:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stdout.write(payload)
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
