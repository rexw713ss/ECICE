#!/usr/bin/env python3
"""Generate single-pass PaddleOCR baseline predictions from original images."""

import argparse
import json
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:
    print("Missing OpenCV.")
    print(r"Install with: .\env\Scripts\python.exe -m pip install opencv-python-headless")
    raise SystemExit(1) from exc

from ocr_recognition import import_paddleocr, result_to_data
from pipeline_paths import BASELINE_OCR_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one unmodified-image PaddleOCR pass as the CER experiment baseline. "
            "No preprocessing variants, tiling, ensemble merge, or LLM are used."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="An input image or a directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASELINE_OCR_DIR),
        help="Directory for *_paddleocr_baseline.txt/json files.",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="PaddleOCR language setting. Default: ch",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PaddleOCR device setting. Default: cpu",
    )
    return parser.parse_args()


def discover_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            return images
        raise ValueError(f"No supported images found in: {input_path}")
    raise ValueError(f"Input path does not exist: {input_path}")


def create_baseline_engine(lang="ch", device="cpu"):
    PaddleOCR = import_paddleocr()
    return PaddleOCR(
        lang=lang,
        device=device,
        enable_mkldnn=False,
    )


def extract_lines(result_data):
    texts = result_data.get("rec_texts", [])
    scores = result_data.get("rec_scores", [])
    boxes = result_data.get("rec_boxes", [])
    lines = []
    for text, score, box in zip(texts, scores, boxes):
        text = str(text).strip()
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "confidence": round(float(score), 6),
                "bbox": [round(float(value), 2) for value in box[:4]],
            }
        )
    return lines


def run_baseline(
    ocr_engine,
    image_path,
    output_dir,
    *,
    lang,
    device,
    output_stem=None,
    output_suffix="_paddleocr_baseline",
    method="single_pass_paddleocr_original_image",
    setting=None,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    predictions = list(ocr_engine.predict([image]))
    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected one PaddleOCR result for {image_path}, received {len(predictions)}."
        )
    lines = extract_lines(result_to_data(predictions[0]))

    stem = output_stem or image_path.stem
    text_path = output_dir / f"{stem}{output_suffix}.txt"
    json_path = output_dir / f"{stem}{output_suffix}.json"
    text_path.write_text(
        "".join(f"{line['text']}\n" for line in lines),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "method": method,
                **({"setting": setting} if setting is not None else {}),
                "input_path": str(image_path),
                "settings": {
                    "lang": lang,
                    "device": device,
                    "enable_mkldnn": False,
                },
                "statistics": {
                    "line_count": len(lines),
                    "character_count": sum(len(line["text"]) for line in lines),
                },
                "lines": lines,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return text_path, json_path


def main():
    args = parse_args()
    try:
        images = discover_images(args.input)
        output_dir = Path(args.output_dir)
        engine = create_baseline_engine(args.lang, args.device)
        for index, image_path in enumerate(images, start=1):
            print(f"[{index}/{len(images)}] PaddleOCR baseline: {image_path}", flush=True)
            text_path, json_path = run_baseline(
                engine,
                image_path,
                output_dir,
                lang=args.lang,
                device=args.device,
            )
            print(f"  text: {text_path}", flush=True)
            print(f"  json: {json_path}", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Baseline generation failed: {exc}") from exc


if __name__ == "__main__":
    main()
