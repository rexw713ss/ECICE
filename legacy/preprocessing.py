import argparse
import gc
import json
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    print("Missing OpenCV or NumPy.")
    print(r"Install with: .\env\Scripts\python.exe -m pip install opencv-python-headless numpy")
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "dataset" / "OCR_test_lin.jpg"
SAVE_FOLDER = BASE_DIR / "output" / "legacy"

IMAGE_STEM = IMAGE_PATH.stem
DOCUMENT_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_document_input.jpg"
PREPROCESSED_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_preprocessed_input.jpg"
ENHANCED_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_enhanced_input.jpg"
SHARPENED_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_sharpened_input.jpg"
LINE_REMOVED_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_line_removed_input.jpg"
BLUE_INK_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_blue_ink_input.jpg"
BLUE_BINARY_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_blue_binary_input.jpg"
BLUE_MASK_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_blue_mask.jpg"
MERGED_JSON_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_merged_ocr.json"
MERGED_TEXT_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_merged_ocr.txt"
MERGED_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_merged_ocr.jpg"
REVIEW_IMAGE_PATH = SAVE_FOLDER / f"{IMAGE_STEM}_merged_ocr_review.jpg"

MIN_CANDIDATE_SCORE = 0.12
TILE_CONFIGS = ((520, 200), (340, 140), (240, 90))
TILE_SOURCES = ("document", "enhanced", "blue_ink", "blue_binary")


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess OCR_test_lin.jpg and run multi-pass PaddleOCR."
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only create the corrected/enhanced OCR input images.",
    )
    return parser.parse_args()


def import_paddleocr():
    try:
        from paddleocr import PaddleOCR, PPStructureV3
    except ModuleNotFoundError as exc:
        print("Missing PaddleOCR.")
        print(r"Install with: .\env\Scripts\python.exe -m pip install paddleocr paddlepaddle")
        print(r"If PPStructureV3 asks for Paddlex, also run:")
        print(r'.\env\Scripts\python.exe -m pip install "paddlex[ocr]==3.6.1"')
        raise SystemExit(1) from exc
    return PaddleOCR, PPStructureV3


def create_layout_engine():
    _, PPStructureV3 = import_paddleocr()
    try:
        return PPStructureV3(
            lang="ch",
            device="cpu",
            enable_mkldnn=False,
            text_recognition_batch_size=8,
        )
    except RuntimeError as exc:
        if "dependency error" in str(exc).lower():
            print("PPStructureV3 dependency error.")
            print(r'.\env\Scripts\python.exe -m pip install "paddlex[ocr]==3.6.1"')
        else:
            print(f"Failed to create PPStructureV3: {exc}")
        raise SystemExit(1) from exc


def create_ocr_engine():
    PaddleOCR, _ = import_paddleocr()
    return PaddleOCR(
        lang="ch",
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=8,
        text_det_limit_side_len=2048,
        text_det_limit_type="max",
        text_det_thresh=0.10,
        text_det_box_thresh=0.22,
        text_det_unclip_ratio=1.8,
        text_rec_score_thresh=0.0,
    )


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def order_points(points):
    points = np.asarray(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def four_point_transform(image, points):
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_width = max(1, int(round(max(width_a, width_b))))
    max_height = max(1, int(round(max(height_a, height_b))))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def find_document_quad(image):
    height, width = image.shape[:2]
    scale = 1000.0 / max(height, width)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    _, light_mask = cv2.threshold(
        lightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    paper_mask = cv2.inRange(hsv, (0, 0, 55), (179, 135, 255))
    paper_mask = cv2.bitwise_and(paper_mask, light_mask)
    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=2,
    )
    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    image_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < image_area * 0.18:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.018, 0.025, 0.035, 0.05):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32") / scale

        box = cv2.boxPoints(cv2.minAreaRect(contour))
        return box.astype("float32") / scale

    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])


def rectify_document(image):
    quad = find_document_quad(image)
    warped = four_point_transform(image, quad)
    if warped.shape[0] < warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def normalize_illumination(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(lightness, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(normalized)
    return cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return cv2.addWeighted(image, 1.45, blurred, -0.45, 0)


def remove_small_components(mask, min_area=5, max_area=18000):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for index in range(1, component_count):
        area = stats[index, cv2.CC_STAT_AREA]
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        if min_area <= area <= max_area and width >= 2 and height >= 2:
            cleaned[labels == index] = 255
    return cleaned


def create_blue_ink_variants(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_from_hsv = cv2.inRange(hsv, (85, 35, 20), (138, 255, 250))

    blue_channel, green_channel, red_channel = cv2.split(image)
    blue_minus_red = blue_channel.astype(np.int16) - red_channel.astype(np.int16)
    blue_minus_green = blue_channel.astype(np.int16) - green_channel.astype(np.int16)
    blue_dominant = (
        (blue_minus_red > 8)
        & (blue_minus_green > -30)
        & (red_channel < 185)
        & (green_channel < 195)
    ).astype("uint8") * 255

    mask = cv2.bitwise_or(blue_from_hsv, blue_dominant)
    mask = cv2.medianBlur(mask, 3)
    mask = remove_small_components(mask)

    soft_mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )

    blue_on_white = np.full_like(image, 255)
    blue_on_white[soft_mask > 0] = image[soft_mask > 0]

    black_on_white = np.full_like(image, 255)
    black_on_white[mask > 0] = (0, 0, 0)
    return blue_on_white, black_on_white, mask


def remove_horizontal_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverse = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9,
    )
    kernel_width = max(80, image.shape[1] // 6)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )
    horizontal_lines = cv2.morphologyEx(
        inverse, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )
    horizontal_lines = cv2.dilate(
        horizontal_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return cv2.inpaint(image, horizontal_lines, 3, cv2.INPAINT_TELEA)


def create_binary_variant(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        13,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def create_image_variants(image_path):
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    document = rectify_document(original)
    enhanced = normalize_illumination(document)
    sharpened = sharpen_image(enhanced)
    line_removed = remove_horizontal_lines(sharpened)
    blue_ink, blue_binary, blue_mask = create_blue_ink_variants(sharpened)
    binary = create_binary_variant(sharpened)

    save_image(DOCUMENT_IMAGE_PATH, document)
    save_image(ENHANCED_IMAGE_PATH, enhanced)
    save_image(SHARPENED_IMAGE_PATH, sharpened)
    save_image(LINE_REMOVED_IMAGE_PATH, line_removed)
    save_image(BLUE_INK_IMAGE_PATH, blue_ink)
    save_image(BLUE_BINARY_IMAGE_PATH, blue_binary)
    save_image(BLUE_MASK_IMAGE_PATH, blue_mask)
    save_image(PREPROCESSED_IMAGE_PATH, binary)

    return {
        "document": document,
        "enhanced": enhanced,
        "sharpened": sharpened,
        "line_removed": line_removed,
        "blue_ink": blue_ink,
        "blue_binary": blue_binary,
        "binary": binary,
    }


def make_vertical_tiles(image, tile_height, tile_overlap):
    height = image.shape[0]
    tile_height = min(tile_height, height)
    step = max(1, tile_height - tile_overlap)
    offsets = list(range(0, max(height - tile_height, 0) + 1, step))
    last_offset = max(0, height - tile_height)
    if not offsets or offsets[-1] != last_offset:
        offsets.append(last_offset)

    return [(offset, image[offset : offset + tile_height, :]) for offset in offsets]


def result_to_data(result):
    data = result.json
    if callable(data):
        data = data()
    return data.get("res", data)


def box_values(box):
    return [float(value) for value in box[:4]]


def extract_candidates(
    result_data,
    source,
    *,
    y_offset=0,
    tile_height=None,
    image_height=None,
):
    candidates = []
    texts = result_data.get("rec_texts", [])
    scores = result_data.get("rec_scores", [])
    boxes = result_data.get("rec_boxes", [])

    for text, score, box in zip(texts, scores, boxes):
        text = str(text).strip()
        score = float(score)
        if not text or score < MIN_CANDIDATE_SCORE or len(box) < 4:
            continue

        x1, y1, x2, y2 = box_values(box)
        if tile_height is not None and image_height is not None:
            touches_top = y1 <= 6 and y_offset > 0
            touches_bottom = (
                y2 >= tile_height - 6
                and y_offset + tile_height < image_height
            )
            if touches_top or touches_bottom:
                continue

        y1 += y_offset
        y2 += y_offset
        if x2 <= x1 or y2 <= y1:
            continue

        candidates.append(
            {
                "text": text,
                "confidence": round(score, 6),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "source": source,
            }
        )
    return candidates


def run_ocr_passes(ocr_engine, variants):
    candidates = []
    full_sources = [
        "document",
        "enhanced",
        "sharpened",
        "line_removed",
        "blue_ink",
        "blue_binary",
        "binary",
    ]
    full_images = [variants[source] for source in full_sources]

    print("Running full-page OCR passes...")
    for source, result in zip(full_sources, ocr_engine.predict(full_images)):
        result_data = result_to_data(result)
        source_candidates = extract_candidates(result_data, f"{source}_full")
        candidates.extend(source_candidates)
        print(f"  {source}: {len(source_candidates)} candidates")

    print("Running overlapping tile OCR passes...")
    for source_name in TILE_SOURCES:
        image = variants[source_name]
        image_height = image.shape[0]
        for tile_height, tile_overlap in TILE_CONFIGS:
            tiles = make_vertical_tiles(image, tile_height, tile_overlap)
            tile_images = [tile for _, tile in tiles]
            source = f"{source_name}_tile_{tile_height}"
            tile_candidates = []
            for (offset, tile), result in zip(tiles, ocr_engine.predict(tile_images)):
                result_data = result_to_data(result)
                tile_candidates.extend(
                    extract_candidates(
                        result_data,
                        source,
                        y_offset=offset,
                        tile_height=tile.shape[0],
                        image_height=image_height,
                    )
                )
            candidates.extend(tile_candidates)
            print(f"  {source}: {len(tile_candidates)} candidates")

    return candidates


def run_layout_analysis(layout_engine):
    print("Running layout OCR on rectified document...")
    page_results = layout_engine.predict(
        str(DOCUMENT_IMAGE_PATH),
        text_det_thresh=0.12,
        text_det_box_thresh=0.25,
        text_rec_score_thresh=0.0,
    )
    candidates = []

    for page_number, page_result in enumerate(page_results, start=1):
        page_data = result_to_data(page_result)
        blocks = page_data.get("parsing_res_list", [])
        print(f"  page {page_number}: {len(blocks)} layout blocks")

        overall_ocr = page_data.get("overall_ocr_res", {})
        candidates.extend(extract_candidates(overall_ocr, "layout_ocr"))

        page_result.save_to_img(str(SAVE_FOLDER))
        page_result.save_to_json(str(SAVE_FOLDER))

    return candidates


def candidate_quality(candidate):
    text_length = min(len(candidate["text"].replace(" ", "")), 100)
    source = candidate["source"]
    if "_tile_" in source:
        source_bonus = 0.08
    else:
        source_bonus = {
            "document_full": 0.09,
            "enhanced_full": 0.08,
            "sharpened_full": 0.08,
            "blue_ink_full": 0.08,
            "line_removed_full": 0.06,
            "blue_binary_full": 0.06,
            "binary_full": 0.03,
            "layout_ocr": 0.03,
        }.get(source, 0.0)
    return candidate["confidence"] + text_length * 0.005 + source_bonus


def same_text_region(first, second):
    ax1, ay1, ax2, ay2 = first["bbox"]
    bx1, by1, bx2, by2 = second["bbox"]
    a_width, a_height = ax2 - ax1, ay2 - ay1
    b_width, b_height = bx2 - bx1, by2 - by1
    width_ratio = min(a_width, b_width) / max(1.0, max(a_width, b_width))
    if width_ratio < 0.42:
        return False

    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    smaller_width = max(1.0, min(a_width, b_width))
    horizontal_overlap = x_overlap / smaller_width

    a_center_y = (ay1 + ay2) / 2
    b_center_y = (by1 + by2) / 2
    center_distance = abs(a_center_y - b_center_y)
    center_limit = max(28.0, min(a_height, b_height) * 0.5)

    return horizontal_overlap >= 0.50 and center_distance <= center_limit


def choose_best_alternative(alternatives):
    highest_confidence = max(item["confidence"] for item in alternatives)
    eligible = [
        item
        for item in alternatives
        if item["confidence"] >= highest_confidence - 0.28
    ]
    return max(
        eligible,
        key=lambda item: (
            len(item["text"].replace(" ", "")),
            item["confidence"],
            candidate_quality(item),
        ),
    )


def merge_candidates(candidates):
    groups = []
    for candidate in sorted(candidates, key=candidate_quality, reverse=True):
        matching_group = next(
            (
                group
                for group in groups
                if any(same_text_region(candidate, item) for item in group)
            ),
            None,
        )
        if matching_group is None:
            groups.append([candidate])
        else:
            matching_group.append(candidate)

    merged = []
    for group in groups:
        unique_alternatives = {}
        for candidate in group:
            text = candidate["text"]
            existing = unique_alternatives.get(text)
            if existing is None or candidate["confidence"] > existing["confidence"]:
                unique_alternatives[text] = candidate

        alternatives = sorted(
            unique_alternatives.values(), key=candidate_quality, reverse=True
        )
        best = choose_best_alternative(alternatives)
        merged.append(
            {
                "text": best["text"],
                "confidence": best["confidence"],
                "bbox": best["bbox"],
                "source": best["source"],
                "alternatives": alternatives,
            }
        )

    merged.sort(key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0]))
    for index, item in enumerate(merged, start=1):
        item["id"] = index
    return merged


def select_clean_results(merged):
    clean = []
    for item in merged:
        source_count = len(
            {alternative["source"] for alternative in item["alternatives"]}
        )
        is_corroborated = source_count >= 2
        is_binary_only = item["source"] == "binary_full" and not is_corroborated
        is_short_low_confidence = len(item["text"].strip()) <= 1 and item["confidence"] < 0.65
        if is_short_low_confidence or is_binary_only:
            continue
        if item["confidence"] >= 0.40 or (is_corroborated and item["confidence"] >= 0.30):
            clean.append(item)
    return clean


def draw_results(image, results):
    visualized = image.copy()
    for item in results:
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
        confidence = item["confidence"]
        color = (40, 180, 40) if confidence >= 0.65 else (0, 180, 255)
        if confidence < 0.4:
            color = (0, 0, 255)
        cv2.rectangle(visualized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            visualized,
            f"{item['id']}:{confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return visualized


def save_merged_results(image, candidates, merged):
    clean_results = select_clean_results(merged)
    high_confidence = sum(item["confidence"] >= 0.65 for item in merged)
    result = {
        "input_path": str(IMAGE_PATH),
        "document_input_path": str(DOCUMENT_IMAGE_PATH),
        "settings": {
            "minimum_candidate_score": MIN_CANDIDATE_SCORE,
            "tile_configs": [
                {"height": height, "overlap": overlap}
                for height, overlap in TILE_CONFIGS
            ],
            "tile_sources": list(TILE_SOURCES),
            "strategy": (
                "rectified document + illumination normalization + blue-ink "
                "isolation + full-page/tile OCR + candidate merging"
            ),
        },
        "statistics": {
            "raw_candidate_count": len(candidates),
            "merged_region_count": len(merged),
            "clean_region_count": len(clean_results),
            "high_confidence_region_count": high_confidence,
        },
        "raw_candidates": candidates,
        "clean_results": clean_results,
        "merged_results": merged,
    }
    with MERGED_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    with MERGED_TEXT_PATH.open("w", encoding="utf-8") as file:
        for item in clean_results:
            file.write(f"{item['text']}\n")

    save_image(MERGED_IMAGE_PATH, draw_results(image, clean_results))
    save_image(REVIEW_IMAGE_PATH, draw_results(image, merged))


def main():
    configure_console()
    args = parse_args()
    if not IMAGE_PATH.is_file():
        print(f"Image not found: {IMAGE_PATH}")
        raise SystemExit(1)

    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    try:
        variants = create_image_variants(IMAGE_PATH)
    except (ValueError, OSError) as exc:
        print(f"Preprocessing failed: {exc}")
        raise SystemExit(1) from exc

    print("Preprocessing complete.")
    print(f"  document: {DOCUMENT_IMAGE_PATH}")
    print(f"  blue ink: {BLUE_INK_IMAGE_PATH}")
    print(f"  blue binary: {BLUE_BINARY_IMAGE_PATH}")

    if args.preprocess_only:
        return

    layout_engine = create_layout_engine()
    layout_candidates = run_layout_analysis(layout_engine)
    del layout_engine
    gc.collect()

    ocr_engine = create_ocr_engine()
    ocr_candidates = run_ocr_passes(ocr_engine, variants)

    candidates = layout_candidates + ocr_candidates
    merged = merge_candidates(candidates)
    save_merged_results(variants["document"], candidates, merged)

    print("\n--- OCR summary ---")
    print(f"Raw candidates: {len(candidates)}")
    print(f"Merged regions: {len(merged)}")
    print(f"Merged JSON: {MERGED_JSON_PATH}")
    print(f"Clean text: {MERGED_TEXT_PATH}")
    print(f"Clean review image: {MERGED_IMAGE_PATH}")
    print(f"All-candidate review image: {REVIEW_IMAGE_PATH}")


if __name__ == "__main__":
    main()
