import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:
    print("Missing OpenCV.")
    print(r"Install with: .\env\Scripts\python.exe -m pip install opencv-python-headless")
    raise SystemExit(1) from exc

from image_preprocessing import DEFAULT_IMAGE_PATH, DEFAULT_OUTPUT_DIR, preprocess_image

MIN_CANDIDATE_SCORE = 0.25
TILE_CONFIGS = ((520, 200), (340, 140), (240, 90))
GRID_CONFIGS = ((900, 180, 520, 180),)
FULL_SOURCES = (
    "document",
    "line_removed",
    "blue_ink",
    "blue_gray",
)
TILE_SOURCES = ("blue_gray",)
GRID_SOURCES = ("blue_gray",)


def import_paddleocr():
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        print("Missing PaddleOCR.")
        print(r"Install with: .\env\Scripts\python.exe -m pip install paddleocr paddlepaddle")
        raise SystemExit(1) from exc
    return PaddleOCR


def parse_source_list(value, default_sources):
    if value is None:
        return tuple(default_sources)
    sources = tuple(source.strip() for source in value.split(",") if source.strip())
    if not sources:
        raise ValueError("Source list cannot be empty.")
    return sources


def parse_tile_configs(value):
    if value is None:
        return tuple(TILE_CONFIGS)
    configs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid tile config '{item}'. Use height:overlap.")
        height, overlap = item.split(":", 1)
        configs.append((int(height), int(overlap)))
    if not configs:
        raise ValueError("Tile config list cannot be empty.")
    return tuple(configs)


def parse_grid_configs(value):
    if value is None:
        return tuple(GRID_CONFIGS)
    configs = []
    for item in value.split(","):
        values = [part.strip() for part in item.split(":")]
        if len(values) != 4:
            raise ValueError(
                f"Invalid grid config '{item}'. Use width:x_overlap:height:y_overlap."
            )
        configs.append(tuple(int(value) for value in values))
    if not configs:
        raise ValueError("Grid config list cannot be empty.")
    return tuple(configs)


def create_ocr_engine(
    det_limit_side_len=2048,
    use_textline_orientation=True,
    det_thresh=0.25,
    det_box_thresh=0.35,
    det_unclip_ratio=1.45,
):
    PaddleOCR = import_paddleocr()
    return PaddleOCR(
        # 如果你的環境有安裝繁體包，可以嘗試 lang="chinese_cht"，若無則維持 "ch"
        lang="ch", 
        
        # 啟用文字行傾斜偵測與方向分類
        use_textline_orientation=use_textline_orientation,
        
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_recognition_batch_size=8,
        text_det_limit_side_len=det_limit_side_len,
        text_det_limit_type="max",
        
        # -----------------------------
        # 修改 2：文字偵測 (Detection) 參數微調
        # -----------------------------
        # text_det_thresh: 判定為文字區塊的門檻 (原0.10 -> 0.35)，過濾橫線雜訊
        text_det_thresh=det_thresh,
        
        # text_det_box_thresh: 輸出框的門檻 (原0.22 -> 0.40)
        text_det_box_thresh=det_box_thresh,
        
        # text_det_unclip_ratio: 【關鍵！】框線擴張比例。
        # 你的筆記行距非常小，原本的 1.8 會讓上下行黏在一起。降到 1.5 甚至 1.4，可以切出更乾淨的單行。
        text_det_unclip_ratio=det_unclip_ratio,
        
        text_rec_score_thresh=0.0,
    )


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def load_manifest(image_path, output_dir, skip_preprocess=False):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    manifest_path = output_dir / f"{image_path.stem}_preprocess_manifest.json"
    if skip_preprocess and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
    return preprocess_image(image_path, output_dir)


def load_variants(manifest):
    variants = {}
    for key, path in manifest["variants"].items():
        if key == "blue_mask":
            continue
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read preprocessed variant: {path}")
        variants[key] = image
    return variants


def make_vertical_tiles(image, tile_height, tile_overlap):
    height = image.shape[0]
    tile_height = min(tile_height, height)
    step = max(1, tile_height - tile_overlap)
    offsets = list(range(0, max(height - tile_height, 0) + 1, step))
    last_offset = max(0, height - tile_height)
    if not offsets or offsets[-1] != last_offset:
        offsets.append(last_offset)
    return [(offset, image[offset : offset + tile_height, :]) for offset in offsets]


def make_grid_tiles(image, tile_width, x_overlap, tile_height, y_overlap):
    height, width = image.shape[:2]
    tile_width = min(tile_width, width)
    tile_height = min(tile_height, height)
    x_step = max(1, tile_width - x_overlap)
    y_step = max(1, tile_height - y_overlap)

    x_offsets = list(range(0, max(width - tile_width, 0) + 1, x_step))
    y_offsets = list(range(0, max(height - tile_height, 0) + 1, y_step))
    last_x = max(0, width - tile_width)
    last_y = max(0, height - tile_height)
    if not x_offsets or x_offsets[-1] != last_x:
        x_offsets.append(last_x)
    if not y_offsets or y_offsets[-1] != last_y:
        y_offsets.append(last_y)

    return [
        (x, y, image[y : y + tile_height, x : x + tile_width])
        for y in y_offsets
        for x in x_offsets
    ]


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
    x_offset=0,
    y_offset=0,
    tile_width=None,
    tile_height=None,
    image_width=None,
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
                y2 >= tile_height - 6 and y_offset + tile_height < image_height
            )
            if touches_top or touches_bottom:
                continue
        if tile_width is not None and image_width is not None:
            touches_left = x1 <= 6 and x_offset > 0
            touches_right = x2 >= tile_width - 6 and x_offset + tile_width < image_width
            if touches_left or touches_right:
                continue

        x1 += x_offset
        x2 += x_offset
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


def run_full_page_ocr(ocr_engine, variants, full_sources=FULL_SOURCES):
    candidates = []
    images = [variants[source] for source in full_sources if source in variants]
    sources = [source for source in full_sources if source in variants]

    print("Running full-page OCR...", flush=True)
    for source, result in zip(sources, ocr_engine.predict(images)):
        source_candidates = extract_candidates(result_to_data(result), f"{source}_full")
        candidates.extend(source_candidates)
        print(f"  {source}: {len(source_candidates)} candidates", flush=True)
    return candidates


def run_tile_ocr(ocr_engine, variants, tile_sources=TILE_SOURCES, tile_configs=TILE_CONFIGS):
    candidates = []
    print("Running overlapping tile OCR...", flush=True)
    for source_name in tile_sources:
        if source_name not in variants:
            continue
        image = variants[source_name]
        image_height = image.shape[0]
        for tile_height, tile_overlap in tile_configs:
            tiles = make_vertical_tiles(image, tile_height, tile_overlap)
            tile_images = [tile for _, tile in tiles]
            source = f"{source_name}_tile_{tile_height}"
            tile_candidates = []
            for (offset, tile), result in zip(tiles, ocr_engine.predict(tile_images)):
                tile_candidates.extend(
                    extract_candidates(
                        result_to_data(result),
                        source,
                        y_offset=offset,
                        tile_height=tile.shape[0],
                        image_height=image_height,
                    )
                )
            candidates.extend(tile_candidates)
            print(f"  {source}: {len(tile_candidates)} candidates", flush=True)
    return candidates


def run_grid_ocr(ocr_engine, variants, grid_sources=GRID_SOURCES, grid_configs=GRID_CONFIGS):
    candidates = []
    print("Running overlapping grid OCR...", flush=True)
    for source_name in grid_sources:
        if source_name not in variants:
            continue
        image = variants[source_name]
        image_height, image_width = image.shape[:2]
        for tile_width, x_overlap, tile_height, y_overlap in grid_configs:
            tiles = make_grid_tiles(
                image, tile_width, x_overlap, tile_height, y_overlap
            )
            tile_images = [tile for _, _, tile in tiles]
            source = f"{source_name}_grid_{tile_width}x{tile_height}"
            grid_candidates = []
            for (x_offset, y_offset, tile), result in zip(
                tiles, ocr_engine.predict(tile_images)
            ):
                grid_candidates.extend(
                    extract_candidates(
                        result_to_data(result),
                        source,
                        x_offset=x_offset,
                        y_offset=y_offset,
                        tile_width=tile.shape[1],
                        tile_height=tile.shape[0],
                        image_width=image_width,
                        image_height=image_height,
                    )
                )
            candidates.extend(grid_candidates)
            print(f"  {source}: {len(grid_candidates)} candidates", flush=True)
    return candidates


def normalize_candidate_text(text):
    return re.sub(r"[\s，。；：、,.!?！？()（）]+", "", text)


def candidate_quality(candidate):
    text_length = min(len(candidate["text"].replace(" ", "")), 100)
    source = candidate["source"]
    if "_grid_" in source:
        source_bonus = 0.11
    elif "_tile_" in source:
        source_bonus = 0.08
    else:
        source_bonus = {
            "document_full": 0.09,
            "enhanced_full": 0.08,
            "sharpened_full": 0.08,
            "clean_document_full": 0.09,
            "blue_ink_full": 0.08,
            "blue_gray_full": 0.10,
            "line_removed_full": 0.06,
            "blue_binary_full": 0.06,
            "binary_full": 0.03,
        }.get(source, 0.0)
    return candidate["confidence"] + text_length * 0.002 + source_bonus


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
        item for item in alternatives if item["confidence"] >= highest_confidence - 0.28
    ]
    normalized = {
        id(item): normalize_candidate_text(item["text"])
        for item in eligible
    }

    def consensus_score(item):
        item_text = normalized[id(item)]
        similarities = [
            SequenceMatcher(None, item_text, normalized[id(other)]).ratio()
            for other in eligible
            if other is not item
        ]
        agreement = sum(similarities) / len(similarities) if similarities else 0.0
        return candidate_quality(item) + agreement * 0.22

    return max(eligible, key=lambda item: (consensus_score(item), item["confidence"]))


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


def save_ocr_results(output_dir, image_stem, document_image, candidates, merged, settings):
    output_dir = Path(output_dir)
    clean_results = select_clean_results(merged)
    paths = {
        "json": output_dir / f"{image_stem}_merged_ocr.json",
        "text": output_dir / f"{image_stem}_merged_ocr.txt",
        "clean_review": output_dir / f"{image_stem}_merged_ocr.jpg",
        "all_review": output_dir / f"{image_stem}_merged_ocr_review.jpg",
    }
    result = {
        "settings": {
            "minimum_candidate_score": MIN_CANDIDATE_SCORE,
            "tile_configs": [
                {"height": height, "overlap": overlap}
                for height, overlap in settings["tile_configs"]
            ],
            "full_sources": list(settings["full_sources"]),
            "tile_sources": list(settings["tile_sources"]),
            "grid_configs": [
                {
                    "width": width,
                    "x_overlap": x_overlap,
                    "height": height,
                    "y_overlap": y_overlap,
                }
                for width, x_overlap, height, y_overlap in settings["grid_configs"]
            ],
            "grid_sources": list(settings["grid_sources"]),
            "det_limit_side_len": settings["det_limit_side_len"],
            "use_textline_orientation": settings["use_textline_orientation"],
            "det_thresh": settings["det_thresh"],
            "det_box_thresh": settings["det_box_thresh"],
            "det_unclip_ratio": settings["det_unclip_ratio"],
        },
        "statistics": {
            "raw_candidate_count": len(candidates),
            "merged_region_count": len(merged),
            "clean_region_count": len(clean_results),
            "high_confidence_region_count": sum(
                item["confidence"] >= 0.65 for item in merged
            ),
            "candidate_count_by_source": dict(
                sorted(Counter(item["source"] for item in candidates).items())
            ),
        },
        "raw_candidates": candidates,
        "clean_results": clean_results,
        "merged_results": merged,
    }
    with paths["json"].open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with paths["text"].open("w", encoding="utf-8") as file:
        for item in clean_results:
            file.write(f"{item['text']}\n")

    save_image(paths["clean_review"], draw_results(document_image, clean_results))
    save_image(paths["all_review"], draw_results(document_image, merged))
    return {key: str(path) for key, path in paths.items()}


def run_ocr(
    image_path=DEFAULT_IMAGE_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    skip_preprocess=False,
    full_sources=FULL_SOURCES,
    tile_sources=TILE_SOURCES,
    tile_configs=TILE_CONFIGS,
    run_tiles=True,
    grid_sources=GRID_SOURCES,
    grid_configs=GRID_CONFIGS,
    run_grids=True,
    det_limit_side_len=2048,
    use_textline_orientation=True,
    det_thresh=0.25,
    det_box_thresh=0.35,
    det_unclip_ratio=1.45,
):
    manifest = load_manifest(image_path, output_dir, skip_preprocess=skip_preprocess)
    variants = load_variants(manifest)
    ocr_engine = create_ocr_engine(
        det_limit_side_len=det_limit_side_len,
        use_textline_orientation=use_textline_orientation,
        det_thresh=det_thresh,
        det_box_thresh=det_box_thresh,
        det_unclip_ratio=det_unclip_ratio,
    )
    candidates = run_full_page_ocr(ocr_engine, variants, full_sources=full_sources)
    if run_tiles:
        candidates.extend(
            run_tile_ocr(
                ocr_engine,
                variants,
                tile_sources=tile_sources,
                tile_configs=tile_configs,
            )
        )
    if run_grids:
        candidates.extend(
            run_grid_ocr(
                ocr_engine,
                variants,
                grid_sources=grid_sources,
                grid_configs=grid_configs,
            )
        )
    merged = merge_candidates(candidates)
    settings = {
        "full_sources": full_sources,
        "tile_sources": tile_sources if run_tiles else (),
        "tile_configs": tile_configs if run_tiles else (),
        "grid_sources": grid_sources if run_grids else (),
        "grid_configs": grid_configs if run_grids else (),
        "det_limit_side_len": det_limit_side_len,
        "use_textline_orientation": use_textline_orientation,
        "det_thresh": det_thresh,
        "det_box_thresh": det_box_thresh,
        "det_unclip_ratio": det_unclip_ratio,
    }
    return save_ocr_results(
        output_dir,
        manifest["image_stem"],
        variants["document"],
        candidates,
        merged,
        settings,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Module 2: OCR recognition for lecture notes.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Input note image.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder.")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse an existing preprocess manifest if available.",
    )
    parser.add_argument(
        "--full-sources",
        default=None,
        help="Comma-separated full-page OCR sources. Example: document,blue_ink",
    )
    parser.add_argument(
        "--tile-sources",
        default=None,
        help="Comma-separated tile OCR sources. Example: blue_ink",
    )
    parser.add_argument(
        "--tile-configs",
        default=None,
        help="Comma-separated height:overlap configs. Example: 520:200",
    )
    parser.add_argument("--no-tiles", action="store_true", help="Disable tile OCR.")
    parser.add_argument(
        "--grid-sources",
        default=None,
        help="Comma-separated grid OCR sources. Example: blue_gray",
    )
    parser.add_argument(
        "--grid-configs",
        default=None,
        help="Comma-separated width:x_overlap:height:y_overlap configs.",
    )
    parser.add_argument("--no-grids", action="store_true", help="Disable grid OCR.")
    parser.add_argument(
        "--det-limit-side-len",
        type=int,
        default=2048,
        help="PaddleOCR detection limit side length.",
    )
    parser.add_argument(
        "--no-textline-orientation",
        action="store_true",
        help="Disable text line orientation classification.",
    )
    parser.add_argument("--det-thresh", type=float, default=0.25)
    parser.add_argument("--det-box-thresh", type=float, default=0.35)
    parser.add_argument("--det-unclip-ratio", type=float, default=1.45)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = run_ocr(
        args.image,
        args.output_dir,
        skip_preprocess=args.skip_preprocess,
        full_sources=parse_source_list(args.full_sources, FULL_SOURCES),
        tile_sources=parse_source_list(args.tile_sources, TILE_SOURCES),
        tile_configs=parse_tile_configs(args.tile_configs),
        run_tiles=not args.no_tiles,
        grid_sources=parse_source_list(args.grid_sources, GRID_SOURCES),
        grid_configs=parse_grid_configs(args.grid_configs),
        run_grids=not args.no_grids,
        det_limit_side_len=args.det_limit_side_len,
        use_textline_orientation=not args.no_textline_orientation,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
        det_unclip_ratio=args.det_unclip_ratio,
    )
    print("OCR complete.")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
