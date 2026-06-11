import argparse
import json
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    print("Missing OpenCV or NumPy.")
    print(r"Install with: .\env\Scripts\python.exe -m pip install opencv-python-headless numpy")
    raise SystemExit(1) from exc


from pipeline_paths import DATASET_DIR, PREPROCESS_DIR


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = DATASET_DIR / "OCR_test_lin.jpg"
DEFAULT_OUTPUT_DIR = PREPROCESS_DIR


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
    _, light_mask = cv2.threshold(
        lab[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
        if cv2.contourArea(contour) < image_area * 0.18:
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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 20, 190, cv2.NORM_MINMAX)
    blue_gray = np.full_like(gray, 255)
    blue_gray[soft_mask > 0] = gray[soft_mask > 0]
    blue_gray = cv2.cvtColor(blue_gray, cv2.COLOR_GRAY2BGR)

    black_on_white = np.full_like(image, 255)
    black_on_white[mask > 0] = (0, 0, 0)
    return blue_on_white, blue_gray, black_on_white, mask


def suppress_red_annotations(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, (0, 45, 20), (15, 255, 255))
    high_red = cv2.inRange(hsv, (165, 45, 20), (179, 255, 255))
    red_mask = cv2.bitwise_or(low_red, high_red)
    red_mask = cv2.dilate(
        red_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return cv2.inpaint(image, red_mask, 3, cv2.INPAINT_TELEA)


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


def preprocess_image(image_path=DEFAULT_IMAGE_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_stem = image_path.stem
    document = rectify_document(original)
    enhanced = normalize_illumination(document)
    sharpened = sharpen_image(enhanced)
    line_removed = remove_horizontal_lines(sharpened)
    red_suppressed = suppress_red_annotations(sharpened)
    clean_document = remove_horizontal_lines(red_suppressed)
    blue_ink, blue_gray, blue_binary, blue_mask = create_blue_ink_variants(sharpened)
    binary = create_binary_variant(sharpened)

    outputs = {
        "document": output_dir / f"{image_stem}_document_input.jpg",
        "enhanced": output_dir / f"{image_stem}_enhanced_input.jpg",
        "sharpened": output_dir / f"{image_stem}_sharpened_input.jpg",
        "line_removed": output_dir / f"{image_stem}_line_removed_input.jpg",
        "clean_document": output_dir / f"{image_stem}_clean_document_input.jpg",
        "blue_ink": output_dir / f"{image_stem}_blue_ink_input.jpg",
        "blue_gray": output_dir / f"{image_stem}_blue_gray_input.jpg",
        "blue_binary": output_dir / f"{image_stem}_blue_binary_input.jpg",
        "blue_mask": output_dir / f"{image_stem}_blue_mask.jpg",
        "binary": output_dir / f"{image_stem}_preprocessed_input.jpg",
    }
    images = {
        "document": document,
        "enhanced": enhanced,
        "sharpened": sharpened,
        "line_removed": line_removed,
        "clean_document": clean_document,
        "blue_ink": blue_ink,
        "blue_gray": blue_gray,
        "blue_binary": blue_binary,
        "blue_mask": blue_mask,
        "binary": binary,
    }
    for key, path in outputs.items():
        save_image(path, images[key])

    manifest = {
        "input_path": str(image_path),
        "output_dir": str(output_dir),
        "image_stem": image_stem,
        "document_shape": {
            "height": int(document.shape[0]),
            "width": int(document.shape[1]),
        },
        "variants": {key: str(path) for key, path in outputs.items()},
    }
    manifest_path = output_dir / f"{image_stem}_preprocess_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Module 1: lecture note image preprocessing.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Input note image.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = preprocess_image(args.image, args.output_dir)
    print("Image preprocessing complete.")
    print(f"Manifest: {manifest['manifest_path']}")
    for key, path in manifest["variants"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
