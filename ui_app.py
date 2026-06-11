#!/usr/bin/env python3
"""Local web UI for viewing ECICE image preprocessing and LLM summaries."""

import argparse
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

from pipeline_paths import (
    DATASET_DIR as DEFAULT_DATASET_DIR,
    OUTPUT_ROOT,
    build_stage_paths,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT
DATASET_DIR = DEFAULT_DATASET_DIR
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_STAGE_DIRECTORIES = (
    Path("01_preprocessing"),
    Path("02_ocr") / "baseline",
    Path("02_ocr") / "ablation",
    Path("02_ocr") / "ensemble",
    Path("03_llm_correction"),
    Path("04_evaluation"),
    Path("05_summary"),
    Path("06_quiz"),
)

VARIANT_LABELS = {
    "document": "文件校正",
    "enhanced": "光照增強",
    "sharpened": "銳化",
    "line_removed": "移除橫線",
    "clean_document": "乾淨文件",
    "blue_ink": "藍色筆跡",
    "blue_gray": "藍筆灰階",
    "blue_binary": "藍筆二值化",
    "blue_mask": "藍色遮罩",
    "binary": "二值化 OCR",
    "merged_ocr": "OCR 框線",
    "merged_ocr_review": "OCR 候選框線",
}

PREFERRED_PROCESSED_KEYS = (
    "binary",
    "clean_document",
    "line_removed",
    "sharpened",
    "enhanced",
    "document",
    "blue_ink",
    "blue_gray",
    "blue_binary",
    "merged_ocr",
)

INDEX_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>筆記摘要檢視</title>
  <style>
    :root {
      --bg: #f7f6f0;
      --surface: #ffffff;
      --surface-2: #f2f5f3;
      --ink: #1c2421;
      --muted: #66736e;
      --line: #d8ded9;
      --accent: #117a72;
      --accent-2: #c47b22;
      --shadow: 0 16px 40px rgba(29, 41, 37, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: "Microsoft JhengHei", "Noto Sans TC", "Segoe UI", Arial, sans-serif;
    }

    button,
    select {
      font: inherit;
    }

    .app-shell {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.84);
      backdrop-filter: blur(14px);
    }

    .brand {
      min-width: 180px;
    }

    .brand h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .brand p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .top-controls {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .select-wrap {
      display: grid;
      gap: 5px;
    }

    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    select {
      min-width: 220px;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 34px 0 12px;
      color: var(--ink);
      background: var(--surface);
    }

    .workspace {
      flex: 1;
      display: grid;
      grid-template-columns: minmax(340px, 52%) minmax(360px, 48%);
      gap: 18px;
      padding: 18px;
      min-height: 0;
    }

    .pane {
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .image-pane {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }

    .pane-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: var(--surface-2);
    }

    .segmented {
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #e8eeea;
    }

    .mode-button {
      min-width: 84px;
      height: 34px;
      border: 0;
      border-radius: 6px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-weight: 700;
    }

    .mode-button.active {
      color: #ffffff;
      background: var(--accent);
    }

    .variant-control {
      display: grid;
      gap: 5px;
      min-width: 190px;
    }

    .image-stage {
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      background: #eceae3;
    }

    .image-frame {
      min-height: 0;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 18px;
      overflow: auto;
    }

    .image-frame img {
      display: block;
      width: auto;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border: 1px solid rgba(28, 36, 33, 0.12);
      background: #ffffff;
    }

    .image-caption {
      padding: 10px 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      background: rgba(255, 255, 255, 0.82);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .summary-pane {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }

    .summary-head {
      padding: 15px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }

    .summary-head h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }

    .summary-section-tabs {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-start;
      gap: 6px;
      margin-top: 12px;
      max-width: 100%;
    }

    .section-button {
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 5px 10px;
      color: #33443e;
      background: var(--surface-2);
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.25;
    }

    .section-button.active {
      border-color: var(--accent);
      color: #ffffff;
      background: var(--accent);
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .pill {
      max-width: 100%;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface-2);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .summary-scroll {
      min-height: 0;
      overflow: auto;
      padding: 22px 28px 36px;
    }

    .markdown {
      max-width: 920px;
      color: #22302b;
      line-height: 1.72;
      font-size: 15px;
    }

    .markdown h1 {
      margin: 0 0 18px;
      font-size: 28px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .markdown h2 {
      margin: 28px 0 10px;
      padding-top: 6px;
      color: #143b37;
      font-size: 20px;
      letter-spacing: 0;
    }

    .markdown h2:first-child {
      margin-top: 0;
    }

    .markdown h3 {
      margin: 22px 0 8px;
      font-size: 17px;
      letter-spacing: 0;
    }

    .markdown p {
      margin: 0 0 12px;
    }

    .markdown ul,
    .markdown ol {
      margin: 0 0 16px 1.3em;
      padding: 0;
    }

    .markdown li {
      margin: 6px 0;
      padding-left: 2px;
    }

    .markdown strong {
      color: #111a17;
    }

    .markdown table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 20px;
      font-size: 14px;
    }

    .markdown th,
    .markdown td {
      border: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }

    .markdown th {
      background: #eaf2ef;
      color: #173b36;
    }

    .markdown tr:nth-child(even) td {
      background: #fafbf8;
    }

    .empty-state {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 180px;
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }

    @media (max-width: 920px) {
      .topbar {
        align-items: stretch;
        flex-direction: column;
      }

      .top-controls {
        justify-content: stretch;
      }

      .select-wrap,
      .select-wrap select,
      .variant-control,
      .variant-control select {
        width: 100%;
      }

      .workspace {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(420px, 48vh) minmax(460px, 1fr);
      }

      .pane-toolbar {
        align-items: stretch;
        flex-direction: column;
      }

      .summary-section-tabs {
        justify-content: flex-start;
      }

      .segmented {
        width: 100%;
      }

      .summary-scroll {
        padding: 18px;
      }
    }
  </style>
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand">
        <h1>筆記摘要檢視</h1>
        <p id="runStatus">載入中</p>
      </div>
      <div class="top-controls">
        <div class="select-wrap">
          <label for="runSelect">筆記結果</label>
          <select id="runSelect"></select>
        </div>
      </div>
    </header>

    <section class="workspace">
      <section class="pane image-pane" aria-label="Image preview">
        <div class="pane-toolbar">
          <div class="segmented" role="group" aria-label="Image mode">
            <button class="mode-button active" id="beforeButton" type="button">處理前</button>
            <button class="mode-button" id="afterButton" type="button">處理後</button>
          </div>
          <div class="variant-control" id="variantControl">
            <label for="variantSelect">處理版本</label>
            <select id="variantSelect"></select>
          </div>
        </div>
        <div class="image-stage">
          <div class="image-frame" id="imageFrame">
            <div class="empty-state">尚無圖片</div>
          </div>
          <div class="image-caption" id="imageCaption"></div>
        </div>
      </section>

      <section class="pane summary-pane" aria-label="LLM summary">
        <div class="summary-head">
          <h2>摘要生成</h2>
          <div class="summary-section-tabs" id="summarySectionTabs"></div>
        </div>
        <div class="summary-scroll">
          <article class="markdown" id="summaryContent"></article>
        </div>
      </section>
    </section>
  </main>

  <script>
    const runSelect = document.getElementById("runSelect");
    const variantSelect = document.getElementById("variantSelect");
    const variantControl = document.getElementById("variantControl");
    const beforeButton = document.getElementById("beforeButton");
    const afterButton = document.getElementById("afterButton");
    const imageFrame = document.getElementById("imageFrame");
    const imageCaption = document.getElementById("imageCaption");
    const summaryContent = document.getElementById("summaryContent");
    const summarySectionTabs = document.getElementById("summarySectionTabs");
    const runStatus = document.getElementById("runStatus");
    const ERROR_ANALYSIS_KEY = "__error_analysis__";

    let currentRun = null;
    let imageMode = "before";
    let processedKey = "";
    let summarySectionKey = "__all__";

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    }

    function setMode(mode) {
      imageMode = mode;
      beforeButton.classList.toggle("active", mode === "before");
      afterButton.classList.toggle("active", mode === "after");
      variantControl.style.visibility = mode === "after" ? "visible" : "hidden";
      updateImage();
    }

    function imageElement(src, alt) {
      const img = document.createElement("img");
      img.src = src;
      img.alt = alt;
      img.onerror = () => {
        imageFrame.innerHTML = '<div class="empty-state">圖片無法載入</div>';
      };
      return img;
    }

    function updateImage() {
      if (!currentRun) {
        imageFrame.innerHTML = '<div class="empty-state">尚無圖片</div>';
        imageCaption.textContent = "";
        return;
      }

      let imageInfo = null;
      if (imageMode === "before") {
        imageInfo = currentRun.original;
      } else {
        imageInfo = currentRun.processed_variants.find((item) => item.key === processedKey);
      }

      if (!imageInfo || !imageInfo.url) {
        imageFrame.innerHTML = '<div class="empty-state">這個結果沒有可顯示的圖片</div>';
        imageCaption.textContent = "";
        return;
      }

      imageFrame.innerHTML = "";
      imageFrame.appendChild(imageElement(imageInfo.url, imageInfo.label));
      imageCaption.textContent = `${imageInfo.label} · ${imageInfo.name}`;
    }

    function renderVariants(run) {
      variantSelect.innerHTML = "";
      for (const item of run.processed_variants) {
        const option = document.createElement("option");
        option.value = item.key;
        option.textContent = item.label;
        variantSelect.appendChild(option);
      }

      processedKey = run.default_processed_key || (run.processed_variants[0] && run.processed_variants[0].key) || "";
      variantSelect.value = processedKey;
      variantControl.style.display = run.processed_variants.length ? "grid" : "none";
      variantControl.style.visibility = imageMode === "after" ? "visible" : "hidden";
    }

    function updateSummaryButtons() {
      for (const button of summarySectionTabs.querySelectorAll(".section-button")) {
        button.classList.toggle("active", button.dataset.key === summarySectionKey);
      }
    }

    function appendTable(headers, rows) {
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const header of headers) {
        const cell = document.createElement("th");
        cell.textContent = header;
        headRow.appendChild(cell);
      }
      head.appendChild(headRow);
      table.appendChild(head);

      const body = document.createElement("tbody");
      for (const row of rows) {
        const tableRow = document.createElement("tr");
        for (const value of row) {
          const cell = document.createElement("td");
          cell.textContent = value;
          tableRow.appendChild(cell);
        }
        body.appendChild(tableRow);
      }
      table.appendChild(body);
      summaryContent.appendChild(table);
    }

    function formatPercent(value) {
      return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "N/A";
    }

    function renderErrorAnalysis(run) {
      summaryContent.innerHTML = "";
      const heading = document.createElement("h1");
      heading.textContent = "CER 與錯誤分析";
      summaryContent.appendChild(heading);

      const aggregate = run.evaluation_summary || {};
      const metricRows = ["paddleocr_baseline", "ensemble_only", "ensemble_llm"]
        .filter((method) => aggregate[method])
        .map((method) => [
          method,
          formatPercent(aggregate[method].micro_cer),
          formatPercent(aggregate[method].micro_character_accuracy),
        ]);
      if (metricRows.length) {
        const metricHeading = document.createElement("h2");
        metricHeading.textContent = "錯誤率與正確率";
        summaryContent.appendChild(metricHeading);
        appendTable(["Method", "Error Rate (CER)", "Character Accuracy"], metricRows);
      }

      let ablationRows = run.ablation_rows || [];
      const ablationHeading = document.createElement("h2");
      ablationHeading.textContent = "Ablation Study";
      summaryContent.appendChild(ablationHeading);
      if (!ablationRows.length) {
        const message = document.createElement("p");
        message.textContent = "尚無 ablation study，請先提供人工 ground truth 並執行 evaluation。";
        summaryContent.appendChild(message);
        ablationRows = [
          ["Raw image + PaddleOCR", "baseline"],
          ["+ document rectification", "只加文件矯正"],
          ["+ blue ink extraction", "看藍筆萃取是否有效"],
          ["+ line removal", "看去橫線是否有效"],
          ["+ multi-variant ensemble", "看 ensemble 是否有效"],
          ["+ LLM correction", "完整 correction stage"],
        ].map(([setting, description]) => ({ setting, description }));
      }
      appendTable(
        ["Setting", "CER ↓", "Accuracy ↑", "Δ CER vs baseline", "說明"],
        ablationRows.map((row) => [
          row.setting,
          formatPercent(row.cer),
          formatPercent(row.character_accuracy),
          formatPercent(row.delta_cer_vs_baseline),
          row.description,
        ]),
      );
      const ablationNote = document.createElement("p");
      ablationNote.textContent = "Δ CER < 0 表示相較 raw baseline 改善；單一 preprocessing variant 彼此不是累加設定。";
      summaryContent.appendChild(ablationNote);

      const analysisHeading = document.createElement("h2");
      analysisHeading.textContent = "錯誤分析表";
      summaryContent.appendChild(analysisHeading);
      let rows = run.error_analysis_rows || [];
      if (!rows.length) {
        const message = document.createElement("p");
        message.textContent = "尚無錯誤分析，請先提供人工 ground truth 並執行 CER evaluation。";
        summaryContent.appendChild(message);
        rows = ["繁簡混用", "相似字誤認", "缺字", "LLM hallucination"].map((errorType) => ({
          error_type: errorType,
          example: "尚未評估",
          raw_ocr: "-",
          corrected: "-",
          success: "不適用",
        }));
      }
      appendTable(
        ["Error Type", "Example", "Raw OCR", "Corrected", "是否成功"],
        rows.map((row) => [
          row.error_type,
          row.example,
          row.raw_ocr,
          row.corrected,
          row.success,
        ]),
      );
    }

    function showSummarySection(key) {
      if (!currentRun) {
        return;
      }

      summarySectionKey = key;
      if (key === ERROR_ANALYSIS_KEY) {
        renderErrorAnalysis(currentRun);
      } else if (key === "__all__") {
        summaryContent.innerHTML = currentRun.summary_html || '<p>尚無摘要內容</p>';
      } else {
        const section = (currentRun.summary_sections || []).find((item) => item.key === key);
        summaryContent.innerHTML = section ? section.html : '<p>尚無這個分類內容</p>';
      }

      const scrollArea = summaryContent.closest(".summary-scroll");
      if (scrollArea) {
        scrollArea.scrollTop = 0;
      }
      updateSummaryButtons();
    }

    function renderSummarySections(run) {
      summarySectionTabs.innerHTML = "";
      const sections = run.summary_sections || [];
      const buttons = [
        { key: "__all__", title: "全文" },
        ...sections,
        { key: ERROR_ANALYSIS_KEY, title: "CER 與錯誤分析" },
      ];

      for (const section of buttons) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "section-button";
        button.dataset.key = section.key;
        button.textContent = section.title;
        button.addEventListener("click", () => showSummarySection(section.key));
        summarySectionTabs.appendChild(button);
      }

      const defaultKey = run.default_summary_section_key || (sections[0] && sections[0].key) || "__all__";
      showSummarySection(defaultKey);
    }

    async function loadRun(stem) {
      currentRun = await fetchJson(`/api/runs/${encodeURIComponent(stem)}`);
      runStatus.textContent = currentRun.stem;
      renderVariants(currentRun);
      renderSummarySections(currentRun);
      updateImage();
    }

    async function boot() {
      try {
        const runs = await fetchJson("/api/runs");
        runSelect.innerHTML = "";
        if (!runs.length) {
          runStatus.textContent = "找不到輸出結果";
          summaryContent.innerHTML = '<p>output/05_summary 尚未產生 summary.md。</p>';
          return;
        }

        for (const run of runs) {
          const option = document.createElement("option");
          option.value = run.stem;
          option.textContent = run.display_name;
          runSelect.appendChild(option);
        }
        await loadRun(runs[0].stem);
      } catch (error) {
        runStatus.textContent = "載入失敗";
        summaryContent.innerHTML = `<p>${error.message}</p>`;
      }
    }

    beforeButton.addEventListener("click", () => setMode("before"));
    afterButton.addEventListener("click", () => setMode("after"));
    runSelect.addEventListener("change", () => loadRun(runSelect.value));
    variantSelect.addEventListener("change", () => {
      processedKey = variantSelect.value;
      updateImage();
    });

    boot();
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Start the ECICE local result viewer.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Pipeline output root containing numbered stage folders. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_text(path):
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def path_candidates(value):
    if not value:
        return []

    raw = str(value).replace("\\", "/")
    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((BASE_DIR / path, OUTPUT_DIR / path))
        candidates.extend(
            OUTPUT_DIR / stage_dir / path.name
            for stage_dir in OUTPUT_STAGE_DIRECTORIES
        )
        candidates.extend((DATASET_DIR / path.name, BASE_DIR / path.name))
    return candidates


def first_existing(paths):
    seen = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def resolve_original_image(stem, manifest):
    candidates = path_candidates(manifest.get("input_path"))
    for extension in IMAGE_EXTENSIONS:
        candidates.extend((DATASET_DIR / f"{stem}{extension}", BASE_DIR / f"{stem}{extension}"))
    return first_existing(candidates)


def resolve_processed_variants(stem, manifest):
    variants = []
    manifest_variants = manifest.get("variants", {})
    if isinstance(manifest_variants, dict):
        for key, value in manifest_variants.items():
            path = first_existing(path_candidates(value))
            if path:
                variants.append(
                    {
                        "key": str(key),
                        "label": VARIANT_LABELS.get(str(key), str(key).replace("_", " ")),
                        "path": path,
                    }
                )

    paths = build_stage_paths(Path(f"{stem}.jpg"), OUTPUT_DIR)
    extra_paths = {
        "merged_ocr": paths["ensemble_dir"] / f"{stem}_merged_ocr.jpg",
        "merged_ocr_review": paths["ensemble_dir"] / f"{stem}_merged_ocr_review.jpg",
    }
    known_keys = {item["key"] for item in variants}
    for key, path in extra_paths.items():
        if key not in known_keys and path.is_file():
            variants.append({"key": key, "label": VARIANT_LABELS[key], "path": path.resolve()})

    return variants


def discover_runs():
    runs = []
    summary_dir = OUTPUT_DIR / "05_summary"
    for summary_path in sorted(summary_dir.glob("*_summary.md")):
        stem = summary_path.name[: -len("_summary.md")]
        paths = build_stage_paths(Path(f"{stem}.jpg"), OUTPUT_DIR)
        manifest_path = paths["manifest"]
        summary_json_path = paths["summary_json"]
        manifest = load_json(manifest_path)
        summary_data = load_json(summary_json_path)
        evaluation_data = load_json(paths["cer_json"])
        ablation_data = load_json(paths["ablation_json"])
        original = resolve_original_image(stem, manifest)
        processed_variants = resolve_processed_variants(stem, manifest)
        default_processed_key = next(
            (key for key in PREFERRED_PROCESSED_KEYS if any(item["key"] == key for item in processed_variants)),
            processed_variants[0]["key"] if processed_variants else "",
        )
        runs.append(
            {
                "stem": stem,
                "display_name": stem.replace("_", " "),
                "summary_path": summary_path.resolve(),
                "summary_json_path": summary_json_path.resolve(),
                "summary_data": summary_data,
                "evaluation_data": evaluation_data,
                "ablation_data": ablation_data,
                "original": original,
                "processed_variants": processed_variants,
                "default_processed_key": default_processed_key,
            }
        )
    return runs


def find_run(stem):
    for run in discover_runs():
        if run["stem"] == stem:
            return run
    return None


def media_url(stem, key):
    return f"/media/{stem}/{key}"


def basename(path):
    return path.name if path else ""


def public_run(run, include_summary=False):
    summary_data = run["summary_data"]
    evaluation_data = run.get("evaluation_data", {})
    ablation_data = run.get("ablation_data", {})
    ablation_aggregate = ablation_data.get("aggregate", {}).get("no_whitespace", {})
    ablation_rows = [
        {
            "setting": setting.get("label", setting.get("id", "")),
            "setting_id": setting.get("id", ""),
            "description": setting.get("description", ""),
            **ablation_aggregate.get(setting.get("id", ""), {}),
        }
        for setting in ablation_data.get("settings", [])
    ]
    original = (
        {
            "key": "original",
            "label": "處理前原圖",
            "name": basename(run["original"]),
            "url": media_url(run["stem"], "original"),
        }
        if run["original"]
        else None
    )
    processed_variants = [
        {
            "key": item["key"],
            "label": item["label"],
            "name": basename(item["path"]),
            "url": media_url(run["stem"], item["key"]),
        }
        for item in run["processed_variants"]
    ]
    payload = {
        "stem": run["stem"],
        "display_name": run["display_name"],
        "summary_name": basename(run["summary_path"]),
        "used_llm": bool(summary_data.get("used_llm")),
        "provider": summary_data.get("provider", ""),
        "model": summary_data.get("model", ""),
        "original": original,
        "processed_variants": processed_variants,
        "default_processed_key": run["default_processed_key"],
        "evaluation_summary": evaluation_data.get("aggregate", {}).get("no_whitespace", {}),
        "error_analysis_rows": evaluation_data.get("error_analysis", {}).get("rows", []),
        "ablation_rows": ablation_rows,
    }
    if include_summary:
        summary = read_text(run["summary_path"])
        sections = split_summary_sections(summary)
        default_section_key = next(
            (section["key"] for section in sections if section["title"] == "重點摘要"),
            sections[0]["key"] if sections else "__all__",
        )
        payload["summary_markdown"] = summary
        payload["summary_html"] = render_markdown(summary)
        payload["summary_sections"] = sections
        payload["default_summary_section_key"] = default_section_key
    return payload


def split_summary_sections(markdown):
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections = []
    title = None
    section_lines = []

    def append_section():
        if not title:
            return
        body = "\n".join(section_lines).strip()
        if not body:
            return
        index = len(sections) + 1
        sections.append(
            {
                "key": f"section-{index}",
                "title": title,
                "markdown": body,
                "html": render_markdown(body),
            }
        )

    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line.strip())
        if heading:
            append_section()
            title = heading.group(1)
            section_lines = [line]
            continue
        if title:
            section_lines.append(line)

    append_section()

    if not sections and markdown.strip():
        sections.append(
            {
                "key": "section-1",
                "title": "完整摘要",
                "markdown": markdown.strip(),
                "html": render_markdown(markdown),
            }
        )

    return sections


def render_inline(text):
    text = html.escape(text)
    text = text.replace("\\rightarrow", "→")
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\$([^$]+)\$", r'<span class="math">\1</span>', text)
    return text


def is_table_separator(line):
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def render_table(lines):
    header = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines[2:]
    ]
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{render_inline(cell)}</th>" for cell in header)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        cells = row + [""] * max(0, len(header) - len(row))
        parts.extend(f"<td>{render_inline(cell)}</td>" for cell in cells[: len(header)])
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def flush_paragraph(parts, paragraph):
    if paragraph:
        parts.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
        paragraph.clear()


def flush_list(parts, list_stack):
    while list_stack:
        parts.append(f"</{list_stack.pop()}>")


def render_markdown(markdown):
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    parts = []
    paragraph = []
    list_stack = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_stack)
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_stack)
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            parts.append(render_table(table_lines))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_stack)
            level = len(heading.group(1))
            parts.append(f"<h{level}>{render_inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        bullet = re.match(r"^(\s*)([-*+])\s+(.+)$", line)
        numbered = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph(parts, paragraph)
            indent = len((bullet or numbered).group(1).replace("\t", "    "))
            list_type = "ul" if bullet else "ol"
            target_depth = indent // 4 + 1
            while len(list_stack) > target_depth:
                parts.append(f"</{list_stack.pop()}>")
            while len(list_stack) < target_depth:
                list_stack.append(list_type)
                parts.append(f"<{list_type}>")
            if list_stack and list_stack[-1] != list_type:
                parts.append(f"</{list_stack.pop()}>")
                list_stack.append(list_type)
                parts.append(f"<{list_type}>")
            text = bullet.group(3) if bullet else numbered.group(2)
            parts.append(f"<li>{render_inline(text)}</li>")
            index += 1
            continue

        flush_list(parts, list_stack)
        paragraph.append(stripped)
        index += 1

    flush_paragraph(parts, paragraph)
    flush_list(parts, list_stack)
    return "\n".join(parts)


def resolve_media_path(run, key):
    if key == "original":
        return run["original"]
    for item in run["processed_variants"]:
        if item["key"] == key:
            return item["path"]
    return None


def is_allowed_path(path):
    if not path:
        return False
    try:
        resolved = path.resolve()
        allowed_roots = (BASE_DIR.resolve(), OUTPUT_DIR.resolve())
        return resolved.is_file() and any(resolved.is_relative_to(root) for root in allowed_roots)
    except OSError:
        return False


class AppHandler(BaseHTTPRequestHandler):
    def send_bytes(self, content, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload, status=HTTPStatus.OK):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(content, "application/json; charset=utf-8", status)

    def send_text(self, text, status=HTTPStatus.OK):
        self.send_bytes(text.encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        if route in ("/", "/index.html"):
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if route == "/api/runs":
            self.send_json([public_run(run) for run in discover_runs()])
            return

        if route.startswith("/api/runs/"):
            stem = route.removeprefix("/api/runs/")
            run = find_run(stem)
            if not run:
                self.send_json({"error": "Result not found."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(public_run(run, include_summary=True))
            return

        if route.startswith("/media/"):
            parts = route.removeprefix("/media/").split("/", 1)
            if len(parts) != 2:
                self.send_text("Invalid media URL.", HTTPStatus.BAD_REQUEST)
                return
            stem, key = parts
            run = find_run(stem)
            path = resolve_media_path(run, key) if run else None
            if not is_allowed_path(path):
                self.send_text("Media not found.", HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_bytes(path.read_bytes(), content_type)
            return

        self.send_text("Not found.", HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"ECICE UI running at {url}")
    print(f"Reading output from {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
