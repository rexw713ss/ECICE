import argparse
import json
import os
import re
from pathlib import Path

from llm_correction import (
    BASE_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OCR_TEXT,
    DEFAULT_OLLAMA_BASE_URL,
    chat_completion,
    normalize_ocr_noise,
)
from traditional_chinese import to_traditional_chinese
from pipeline_paths import LLM_CORRECTION_DIR, SUMMARY_DIR


DEFAULT_CORRECTED_TEXT = LLM_CORRECTION_DIR / "OCR_test_lin_corrected.txt"
SUMMARY_SECTIONS = (
    "本頁主題",
    "重點摘要",
    "核心概念與定義",
    "見解比較",
    "判斷架構",
    "易混淆重點",
    "待核對原文",
)
SECTION_ALIASES = {
    "本頁主題": ("Quiz 範圍", "主題"),
    "重點摘要": ("一頁必背", "摘要"),
    "見解比較": ("爭議與見解比較表", "爭議與見解比較"),
    "判斷架構": ("判斷流程",),
    "易混淆重點": ("易錯字與易混淆概念", "易混淆概念"),
    "待核對原文": ("待核對 OCR",),
}


def choose_default_input():
    if DEFAULT_CORRECTED_TEXT.is_file():
        return DEFAULT_CORRECTED_TEXT
    return DEFAULT_OCR_TEXT


def default_summary_path(input_path):
    input_path = Path(input_path)
    stem = input_path.stem
    for suffix in ("_corrected", "_merged_ocr"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return SUMMARY_DIR / f"{stem}_summary.md"


def strip_markdown_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def extract_named_sections(text, names):
    text = strip_markdown_fences(text)
    lines = text.splitlines()
    matches = []
    longest_first = sorted(names, key=len, reverse=True)
    for index, line in enumerate(lines):
        label = re.sub(r"^\s*#{1,6}\s*", "", line).strip().strip("*_ ")
        for name in longest_first:
            if name in label:
                matches.append((index, name))
                break

    parts = {}
    for position, (start, name) in enumerate(matches):
        end = matches[position + 1][0] if position + 1 < len(matches) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if body:
            parts[name] = body
    return parts


def count_numbered_items(text):
    return len(
        re.findall(
            r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?(?:Q\s*)?\d+"
            r"\s*(?:[.、)]|\*\*[.、)]?)\s*",
            text,
        )
    )


def has_suspicious_ocr_noise(text):
    suspicious_tokens = ("�", "銝", "嚗", "", "", "", "")
    return any(token in text for token in suspicious_tokens)


def load_quality_context(input_path):
    input_path = Path(input_path)
    correction_json_path = input_path.with_suffix(".json")
    quality = {"ocr_statistics": {}, "uncertain_region_count": 0}
    if not correction_json_path.is_file():
        return quality

    correction_data = json.loads(correction_json_path.read_text(encoding="utf-8"))
    evidence_path = correction_data.get("evidence_path")
    if not evidence_path or not Path(evidence_path).is_file():
        return quality

    ocr_data = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    quality["ocr_statistics"] = ocr_data.get("statistics", {})
    uncertain_count = 0
    for item in ocr_data.get("clean_results", []):
        confidence = float(item.get("confidence", 0))
        alternatives = item.get("alternatives", [])
        if confidence < 0.65 or len(alternatives) > 1:
            uncertain_count += 1
    quality["uncertain_region_count"] = uncertain_count
    return quality


def format_quality_context(quality):
    statistics = quality.get("ocr_statistics", {})
    return (
        "OCR 品質資訊：\n"
        f"- 原始候選數：{statistics.get('raw_candidate_count', '未知')}\n"
        f"- 合併區域數：{statistics.get('merged_region_count', '未知')}\n"
        f"- 高信心區域數：{statistics.get('high_confidence_region_count', '未知')}\n"
        f"- 低信心或候選分歧區域數：{quality.get('uncertain_region_count', '未知')}"
    )


def fallback_summary(text):
    lines = [line.strip() for line in normalize_ocr_noise(text).splitlines() if line.strip()]
    if not lines:
        return "# 單頁筆記摘要\n\n## 本頁主題\n無可摘要內容。\n"
    bullets = "\n".join(f"- {line}" for line in lines)
    return (
        "# 單頁筆記摘要\n\n"
        "## 本頁主題\n"
        "本頁課堂筆記重點整理。\n\n"
        "## 重點摘要\n"
        f"{bullets}\n\n"
        "## 核心概念與定義\n"
        "- 請依重點摘要整理核心名詞。\n\n"
        "## 見解比較\n"
        "- 請比較筆記中的實務、學說及其他見解。\n\n"
        "## 判斷架構\n"
        "- 先確認爭議問題，再區分各見解及其判斷標準。\n\n"
        "## 易混淆重點\n"
        "- 注意不同見解的歸屬與適用範圍。\n\n"
        "## 待核對原文\n"
        "- 法條號碼、專有名詞與低信心 OCR 內容仍建議對照原圖。\n"
    )


def build_summary_messages(text, quality_context=""):
    return [
        {
            "role": "system",
            "content": (
                "你是繁體中文課堂筆記編輯。請將重建後筆記整理成該頁的摘要 Markdown。"
                "只整理來源已有內容，不得加入外部法律知識、題目、答案或考試指示。"
                "先修正殘留的明顯錯字與漏字；無法確定者放入待核對原文，不得當成確定內容。"
                "必須保留實務、學說、有見解、本文見解等歸屬，不得互換。"
                "只輸出以下七個二級標題，依序為：本頁主題、重點摘要、核心概念與定義、"
                "見解比較、判斷架構、易混淆重點、待核對原文。"
                "見解比較優先使用 Markdown 表格。"
                "不要輸出 Quiz、題庫、答案、emoji、Markdown code fence 或前言。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{quality_context}\n\n"
                "重建後筆記：\n"
                f"{text}"
            ),
        },
    ]


def normalize_summary(summary):
    aliases = []
    alias_to_section = {}
    for section in SUMMARY_SECTIONS:
        for alias in SECTION_ALIASES.get(section, ()):
            aliases.append(alias)
            alias_to_section[alias] = section

    extracted = extract_named_sections(summary, (*SUMMARY_SECTIONS, *aliases))
    normalized = {}
    for name, body in extracted.items():
        normalized[alias_to_section.get(name, name)] = body

    parts = ["# 單頁筆記摘要"]
    for section in SUMMARY_SECTIONS:
        body = normalized.get(section, "")
        if body:
            parts.append(f"## {section}\n{body}")
    return "\n\n".join(parts).strip() + "\n"


def validate_summary_document(summary):
    errors = []
    for section in SUMMARY_SECTIONS:
        if not re.search(rf"(?m)^##\s+{re.escape(section)}\s*$", summary):
            errors.append(f"缺少摘要章節：{section}")
    for forbidden in ("## Quiz 題庫", "## 答案與解析", "### 單選題", "### 是非題"):
        if forbidden in summary:
            errors.append(f"摘要不應包含：{forbidden.lstrip('# ')}")
    if has_suspicious_ocr_noise(summary):
        errors.append("摘要含疑似 OCR 亂碼")
    if "```" in summary:
        errors.append("摘要含 Markdown code fence")
    return errors


def summarize_text(
    text,
    *,
    use_llm=False,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=120,
    quality_context="",
):
    normalized = normalize_ocr_noise(text)
    result = {
        "used_llm": False,
        "provider": None,
        "model": None,
        "input_length": len(text),
        "normalized_length": len(normalized),
        "summary": fallback_summary(normalized),
    }
    if not use_llm:
        result["validation_errors"] = validate_summary_document(result["summary"])
        return result

    result["provider"] = provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER
    result["model"] = model or os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LOCAL_MODEL
    print("Generating page summary...")
    generated = chat_completion(
        build_summary_messages(normalized, quality_context),
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
    summary = to_traditional_chinese(normalize_summary(generated))
    validation_errors = validate_summary_document(summary)
    if validation_errors:
        raise RuntimeError("LLM summary failed validation: " + "; ".join(validation_errors))

    result["used_llm"] = True
    result["summary"] = summary
    result["validation_errors"] = validation_errors
    return result


def summarize_file(
    input_path=None,
    output_path=None,
    *,
    use_llm=False,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=120,
):
    input_path = Path(input_path) if input_path else choose_default_input()
    output_path = Path(output_path) if output_path else default_summary_path(input_path)
    text = input_path.read_text(encoding="utf-8", errors="replace")
    quality = load_quality_context(input_path)
    result = summarize_text(
        text,
        use_llm=use_llm,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        quality_context=format_quality_context(quality),
    )
    result["quality_context"] = quality
    result["document_type"] = "page_summary"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["summary"] = to_traditional_chinese(result["summary"])
    result["output_script"] = "Traditional Chinese (Taiwan)"
    output_path.write_text(result["summary"], encoding="utf-8")
    result_path = output_path.with_suffix(".json")
    result_path.write_text(
        json.dumps(
            {"input_path": str(input_path), "output_path": str(output_path), **result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"markdown": str(output_path), "json": str(result_path), "used_llm": result["used_llm"]}


def validate_existing_summary(output_path):
    output_path = Path(output_path)
    summary = output_path.read_text(encoding="utf-8", errors="replace")
    validation_errors = validate_summary_document(summary)
    result_path = output_path.with_suffix(".json")
    data = {}
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
    for obsolete_key in ("missing_sections", "question_counts", "quiz_preparation", "quiz"):
        data.pop(obsolete_key, None)
    data.update(
        {
            "output_path": str(output_path),
            "summary": summary,
            "validation_errors": validation_errors,
            "document_type": "page_summary",
        }
    )
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": str(output_path), "json": str(result_path), "validation_errors": validation_errors}


def parse_args():
    parser = argparse.ArgumentParser(description="Module 4: summarize one page of corrected notes.")
    parser.add_argument("--input", default=None, help="Corrected/OCR text input.")
    parser.add_argument("--output", default=None, help="Page-summary Markdown output.")
    parser.add_argument("--use-llm", action="store_true", help="Use a local or API-based LLM.")
    parser.add_argument("--provider", default=None, help="LLM provider. Default: ollama.")
    parser.add_argument("--model", default=None, help=f"LLM model. Default: {DEFAULT_LOCAL_MODEL}.")
    parser.add_argument("--api-key", default=None, help="API key for a remote provider.")
    parser.add_argument("--base-url", default=None, help=f"LLM base URL. Default: {DEFAULT_OLLAMA_BASE_URL}.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    parser.add_argument("--validate-existing", action="store_true", help="Validate an existing summary Markdown.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_existing:
        output_path = Path(args.output) if args.output else default_summary_path(
            Path(args.input) if args.input else choose_default_input()
        )
        validation = validate_existing_summary(output_path)
        print("Existing page summary validation complete.")
        print(f"validation_errors: {validation['validation_errors']}")
        print(f"markdown: {validation['markdown']}")
        print(f"json: {validation['json']}")
        return

    paths = summarize_file(
        args.input,
        args.output,
        use_llm=args.use_llm,
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    print("Page summarization complete.")
    print(f"used_llm: {paths['used_llm']}")
    print(f"markdown: {paths['markdown']}")
    print(f"json: {paths['json']}")


if __name__ == "__main__":
    main()
