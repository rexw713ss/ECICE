import argparse
import json
import os
from pathlib import Path

from llm_correction import (
    BASE_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    chat_completion,
)
from note_summarization import (
    count_numbered_items,
    extract_named_sections,
    has_suspicious_ocr_noise,
    strip_markdown_fences,
)
from traditional_chinese import to_traditional_chinese
from pipeline_paths import QUIZ_DIR, SUMMARY_DIR


DEFAULT_SUMMARY = SUMMARY_DIR / "OCR_test_lin_summary.md"
QUESTION_SECTIONS = ("單選題", "是非題", "簡答題")
ANSWER_SECTIONS = ("單選題答案與解析", "是非題答案與解析", "簡答題答案與解析")
ALL_QUIZ_SECTIONS = (*QUESTION_SECTIONS, *ANSWER_SECTIONS)
SECTION_ALIASES = {
    "是非題": ("判斷題",),
    "是非題答案與解析": ("判斷題答案與解析", "判斷題解析"),
    "單選題答案與解析": ("單選題解析", "選擇題答案與解析"),
    "簡答題答案與解析": ("簡答題解析", "參考答案與解析"),
}


def default_quiz_path(summary_path):
    summary_path = Path(summary_path)
    stem = summary_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    return QUIZ_DIR / f"{stem}_quiz.md"


def prepare_quiz_source(summary):
    sections = extract_named_sections(summary, ("待核對原文",))
    pending = sections.get("待核對原文", "")
    if not pending:
        return summary
    marker = "## 待核對原文"
    marker_index = summary.find(marker)
    return summary[:marker_index].rstrip() if marker_index >= 0 else summary


def build_objective_quiz_messages(summary, question_count):
    return [
        {
            "role": "system",
            "content": (
                "你是繁體中文課堂小考命題教師。"
                "只能根據提供的單頁筆記摘要命題，不得使用外部知識或改變見解歸屬。"
                f"輸出四個三級標題：單選題、是非題、單選題答案與解析、是非題答案與解析。"
                f"單選題恰好 {question_count} 題，每題四個選項 A-D 且只有一個正確答案。"
                f"是非題恰好 {question_count} 題。"
                "答案與解析必須逐題對應，且答案可由摘要直接推出。"
                "不得從待核對原文命題，不要輸出總標題、二級標題、emoji、code fence 或前言。"
            ),
        },
        {"role": "user", "content": f"單頁筆記摘要：\n{summary}"},
    ]


def build_short_answer_messages(summary, question_count):
    return [
        {
            "role": "system",
            "content": (
                "你是繁體中文課堂小考命題教師。"
                "只能根據提供的單頁筆記摘要命題，不得使用外部知識或改變見解歸屬。"
                "輸出兩個三級標題：簡答題、簡答題答案與解析。"
                f"簡答題恰好 {question_count} 題，答案須列出得分關鍵字。"
                "不得從待核對原文命題，不要輸出總標題、二級標題、emoji、code fence 或前言。"
            ),
        },
        {"role": "user", "content": f"單頁筆記摘要：\n{summary}"},
    ]


def build_quiz_repair_messages(summary, missing_sections, question_count):
    section_list = "、".join(missing_sections)
    requirements = []
    if "簡答題" in missing_sections:
        requirements.append(f"簡答題恰好 {question_count} 題")
    if "簡答題答案與解析" in missing_sections:
        requirements.append("簡答題答案與解析必須逐題對應並列出得分關鍵字")
    if "單選題" in missing_sections:
        requirements.append(f"單選題恰好 {question_count} 題，每題四個選項 A-D")
    if "是非題" in missing_sections:
        requirements.append(f"是非題恰好 {question_count} 題")
    return [
        {
            "role": "system",
            "content": (
                "你是繁體中文課堂小考命題教師。"
                "上一個輸出缺少必要章節，請只重新輸出指定的三級標題與內容。"
                "只能根據摘要命題，不得使用外部知識，不得輸出前言、總標題、"
                "二級標題、emoji 或 code fence。"
                f"必須完整輸出：{section_list}。"
                + "；".join(requirements)
                + "。"
            ),
        },
        {"role": "user", "content": f"單頁筆記摘要：\n{summary}"},
    ]


def normalize_quiz_fragment(fragment, section_names):
    aliases = []
    alias_to_name = {}
    for name in section_names:
        for alias in SECTION_ALIASES.get(name, ()):
            aliases.append(alias)
            alias_to_name[alias] = name
    extracted = extract_named_sections(
        strip_markdown_fences(fragment),
        (*section_names, *aliases),
    )
    return {alias_to_name.get(name, name): body for name, body in extracted.items()}


def assemble_quiz(objective_parts, short_parts):
    return (
        "# 單頁筆記小考\n\n"
        "## 題目\n\n"
        f"### 單選題\n{objective_parts['單選題'].strip()}\n\n"
        f"### 是非題\n{objective_parts['是非題'].strip()}\n\n"
        f"### 簡答題\n{short_parts['簡答題'].strip()}\n\n"
        "## 答案與解析\n\n"
        f"### 單選題答案與解析\n{objective_parts['單選題答案與解析'].strip()}\n\n"
        f"### 是非題答案與解析\n{objective_parts['是非題答案與解析'].strip()}\n\n"
        f"### 簡答題答案與解析\n{short_parts['簡答題答案與解析'].strip()}\n"
    )


def validate_quiz_document(quiz, question_count=5):
    parts = extract_named_sections(quiz, ALL_QUIZ_SECTIONS)
    errors = []
    for section in ALL_QUIZ_SECTIONS:
        if not parts.get(section):
            errors.append(f"缺少小考章節：{section}")
    for section in QUESTION_SECTIONS:
        count = count_numbered_items(parts.get(section, ""))
        if count != question_count:
            errors.append(f"{section}應為 {question_count} 題，目前為 {count} 題")
    if has_suspicious_ocr_noise(quiz):
        errors.append("小考含疑似 OCR 亂碼")
    if "```" in quiz:
        errors.append("小考含 Markdown code fence")
    return {
        "validation_errors": errors,
        "question_counts": {
            "multiple_choice": count_numbered_items(parts.get("單選題", "")),
            "true_false": count_numbered_items(parts.get("是非題", "")),
            "short_answer": count_numbered_items(parts.get("簡答題", "")),
        },
    }


def fallback_quiz(summary, question_count=5):
    sections = extract_named_sections(
        summary,
        ("重點摘要", "核心概念與定義", "見解比較", "判斷架構"),
    )
    source_lines = []
    for body in sections.values():
        source_lines.extend(line.strip("-* ") for line in body.splitlines() if line.strip())
    source_lines = [line for line in source_lines if not line.startswith("|")][:question_count]
    questions = "\n".join(
        f"{index}. 請說明「{line[:45]}」的重點。"
        for index, line in enumerate(source_lines, start=1)
    )
    answers = "\n".join(
        f"{index}. 參考摘要中的相關段落作答。"
        for index in range(1, len(source_lines) + 1)
    )
    return (
        "# 單頁筆記小考\n\n"
        "## 題目\n\n"
        "### 簡答題\n"
        f"{questions}\n\n"
        "## 答案與解析\n\n"
        "### 簡答題答案與解析\n"
        f"{answers}\n"
    )


def generate_quiz(
    summary,
    *,
    use_llm=False,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=120,
    question_count=5,
):
    quiz_source = prepare_quiz_source(summary)
    result = {
        "used_llm": False,
        "provider": None,
        "model": None,
        "question_count_per_type": question_count,
        "quiz": fallback_quiz(quiz_source, question_count),
    }
    if not use_llm:
        result.update(validate_quiz_document(result["quiz"], question_count))
        return result

    result["provider"] = provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER
    result["model"] = model or os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LOCAL_MODEL
    print("Generating objective quiz...")
    objective = chat_completion(
        build_objective_quiz_messages(quiz_source, question_count),
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
    print("Generating short-answer quiz...")
    short_answer = chat_completion(
        build_short_answer_messages(quiz_source, question_count),
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
    objective_parts = normalize_quiz_fragment(
        objective,
        ("單選題", "是非題", "單選題答案與解析", "是非題答案與解析"),
    )
    short_parts = normalize_quiz_fragment(
        short_answer,
        ("簡答題", "簡答題答案與解析"),
    )
    missing_objective = [
        name
        for name in ("單選題", "是非題", "單選題答案與解析", "是非題答案與解析")
        if not objective_parts.get(name)
    ]
    if missing_objective:
        print("Repairing missing objective quiz sections...")
        repaired = chat_completion(
            build_quiz_repair_messages(quiz_source, missing_objective, question_count),
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        objective_parts.update(normalize_quiz_fragment(repaired, missing_objective))

    missing_short = [
        name
        for name in ("簡答題", "簡答題答案與解析")
        if not short_parts.get(name)
    ]
    if missing_short:
        print("Repairing missing short-answer quiz sections...")
        repaired = chat_completion(
            build_quiz_repair_messages(quiz_source, missing_short, question_count),
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        short_parts.update(normalize_quiz_fragment(repaired, missing_short))

    missing = [
        name
        for name in ALL_QUIZ_SECTIONS
        if not objective_parts.get(name) and not short_parts.get(name)
    ]
    if missing:
        raise RuntimeError("LLM quiz is missing sections: " + ", ".join(missing))

    quiz = to_traditional_chinese(assemble_quiz(objective_parts, short_parts))
    validation = validate_quiz_document(quiz, question_count)
    if validation["validation_errors"]:
        raise RuntimeError("LLM quiz failed validation: " + "; ".join(validation["validation_errors"]))
    result.update(validation)
    result["used_llm"] = True
    result["quiz"] = quiz
    return result


def generate_quiz_file(
    input_path=DEFAULT_SUMMARY,
    output_path=None,
    *,
    use_llm=False,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=120,
    question_count=5,
):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_quiz_path(input_path)
    summary = input_path.read_text(encoding="utf-8", errors="replace")
    result = generate_quiz(
        summary,
        use_llm=use_llm,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        question_count=question_count,
    )
    result["document_type"] = "quiz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["quiz"] = to_traditional_chinese(result["quiz"])
    result["output_script"] = "Traditional Chinese (Taiwan)"
    output_path.write_text(result["quiz"], encoding="utf-8")
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


def validate_existing_quiz(output_path, question_count=5):
    output_path = Path(output_path)
    quiz = output_path.read_text(encoding="utf-8", errors="replace")
    validation = validate_quiz_document(quiz, question_count)
    result_path = output_path.with_suffix(".json")
    data = {}
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
    data.update(
        {
            "output_path": str(output_path),
            "quiz": quiz,
            **validation,
            "document_type": "quiz",
        }
    )
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": str(output_path), "json": str(result_path), **validation}


def parse_args():
    parser = argparse.ArgumentParser(description="Module 5: generate a quiz from a page summary.")
    parser.add_argument("--input", default=str(DEFAULT_SUMMARY), help="Page-summary Markdown input.")
    parser.add_argument("--output", default=None, help="Quiz Markdown output.")
    parser.add_argument("--use-llm", action="store_true", help="Use a local or API-based LLM.")
    parser.add_argument("--provider", default=None, help="LLM provider. Default: ollama.")
    parser.add_argument("--model", default=None, help=f"LLM model. Default: {DEFAULT_LOCAL_MODEL}.")
    parser.add_argument("--api-key", default=None, help="API key for a remote provider.")
    parser.add_argument("--base-url", default=None, help=f"LLM base URL. Default: {DEFAULT_OLLAMA_BASE_URL}.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    parser.add_argument("--question-count", type=int, default=5, help="Questions per type.")
    parser.add_argument("--validate-existing", action="store_true", help="Validate an existing quiz Markdown.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else default_quiz_path(args.input)
    if args.validate_existing:
        validation = validate_existing_quiz(output_path, args.question_count)
        print("Existing quiz validation complete.")
        print(f"validation_errors: {validation['validation_errors']}")
        print(f"question_counts: {validation['question_counts']}")
        print(f"markdown: {validation['markdown']}")
        print(f"json: {validation['json']}")
        return

    paths = generate_quiz_file(
        args.input,
        output_path,
        use_llm=args.use_llm,
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        question_count=args.question_count,
    )
    print("Quiz generation complete.")
    print(f"used_llm: {paths['used_llm']}")
    print(f"markdown: {paths['markdown']}")
    print(f"json: {paths['json']}")


if __name__ == "__main__":
    main()
