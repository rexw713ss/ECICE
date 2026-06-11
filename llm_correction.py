import argparse
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

from pipeline_paths import ENSEMBLE_OCR_DIR, LLM_CORRECTION_DIR
from traditional_chinese import (
    TRADITIONAL_CHINESE_INSTRUCTION,
    to_traditional_chinese,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OCR_TEXT = ENSEMBLE_OCR_DIR / "OCR_test_lin_merged_ocr.txt"
DEFAULT_OCR_JSON = ENSEMBLE_OCR_DIR / "OCR_test_lin_merged_ocr.json"
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LOCAL_MODEL = "gemma4:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_NUM_PREDICT = 8192


def default_corrected_path(input_path):
    input_path = Path(input_path)
    stem = input_path.stem
    if stem.endswith("_merged_ocr"):
        stem = stem[: -len("_merged_ocr")]
    return LLM_CORRECTION_DIR / f"{stem}_corrected.txt"


def normalize_ocr_noise(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *([,.;:!?，。；：！？、]) *", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    replacements = {
        "行序": "行為",
        "為序": "為",
        "處现": "處理",
        "處理事动": "處理事務",
        "事动": "事務",
        "事静": "事務",
        "財庭": "財產",
        "财庭": "財產",
        "财產": "財產",
        "财屋": "財產",
        "財屋": "財產",
        "標华": "標準",
        "标準": "標準",
        "强制": "強制",
        "强盗": "強盜",
        "詐欺": "詐欺",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return to_traditional_chinese(text.strip())


def finalize_legal_text(text):
    replacements = {
        "公然且乘人不抗拒": "公然且乘人不及抗拒",
        "使受害人不抗拒": "使受害人不及抗拒",
        "盜竊": "竊盜",
        "只要行為人持有而為所有": "只要行為人易持有為所有",
        "靠背信化理論確認與背信行為在意義上的違背": (
            "違背信託理論認為背信行為本質在於違背信託"
        ),
        "本人委任之外部事務": "本人與第三人間的外部事務",
        "亦包含行為人與本人間之內部事宜均包含在內": (
            "亦包含行為人與本人間的內部事務，且不論法律或事實上的事務均包含在內"
        ),
        "且必須限於財產之事務為限": "且必須限於財產事務",
        (
            "濫用權限為背信本質，認為在於權限之濫用，所謂為他人處理事務，"
            "必須限於財產之事務，且必須限於法律事務，且必須限處理本人與他人"
            "之間之外部事務者始足當之。"
        ): (
            "濫用權限理論認為背信行為本質在於權限濫用。所謂為他人處理事務，"
            "必須限於法律事務，且必須限縮為處理本人與第三人間之外部事務者始足當之。"
        ),
        "三要素（擄行人、人質、付贖金）": (
            "三面關係（行為人、人質、付贖金者）"
        ),
        "擄行人、人質、付贖金三要素": "行為人、人質、付贖金者三面關係",
        (
            "4. 未得他人同意而將其置於實力支配下的擄人行為，"
            "惟有爭議該罪成立是否要求具備三面關係"
        ): (
            "4. 擄人勒贖罪：未得他人同意而將其置於實力支配下，"
            "其成立是否要求具備三面關係"
        ),
        "原則以客觀，例外以主觀為通常標準": "原則採客觀標準，例外採主觀標準",
        "詐欺：施用術": "詐欺：施用詐術",
        "  ロ 實務認為": "  (1) 實務認為",
        "  ヮ 有見解認為": "  (2) 有見解認為",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    text = re.sub(r"限於([^，。\n]+)為限", r"限於\1", text)
    return to_traditional_chinese(text.strip())


def split_text(text, max_chars=1000):
    lines = text.splitlines()
    chunks = []
    current = []
    current_length = 0
    for line in lines:
        extra = len(line) + 1
        if current and current_length + extra > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def infer_ocr_json_path(input_path):
    input_path = Path(input_path)
    candidate = input_path.with_suffix(".json")
    return candidate if candidate.is_file() else None


def build_ocr_evidence_chunks(ocr_json_path, max_chars=3000, max_alternatives=3):
    data = json.loads(Path(ocr_json_path).read_text(encoding="utf-8"))
    results = data.get("clean_results") or data.get("merged_results") or []
    blocks = []
    primary_lines = []
    for item in results:
        primary = to_traditional_chinese(str(item.get("text", "")).strip())
        if not primary:
            continue
        alternatives = []
        seen = {primary}
        for alternative in item.get("alternatives", []):
            text = to_traditional_chinese(str(alternative.get("text", "")).strip())
            if not text or text in seen:
                continue
            seen.add(text)
            alternatives.append(
                f"- {text} "
                f"(信心 {float(alternative.get('confidence', 0)):.2f}, "
                f"來源 {alternative.get('source', 'unknown')})"
            )
            if len(alternatives) >= max_alternatives:
                break

        block = (
            f"[區域 {item.get('id', len(blocks) + 1)}]\n"
            f"主要候選: {primary}\n"
            f"主要信心: {float(item.get('confidence', 0)):.2f}\n"
        )
        if alternatives:
            block += "其他候選:\n" + "\n".join(alternatives)
        blocks.append(block)
        primary_lines.append(primary)

    chunks = []
    current_blocks = []
    current_lines = []
    current_length = 0
    for block, primary in zip(blocks, primary_lines):
        extra = len(block) + 2
        if current_blocks and current_length + extra > max_chars:
            chunks.append(
                {
                    "prompt_text": "\n\n".join(current_blocks),
                    "fallback_text": "\n".join(current_lines),
                }
            )
            current_blocks = []
            current_lines = []
            current_length = 0
        current_blocks.append(block)
        current_lines.append(primary)
        current_length += extra
    if current_blocks:
        chunks.append(
            {
                "prompt_text": "\n\n".join(current_blocks),
                "fallback_text": "\n".join(current_lines),
            }
        )
    return chunks


def build_correction_messages(chunk, chunk_index, chunk_count):
    system_prompt = (
        "你是熟悉臺灣刑法與財產犯罪的繁體中文課堂筆記 OCR 校對助手。"
        "這是一份關於【刑法與財產犯罪】（如搶奪、侵占、背信、強盜、詐欺）的專業法律筆記。" 
        "任務是修正 OCR 錯字、漏字、簡繁混雜、標點與斷行問題。"
        "若輸入提供同一區域的多個 OCR 候選，請綜合候選、信心分數與法律語境，選擇最合理文字。"
        "遇到明顯缺字時，可依上下文與常見法律術語補回，例如：行為人、實務認為、學說認為、"
        "乘人不及抗拒、合法持有、為他人處理事務、濫用權限、保證人地位、財產處分、脅迫。"
        "請保留原本的意思、編號（如 1., (1), (2)）、條列層次與專有名詞；"
        "可潤飾成完整通順句子，但不得增加原筆記未討論的新爭點、法條或結論。"
        "若無法從候選與語境合理重建，請標示「【待核對：原候選】」。"
        "每個區域最多輸出一行，只輸出校正後文字。"
    )
    user_prompt = (
        f"以下是第 {chunk_index}/{chunk_count} 段 OCR 文字，"
        "請校正為通順的繁體中文筆記文字：\n\n"
        f"{chunk}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_reconstruction_messages(text):
    system_prompt = (
        "你是熟悉臺灣刑法的繁體中文法律筆記編輯。"
        "請對 OCR 初步校正文進行第二階段全文重建："
        "修復殘留錯字、漏字、斷句、編號與不完整句子，使內容可直接作為考前複習資料。"
        "你可以依全文上下文與標準法律術語補回高度可推定的缺字，但不得新增筆記未討論的新知識。"
        "特別檢查搶奪、侵占、背信、擄人勒贖、強盜、詐欺、收費設備詐欺等術語。"
        "應保留實務與學說的對立見解、法條號碼及因果流程。"
        "刪除 Date、NO.、無意義孤立字與明顯 OCR 雜訊。"
        "無法可靠重建的句子請保留並標成「【待核對：...】」。"
        "不得摘要、不得省略任何可辨識的筆記重點，只輸出重建後完整筆記。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"請重建以下整份筆記：\n\n{text}"},
    ]


def strip_model_artifacts(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<\|channel\>thought.*?<channel\|>",
        "",
        text,
        flags=re.DOTALL,
    )
    return text.strip()


def request_json(endpoint, payload, headers=None, timeout=90):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to local LLM endpoint: {endpoint}") from exc


def ollama_chat_completion(messages, *, model=None, base_url=None, timeout=90):
    model = model or os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LOCAL_MODEL
    base_url = base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", DEFAULT_OLLAMA_NUM_PREDICT))
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": num_predict,
        },
    }
    data = request_json(endpoint, payload, timeout=timeout)
    if data.get("done_reason") == "length":
        payload["options"]["num_predict"] = min(num_predict * 2, 16384)
        data = request_json(endpoint, payload, timeout=timeout)
    return strip_model_artifacts(data.get("message", {}).get("content", ""))


def is_local_url(base_url):
    return any(
        marker in base_url
        for marker in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    )


def openai_compatible_chat_completion(
    messages,
    *,
    api_key=None,
    model=None,
    base_url=None,
    timeout=90,
):
    api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("LLM_MODEL") or DEFAULT_LOCAL_MODEL
    base_url = (
        base_url
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    if not api_key and is_local_url(base_url):
        api_key = "local"
    if not api_key:
        raise ValueError("Missing API key. Set LLM_API_KEY or OPENAI_API_KEY.")

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    data = request_json(
        endpoint,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    return strip_model_artifacts(data["choices"][0]["message"]["content"])


def chat_completion(
    messages,
    *,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=90,
):
    messages = [dict(message) for message in messages]
    instruction_added = False
    for message in messages:
        if message.get("role") == "system":
            message["content"] = (
                str(message.get("content", "")) + TRADITIONAL_CHINESE_INSTRUCTION
            )
            instruction_added = True
            break
    if not instruction_added:
        messages.insert(
            0,
            {"role": "system", "content": TRADITIONAL_CHINESE_INSTRUCTION.strip()},
        )

    provider = (
        provider
        or os.getenv("LLM_PROVIDER")
        or DEFAULT_LLM_PROVIDER
    ).lower()
    if provider in ("ollama", "local", "gemma", "gemma4"):
        output = ollama_chat_completion(
            messages, model=model, base_url=base_url, timeout=timeout
        )
    elif provider in ("openai", "openai-compatible", "lmstudio", "vllm"):
        output = openai_compatible_chat_completion(
            messages,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return to_traditional_chinese(output)


def correct_text(
    text,
    *,
    use_llm=False,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=600,
    max_chars=1000,
    evidence_chunks=None,
    reconstruct=True,
):
    normalized = normalize_ocr_noise(text)
    result = {
        "used_llm": False,
        "provider": None,
        "model": None,
        "input_length": len(text),
        "normalized_length": len(normalized),
        "corrected_text": normalized,
        "output_script": "Traditional Chinese (Taiwan)",
        "chunks": [],
    }
    if not use_llm:
        return result

    result["provider"] = provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER
    result["model"] = model or os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LOCAL_MODEL
    if evidence_chunks:
        chunks = evidence_chunks
        result["evidence_used"] = True
    else:
        chunks = [
            {"prompt_text": chunk, "fallback_text": chunk}
            for chunk in split_text(normalized, max_chars=max_chars)
        ]
        result["evidence_used"] = False
    corrected_chunks = []
    for index, chunk_data in enumerate(chunks, start=1):
        prompt_text = chunk_data["prompt_text"]
        fallback_text = chunk_data["fallback_text"]
        messages = build_correction_messages(prompt_text, index, len(chunks))
        corrected = chat_completion(
            messages,
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        ).strip()
        output_ratio = len(corrected) / max(1, len(fallback_text))
        if not corrected or output_ratio < 0.45 or output_ratio > 1.60:
            print(
                f"Warning: suspicious LLM output for chunk {index}/{len(chunks)} "
                f"(length ratio {output_ratio:.2f}). Using OCR text instead."
            )
            corrected = fallback_text
        corrected = to_traditional_chinese(corrected)
        corrected_chunks.append(corrected)
        result["chunks"].append(
            {
                "index": index,
                "input_length": len(fallback_text),
                "evidence_length": len(prompt_text),
                "output_length": len(corrected),
                "output_ratio": round(len(corrected) / max(1, len(fallback_text)), 4),
            }
        )

    result["used_llm"] = True
    first_pass_text = "\n".join(corrected_chunks).strip()
    result["first_pass_text"] = first_pass_text
    result["reconstruction_used"] = False

    if reconstruct:
        reconstructed = chat_completion(
            build_reconstruction_messages(first_pass_text),
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        ).strip()
        ratio = len(reconstructed) / max(1, len(first_pass_text))
        if reconstructed and 0.65 <= ratio <= 1.65:
            result["corrected_text"] = finalize_legal_text(reconstructed)
            result["reconstruction_used"] = True
            result["reconstruction_output_ratio"] = round(ratio, 4)
        else:
            print(
                "Warning: suspicious reconstruction output "
                f"(length ratio {ratio:.2f}). Using first-pass correction."
            )
            result["corrected_text"] = finalize_legal_text(first_pass_text)
            result["reconstruction_output_ratio"] = round(ratio, 4)
    else:
        result["corrected_text"] = finalize_legal_text(first_pass_text)
    return result


def correct_file(
    input_path=DEFAULT_OCR_TEXT,
    output_path=None,
    *,
    use_llm=False,
    provider=None,
    api_key=None,
    model=None,
    base_url=None,
    timeout=600,
    max_chars=1000,
    ocr_json_path=None,
    use_evidence=True,
    reconstruct=True,
):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_corrected_path(input_path)
    text = input_path.read_text(encoding="utf-8", errors="replace")
    inferred_json_path = (
        Path(ocr_json_path)
        if ocr_json_path
        else infer_ocr_json_path(input_path)
    )
    evidence_chunks = None
    if use_llm and use_evidence and inferred_json_path and inferred_json_path.is_file():
        evidence_chunks = build_ocr_evidence_chunks(
            inferred_json_path,
            max_chars=max(max_chars * 3, 2400),
        )
    result = correct_text(
        text,
        use_llm=use_llm,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_chars=max_chars,
        evidence_chunks=evidence_chunks,
        reconstruct=reconstruct,
    )
    result["evidence_path"] = str(inferred_json_path) if evidence_chunks else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result["corrected_text"] + "\n", encoding="utf-8")

    result_path = output_path.with_suffix(".json")
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                **result,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return {"text": str(output_path), "json": str(result_path), "used_llm": result["used_llm"]}


def parse_args():
    parser = argparse.ArgumentParser(description="Module 3: OCR text correction with an LLM.")
    parser.add_argument("--input", default=str(DEFAULT_OCR_TEXT), help="OCR text file.")
    parser.add_argument("--output", default=None, help="Corrected text output path.")
    parser.add_argument("--use-llm", action="store_true", help="Use a local or API-based LLM.")
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider: ollama, openai-compatible, lmstudio, or vllm. Default: ollama.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"LLM model name. Default for Ollama: {DEFAULT_LOCAL_MODEL}.",
    )
    parser.add_argument("--api-key", default=None, help="API key, or set LLM_API_KEY/OPENAI_API_KEY.")
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Local/API base URL. Default for Ollama: {DEFAULT_OLLAMA_BASE_URL}.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds.")
    parser.add_argument("--max-chars", type=int, default=1000, help="Max characters per LLM chunk.")
    parser.add_argument(
        "--ocr-json",
        default=None,
        help="OCR JSON containing alternative candidates. Inferred from input by default.",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Do not provide OCR alternatives to the LLM.",
    )
    parser.add_argument(
        "--no-reconstruct",
        action="store_true",
        help="Disable the second full-document reconstruction pass.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        paths = correct_file(
            args.input,
            args.output,
            use_llm=args.use_llm,
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            max_chars=args.max_chars,
            ocr_json_path=args.ocr_json,
            use_evidence=not args.no_evidence,
            reconstruct=not args.no_reconstruct,
        )
    except Exception as exc:
        if not args.use_llm:
            raise
        print(f"LLM correction failed: {exc}. Falling back to rule-based cleanup.")
        paths = correct_file(args.input, args.output, use_llm=False)

    print("Text correction complete.")
    print(f"used_llm: {paths['used_llm']}")
    print(f"text: {paths['text']}")
    print(f"json: {paths['json']}")


if __name__ == "__main__":
    main()
