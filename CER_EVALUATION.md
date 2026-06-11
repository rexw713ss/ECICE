# CER Evaluation

This experiment compares:

1. `paddleocr_baseline`: one PaddleOCR pass on the original image.
2. `ensemble_only`: multi-variant ensemble OCR before LLM correction.
3. `ensemble_llm`: multi-variant ensemble OCR after LLM correction.

## Ground Truth

Create one UTF-8 human transcription per image:

```text
ground_truth/
  OCR_test_lin.txt
  OCR_test_su.txt
```

The filename stem must match the image stem. Transcribe all Chinese content as Taiwan
Traditional Chinese. Preserve personal symbols exactly when their meaning is unknown.
Do not use OCR or an LLM to generate ground truth. Use one consistent policy for
punctuation, Latin-letter case, illegible characters, and text reading order across
all pages.

## Run

Generate direct PaddleOCR baseline predictions:

```powershell
.\env\Scripts\python.exe paddleocr_baseline.py --input dataset
```

The baseline remains raw PaddleOCR output for a scientifically valid comparison. The
multi-variant ensemble and LLM-corrected pipeline outputs are normalized to Taiwan
Traditional Chinese.

Run the existing pipeline for every evaluated image so that
`output/02_ocr/ensemble/<stem>_merged_ocr.txt` and
`output/03_llm_correction/<stem>_corrected.txt` exist.

Calculate CER:

```powershell
.\env\Scripts\python.exe cer_evaluation.py
```

Aggregate reports are written to `output/04_evaluation/`:

- `cer_report.md`: aggregate comparison for reporting.
- `cer_report.json`: complete machine-readable results.
- `cer_per_document.csv`: per-page results for analysis and plotting.

Every report shows both:

- Error rate: `CER = (substitutions + deletions + insertions) / reference characters`.
- Character accuracy: `max(0, 1 - CER)`.

The report uses Unicode NFC so personal symbols and compatibility characters are
not silently rewritten. It includes standard CER with internal whitespace retained
and a whitespace-insensitive CER suitable for evaluating Chinese OCR independently
of line-layout differences.
