# Traditional Chinese Note OCR Experiment

This project compares raw PaddleOCR with a multi-variant ensemble OCR and LLM
correction workflow for Traditional Chinese handwritten notes.

## Correct Experiment Order

```text
Input image
  |
  +-> 01 Preprocessing
  |
  +-> 02 OCR
  |     +-> Raw PaddleOCR baseline ----------+
  |     +-> Single-variant ablation OCR ------+
  |     +-> Multi-variant ensemble OCR -------+
  |                                          |
  +-> 03 LLM correction ---------------------+-> 04 CER + ablation evaluation
  |                                                  ^
  |                                                  +--- Ground truth
  |
  +-> 05 Summary
  |
  +-> 06 Quiz
```

CER is an evaluation branch. It compares ground truth against the raw PaddleOCR,
ensemble-only, and ensemble + LLM outputs. CER does not provide content to the quiz.
The quiz is generated from the summary of the LLM-corrected text.

LLM correction has two explicit layers:

1. Rule-based normalization: Taiwan Traditional conversion, Unicode NFC,
   punctuation/whitespace cleanup, and common OCR-noise replacements.
2. Grounded LLM correction: contextual typo correction using OCR evidence, with a
   strict policy that forbids adding information absent from the original image.

## Organized Output

```text
output/
  01_preprocessing/       Image variants and manifests
  02_ocr/
    baseline/             Raw original-image PaddleOCR
    ablation/             Single-variant PaddleOCR predictions
    ensemble/             Multi-variant ensemble OCR
  03_llm_correction/      Corrected Traditional Chinese text
  04_evaluation/          CER, accuracy, and error-analysis reports
  05_summary/             Page summaries
  06_quiz/                Generated quizzes
```

Input images belong in `dataset/`. Human-created Traditional Chinese
transcriptions belong in `ground_truth/`.

## Run

Normal digitization and quiz generation:

```powershell
.\env\Scripts\python.exe main_pipeline.py --input dataset\OCR_test_lin.jpg
```

Full experiment including baseline and CER:

```powershell
.\env\Scripts\python.exe main_pipeline.py `
  --input dataset\OCR_test_lin.jpg `
  --evaluate
```

The `--evaluate` command requires `ground_truth/OCR_test_lin.txt`.

Preview every command and output path without running models:

```powershell
.\env\Scripts\python.exe main_pipeline.py `
  --input dataset\OCR_test_lin.jpg `
  --evaluate `
  --dry-run
```

Open the local result viewer:

```powershell
.\env\Scripts\python.exe ui_app.py
```

The UI reads the numbered stage folders under `output/`. For a custom output root,
pass `--output-dir <path>`.

See [CER_EVALUATION.md](CER_EVALUATION.md) for metric definitions and batch CER
evaluation instructions. See [ABLATION_STUDY.md](ABLATION_STUDY.md) for the
component-contribution experiment.
