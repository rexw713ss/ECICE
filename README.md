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
  |     +-> Raw PaddleOCR baseline ---------+
  |     +-> Multi-variant ensemble OCR -----+
  |                                         |
  +-> 03 LLM correction --------------------+-> 04 CER evaluation
  |                                         |      ^
  |                                         +------| Ground truth
  |
  +-> 05 Summary
  |
  +-> 06 Quiz
```

CER is an evaluation branch. It compares ground truth against the raw PaddleOCR,
ensemble-only, and ensemble + LLM outputs. CER does not provide content to the quiz.
The quiz is generated from the summary of the LLM-corrected text.

## Organized Output

```text
output/
  01_preprocessing/       Image variants and manifests
  02_ocr/
    baseline/             Raw original-image PaddleOCR
    ensemble/             Multi-variant ensemble OCR
  03_llm_correction/      Corrected Traditional Chinese text
  04_evaluation/          CER error-rate and accuracy reports
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

See [CER_EVALUATION.md](CER_EVALUATION.md) for metric definitions and batch CER
evaluation instructions.
