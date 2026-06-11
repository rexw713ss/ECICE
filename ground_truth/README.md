# Ground-truth Transcriptions

Place one human-created UTF-8 transcription per evaluated image in this directory.
Name each file `<image_stem>.txt`, for example:

```text
OCR_test_lin.txt
OCR_test_su.txt
```

Transcribe only visible source-image text and write all Chinese as Taiwan Traditional
Chinese. Preserve personal symbols exactly when their meaning is unknown. Keep one
consistent policy for reading order, punctuation, Latin-letter case, and illegible
characters across the full evaluation set. Do not use OCR, summaries, or LLM output
as ground truth.
