# ECICE 繁體中文手寫筆記 OCR 實驗

本專案研究如何將含有繁簡混用、個人符號、紙張透視、光照不均、橫線、
紅筆標註與藍色手寫墨水的課堂筆記影像，轉換為可閱讀、可評估、可摘要及
可產生測驗的臺灣繁體中文內容。

實驗比較原始 PaddleOCR、單一影像處理版本、多版本 Ensemble OCR、
Rule-based normalization 與 grounded LLM correction，並使用人工逐字轉錄的
ground truth 計算 Character Error Rate（CER）。

> 本 README 是專案的實驗主文件，記錄研究問題、方法、檔案結構、執行方式、
> 評估定義、目前結果、限制與論文撰寫注意事項。

## 研究問題

本實驗主要回答以下問題：

1. 文件矯正、藍筆萃取與去橫線等影像處理是否能降低手寫中文 OCR 的 CER？
2. 多版本 OCR 候選合併是否優於單次原圖 PaddleOCR？
3. Rule-based normalization 能改善多少錯誤？
4. 在 Rule-based normalization 之後，grounded LLM correction 還能額外改善多少？
5. LLM 是否會新增原圖與 OCR 證據中不存在的資訊？
6. 校正後內容是否能可靠地支援摘要與 Quiz 生成？

## 實驗貢獻

目前系統包含：

- 多版本影像 preprocessing 與可追蹤 manifest。
- Raw PaddleOCR baseline、單一 variant ablation 與 multi-variant ensemble。
- 強制臺灣繁體中文的 deterministic rule-based normalization。
- 使用 OCR 候選、信心分數與上下文的 grounded LLM correction。
- Raw、Ensemble、Rule-based 與 LLM 階段的 CER／正確率比較。
- 繁簡混用、相似字誤認、缺字與 LLM hallucination 錯誤分析。
- 摘要與 Quiz 的格式驗證及本機 UI。

## 目前實驗快照

以下資訊對應目前 `output/` 中的正式結果：

| 項目 | 目前狀態 |
|---|---|
| 結果更新日期 | 2026-06-13 |
| Dataset 影像數 | 26 |
| Ground-truth 檔案數 | 2 |
| 已具完整預測與正式評估的頁面 | `OCR_test_lin` |
| 目前正式評估頁數 | 1，結果不可直接泛化至完整資料集 |
| `OCR_test_lin` ground-truth 非空白字元 | 1,027 |
| 主要報告指標 | No-whitespace CER |
| OCR 執行裝置 | CPU |
| 目前 Ensemble 實際設定 | `--fast-ocr` |
| Correction artifact model | `gemma4:31b` |
| Summary artifact model | `gemma4:31b` |
| Quiz artifact model | `gemma4:31b` |
| Ollama 版本 | 0.30.5 |
| Python 版本 | 3.12.10 |

目前 correction、summary 與 Quiz artifacts 已統一使用 `gemma4:31b`。

## 完整流程

```text
Input note image
  |
  +-> 01 Preprocessing
  |     +-> document rectification
  |     +-> illumination normalization
  |     +-> sharpening
  |     +-> line removal
  |     +-> red annotation suppression
  |     +-> blue ink variants
  |     +-> binary variant
  |
  +-> 02 OCR
  |     +-> raw-image PaddleOCR baseline ------------+
  |     +-> single-variant ablation OCR --------------+
  |     +-> multi-variant ensemble OCR ---------------+
  |                                                   |
  +-> 03 Correction                                   |
  |     +-> rule-based normalization -----------------+-> 04 Evaluation
  |     +-> grounded LLM correction ------------------+      ^
  |                                                          |
  |                                                   Human ground truth
  |
  +-> 05 Summary
  |
  +-> 06 Quiz
```

CER、error analysis 與 ablation 是獨立的 evaluation branch，不會將 ground truth
提供給 OCR、LLM correction、摘要或 Quiz。摘要與 Quiz 也不參與 CER 計算。

## Stage 0：資料與 Ground Truth

輸入影像放在 `dataset/`。人工逐字轉錄放在 `ground_truth/`，檔名 stem 必須與
影像相同：

```text
dataset/OCR_test_lin.jpg
ground_truth/OCR_test_lin.txt
```

Ground-truth 規則：

- 只能人工轉錄原圖可見內容，不可使用 OCR 或 LLM 產生。
- 所有中文使用臺灣繁體中文。
- 未知個人符號、公式、箭頭與特殊字元應原樣保留。
- 全部頁面應採用一致的閱讀順序、標點、英文字母大小寫與難辨字政策。
- 每次修改 ground truth 後，必須重新執行 CER 與 ablation evaluation。

目前以 OpenCC `s2twp` 自動稽核 `OCR_test_lin.txt` 時仍會產生 14 個 edit
operations，其中包含 `占/佔`、`内/內` 與臺灣詞彙轉換。這不代表全部都是
簡體字錯誤，但在論文凍結 ground truth 前應再次人工核對。

詳細標註規則見 [ground_truth/README.md](ground_truth/README.md)。

## Stage 1：影像 Preprocessing

實作檔案：[image_preprocessing.py](image_preprocessing.py)

每張影像會產生以下 variants，並將所有路徑寫入
`<stem>_preprocess_manifest.json`：

| Variant | 處理內容 | 目前正式 fast Ensemble 使用 | Ablation 主表 |
|---|---|---:|---:|
| `document` | 偵測紙張四角、透視矯正、方向校正 | 是 | 是 |
| `enhanced` | `document` + LAB 光照正規化 + CLAHE | 否 | 否 |
| `sharpened` | `enhanced` + unsharp sharpening | 否 | 否 |
| `line_removed` | `sharpened` + 橫線偵測與 inpainting | 否 | 是 |
| `clean_document` | `sharpened` + 紅筆抑制 + 去橫線 | 否 | 否 |
| `blue_ink` | `sharpened` + 藍色墨水遮罩，保留原色 | 是 | 是 |
| `blue_gray` | 藍色墨水遮罩內的灰階版本 | 是 | 否 |
| `blue_binary` | 藍色墨水黑白二值版本 | 否 | 否 |
| `blue_mask` | 藍色墨水遮罩，僅供檢查 | 否 | 否 |
| `binary` | `sharpened` 的 adaptive threshold 版本 | 否 | 否 |

`blue_ink` 與 `line_removed` ablation 是模組層級比較，兩者都包含上游的文件
矯正、光照正規化與銳化，不應寫成只執行單一影像運算。

紅筆抑制與 binary variants 雖然會產生，但目前不參與正式 fast Ensemble，
因此不能宣稱它們造成目前 CER 改善。

## Stage 2：OCR

### Raw PaddleOCR Baseline

實作檔案：[paddleocr_baseline.py](paddleocr_baseline.py)

- 對原始影像執行一次 PaddleOCR。
- 不使用 preprocessing、tiling、ensemble merge 或 LLM。
- 目前設定：`lang=ch`、`device=cpu`、`enable_mkldnn=False`。
- 作為整體系統比較的 raw baseline。

### Single-variant Ablation OCR

實作檔案：[ablation_study.py](ablation_study.py)

`document`、`blue_ink` 與 `line_removed` variants 使用和 raw baseline 相同的
單次 PaddleOCR 設定。如此可避免將 OCR 參數變動誤認為 preprocessing 貢獻。

這三個單一 variant 彼此獨立，並非累加流程。

### Multi-variant Ensemble OCR

實作檔案：[ocr_recognition.py](ocr_recognition.py)

程式完整預設設定：

- Full-page sources：`document`、`line_removed`、`blue_ink`、`blue_gray`
- Vertical-tile source：`blue_gray`
- Grid source：`blue_gray`
- OCR language：`chinese_cht`
- Minimum candidate score：0.25
- Detection threshold：0.25
- Detection box threshold：0.35
- Detection unclip ratio：1.45

目前 `OCR_test_lin` 正式結果使用 `--fast-ocr`，實際 metadata 為：

- Full-page sources：`document`、`blue_ink`、`blue_gray`
- 不執行 vertical tiles
- `blue_gray` grid：`900 x 520`，x/y overlap 分別為 `180/180`
- `det_limit_side_len=1536`
- 關閉 textline orientation
- 125 個 raw candidates、40 個 merged regions、39 個 clean regions

候選合併流程：

1. 將每個 OCR 結果轉為臺灣繁體中文。
2. 依 bounding-box 重疊與垂直中心距離將同區域候選分組。
3. 依 confidence、文字長度、source bonus 與候選共識選出主要候選。
4. 保留同區域 alternatives 與信心分數，供 grounded LLM correction 使用。
5. 過濾低信心單字與未被其他來源支持的低品質結果。

## Stage 3：文字校正

實作檔案：

- [llm_correction.py](llm_correction.py)
- [traditional_chinese.py](traditional_chinese.py)

### Rule-based Normalization

在 LLM 前固定執行：

1. Unicode NFC normalization。
2. CRLF／CR 統一為 LF。
3. BOM、zero-width space、標點周圍空白與多餘空白修正。
4. 常見 OCR 噪聲字詞替換。
5. OpenCC `s2twp` 臺灣繁體中文轉換。

Rule-only 結果寫入：

```text
output/03_llm_correction/<stem>_rule_based.txt
output/03_llm_correction/<stem>_rule_based.json
```

### Grounded LLM Correction

LLM 僅可根據以下證據校正：

- Ensemble 主要 OCR 文字。
- 同區域 OCR alternatives。
- 候選信心分數與來源。
- 原文上下文。

限制政策：

- 不得新增原圖與 OCR 證據不存在的事實、例子、法條、句子或段落。
- 不確定時必須保留原文，不得臆測。
- 保留編號、條列、特殊符號、公式與可能屬於原圖的孤立片段。
- 第一階段進行區域校正，第二階段進行保守全文重建。
- Chunk 輸出長度比例不在 `0.45–1.60` 時回退 OCR 原文。
- 全文重建長度比例不在 `0.65–1.65` 時回退第一階段結果。

Final corrected 結果寫入：

```text
output/03_llm_correction/<stem>_corrected.txt
output/03_llm_correction/<stem>_corrected.json
```

JSON 會記錄 rule-based 與 LLM 是否執行、模型、provider、長度與 chunk metadata。

## Stage 4：Evaluation

### CER 與 Character Accuracy

實作檔案：[cer_evaluation.py](cer_evaluation.py)

比較三個主要方法：

1. `paddleocr_baseline`
2. `ensemble_only`
3. `ensemble_llm`

定義：

```text
CER = (Substitutions + Deletions + Insertions) / Ground-truth Characters
Character Accuracy = max(0, 1 - CER)
```

報告同時提供：

- `strict`：保留內部空白與換行。
- `no_whitespace`：移除全部 Unicode whitespace，作為中文 OCR 主要比較指標。
- Micro CER：依全部 reference characters 聚合。
- Macro CER：先算每頁 CER，再對頁面平均。

### 錯誤分析

使用 ground truth、ensemble raw OCR 與 LLM corrected text 對齊：

| Error Type | 判定方式 |
|---|---|
| 繁簡混用 | Raw OCR 經繁體轉換後與 ground truth 相符 |
| 相似字誤認 | Raw OCR 將 ground-truth 字元或片段替換成其他內容 |
| 缺字 | Raw OCR 缺少 ground truth 中存在的內容 |
| LLM hallucination | Corrected text 額外插入 raw OCR 未支持且與 ground truth 不符的內容 |

Hallucination detection 是 alignment-based 自動候選偵測，正式論文應人工複核候選。

### Ablation Study

實作檔案：[ablation_study.py](ablation_study.py)

`Δ CER vs baseline` 比較 raw PaddleOCR；`Δ CER vs parent` 比較該模組的直接 parent。
負值代表 CER 改善。

| Setting | Parent | 用途 |
|---|---|---|
| Raw image + PaddleOCR | 無 | baseline |
| + document rectification | Raw | 文件矯正貢獻 |
| + blue ink extraction | Raw | 藍筆萃取模組貢獻 |
| + line removal | Raw | 去橫線模組貢獻 |
| + multi-variant ensemble | Raw | Ensemble 整體貢獻 |
| + rule-based normalization | Ensemble | deterministic normalization 貢獻 |
| + LLM correction | Rule-based | grounded LLM 額外貢獻 |

詳細定義見：

- [CER_EVALUATION.md](CER_EVALUATION.md)
- [ABLATION_STUDY.md](ABLATION_STUDY.md)

## Stage 5：摘要

實作檔案：[note_summarization.py](note_summarization.py)

摘要只使用 corrected text 與 OCR 品質 metadata，不使用 ground truth。輸出固定包含：

1. 本頁主題
2. 重點摘要
3. 核心概念與定義
4. 見解比較
5. 判斷架構
6. 易混淆重點
7. 待核對原文

摘要不得加入外部法律知識，並會驗證缺少章節、疑似 OCR 亂碼、Quiz 內容與
Markdown code fence。

## Stage 6：Quiz

實作檔案：[quiz_generation.py](quiz_generation.py)

Quiz 只根據摘要生成，不使用 ground truth，也不從「待核對原文」命題。
預設每一類產生 5 題：

- 單選題
- 是非題
- 簡答題

輸出包含答案與解析，並驗證章節完整性、每類題數及疑似 OCR 亂碼。

## 目前正式結果：OCR_test_lin

目前結果只來自一頁，以下敘述應寫成 case study／preliminary result，
不可宣稱為完整資料集的普遍結論。

### 主要 CER 結果

主要採用 No-whitespace CER：

| Method | CER ↓ | Character Accuracy ↑ | Edits |
|---|---:|---:|---:|
| Raw PaddleOCR | 37.29% | 62.71% | 383 |
| Multi-variant Ensemble | 37.29% | 62.71% | 383 |
| Ensemble + Rule-based + LLM | **26.58%** | **73.42%** | **273** |

完整流程相較 Raw PaddleOCR：

- Absolute CER reduction：**10.71 個百分點**
- Relative error reduction：**28.72%**
- Character accuracy 提升：**10.71 個百分點**

Strict CER：

| Method | CER ↓ | Character Accuracy ↑ |
|---|---:|---:|
| Raw PaddleOCR | 42.65% | 57.35% |
| Multi-variant Ensemble | 42.65% | 57.35% |
| Ensemble + Rule-based + LLM | **33.13%** | **66.87%** |

### Ablation 結果

以下為 No-whitespace CER：

| Setting | CER ↓ | Accuracy ↑ | Δ CER vs baseline | Δ CER vs parent |
|---|---:|---:|---:|---:|
| Raw image + PaddleOCR | 37.29% | 62.71% | 0.00% | N/A |
| + document rectification | 34.08% | 65.92% | -3.21% | -3.21% |
| + blue ink extraction | 38.66% | 61.34% | +1.36% | +1.36% |
| + line removal | 36.81% | 63.19% | -0.49% | -0.49% |
| + multi-variant ensemble | 37.29% | 62.71% | 0.00% | 0.00% |
| + rule-based normalization | 33.11% | 66.89% | -4.19% | -4.19% |
| + LLM correction | **26.58%** | **73.42%** | **-10.71%** | **-6.52%** |

目前單頁觀察：

- Grounded LLM 是最大增量來源：相較 rule-based 再降低 **6.52 個百分點 CER**。
- Rule-based normalization 相較 ensemble 降低 **4.19 個百分點 CER**。
- 文件矯正是最有效的單一影像模組，降低 **3.21 個百分點 CER**。
- 去橫線小幅改善 **0.49 個百分點 CER**。
- 藍筆萃取在本頁使 CER 增加 **1.36 個百分點**。
- 目前 fast multi-variant ensemble 與 raw baseline 的 CER 相同，尚未顯示改善。

### 錯誤分析結果

| Error Type | 偵測筆數 | 成功修正 | 未成功／候選 |
|---|---:|---:|---:|
| 繁簡混用 | 35 | 35 | 0 |
| 相似字誤認 | 113 | 37 | 76 |
| 缺字 | 54 | 16 | 38 |
| LLM hallucination insertion candidates | 10 | N/A | 10，需人工複核 |

完整報表：

- [output/04_evaluation/OCR_test_lin/cer_report.md](output/04_evaluation/OCR_test_lin/cer_report.md)
- [output/04_evaluation/OCR_test_lin/ablation_report.md](output/04_evaluation/OCR_test_lin/ablation_report.md)
- [output/04_evaluation/OCR_test_lin/error_analysis.csv](output/04_evaluation/OCR_test_lin/error_analysis.csv)

## Repository 檔案說明

| 檔案／資料夾 | 用途 |
|---|---|
| `main_pipeline.py` | 依序執行 preprocessing、OCR、correction、evaluation、summary、Quiz |
| `pipeline_paths.py` | 集中管理所有 numbered output paths |
| `image_preprocessing.py` | 產生影像 variants 與 manifest |
| `paddleocr_baseline.py` | 原圖單次 PaddleOCR baseline |
| `ocr_recognition.py` | Multi-variant／tile／grid OCR 與候選合併 |
| `traditional_chinese.py` | 臺灣繁體轉換與 grounded correction 共用政策 |
| `llm_correction.py` | Rule-based normalization 與 grounded LLM correction |
| `cer_evaluation.py` | CER、accuracy 與錯誤分析 |
| `ablation_study.py` | 單一 variant OCR 與 ablation report |
| `note_summarization.py` | 產生並驗證單頁摘要 |
| `quiz_generation.py` | 產生並驗證單頁 Quiz |
| `ui_app.py` | 本機結果檢視 UI |
| `dataset/` | 原始筆記影像 |
| `ground_truth/` | 人工逐字轉錄 |
| `output/` | 所有 pipeline 產物 |
| `tests/` | Unit tests |
| `legacy/` | 舊版流程，僅供參考，不應用於正式實驗 |
| `requirements.txt` | Python 環境套件版本 |
| `CER_EVALUATION.md` | CER 與錯誤分析細節 |
| `ABLATION_STUDY.md` | Ablation 設計細節 |
| `Engineering_proceedings_Template_ecice2026.docx` | 論文格式範本 |

## Output 結構

```text
output/
  01_preprocessing/
    <stem>_preprocess_manifest.json
    <stem>_<variant>_input.jpg

  02_ocr/
    baseline/
      <stem>_paddleocr_baseline.txt
      <stem>_paddleocr_baseline.json
    ablation/
      <stem>_document_rectification.txt/json
      <stem>_blue_ink_extraction.txt/json
      <stem>_line_removal.txt/json
    ensemble/
      <stem>_merged_ocr.txt
      <stem>_merged_ocr.json
      <stem>_merged_ocr.jpg
      <stem>_merged_ocr_review.jpg

  03_llm_correction/
    <stem>_rule_based.txt/json
    <stem>_corrected.txt/json

  04_evaluation/<stem>/
    cer_report.md/json
    cer_per_document.csv
    error_analysis.csv
    ablation_report.md/json/csv

  05_summary/
    <stem>_summary.md/json

  06_quiz/
    <stem>_quiz.md/json
```

## 安裝環境

Windows PowerShell：

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本機 LLM 使用 Ollama：

```powershell
ollama pull gemma4:31b
ollama list
```

`requirements.txt` 目前包含 `paddlepaddle-gpu`，但正式 OCR 程式明確以 CPU 執行。
若更換 PaddleOCR、PaddlePaddle、OpenCV、OpenCC、Ollama 或模型版本，應在論文
實驗設定與 artifact metadata 中記錄。

## 執行方式

### 建議的論文 Final Run

固定模型與目前 fast OCR 設定：

```powershell
.\env\Scripts\python.exe main_pipeline.py `
  --input dataset\OCR_test_lin.jpg `
  --evaluate `
  --fast-ocr `
  --model gemma4:31b
```

完整預設 OCR（包含 vertical tiles）：

```powershell
.\env\Scripts\python.exe main_pipeline.py `
  --input dataset\OCR_test_lin.jpg `
  --evaluate `
  --model gemma4:31b
```

一般數位化、摘要與 Quiz，不執行 CER：

```powershell
.\env\Scripts\python.exe main_pipeline.py `
  --input dataset\OCR_test_lin.jpg `
  --model gemma4:31b
```

預覽完整命令與路徑：

```powershell
.\env\Scripts\python.exe main_pipeline.py `
  --input dataset\OCR_test_lin.jpg `
  --evaluate `
  --fast-ocr `
  --model gemma4:31b `
  --dry-run
```

### Ground Truth 修改後只重算評估

`--resume` 會依輸出檔是否存在跳過步驟，因此修改 ground truth 後應直接執行：

```powershell
.\env\Scripts\python.exe cer_evaluation.py `
  --ground-truth-dir ground_truth `
  --baseline-dir output\02_ocr\baseline `
  --ensemble-dir output\02_ocr\ensemble `
  --llm-dir output\03_llm_correction `
  --output-dir output\04_evaluation\OCR_test_lin `
  --stem OCR_test_lin

.\env\Scripts\python.exe ablation_study.py evaluate `
  --ground-truth-dir ground_truth `
  --baseline-dir output\02_ocr\baseline `
  --ablation-dir output\02_ocr\ablation `
  --ensemble-dir output\02_ocr\ensemble `
  --llm-dir output\03_llm_correction `
  --output-dir output\04_evaluation\OCR_test_lin `
  --stem OCR_test_lin
```

### 批次 CER

只有當每個 ground-truth stem 都已有完整預測檔時，才可移除 `--stem` 執行批次：

```powershell
.\env\Scripts\python.exe cer_evaluation.py
.\env\Scripts\python.exe ablation_study.py evaluate
```

目前 `OCR_test_liao_1.txt` 尚未具有完整 pipeline predictions，因此目前正式報告
僅包含 `OCR_test_lin`。

### UI

```powershell
.\env\Scripts\python.exe ui_app.py
```

開啟：

```text
http://127.0.0.1:8765
```

UI 可查看：

- 原圖與 preprocessing variants。
- 摘要各章節。
- Raw／Ensemble／LLM CER 與正確率。
- Ablation Study。
- 錯誤分析表。

### Tests

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
```

目前共有 21 項 tests，涵蓋 CER、ablation、路徑、繁體轉換、grounded prompt、
rule-based metadata、Quiz 缺失章節修復與 UI payload。

## 論文撰寫建議

### Methods 可使用的章節

1. Dataset and Human Ground Truth
2. Image Preprocessing
3. Raw PaddleOCR Baseline
4. Multi-variant OCR and Candidate Fusion
5. Rule-based Traditional Chinese Normalization
6. Grounded LLM Correction
7. CER, Error Analysis, and Ablation Design
8. Downstream Summary and Quiz Generation

### Results 可安全描述的內容

目前只可描述為單頁 preliminary result：

> On the evaluated `OCR_test_lin` page, the complete correction pipeline reduced
> no-whitespace CER from 37.29% to 26.58%, corresponding to a 28.72% relative
> error reduction.

> Rule-based normalization reduced CER by 4.19 percentage points relative to the
> ensemble output, while grounded LLM correction provided an additional
> 6.52-percentage-point reduction.

> Document rectification improved the single-pass OCR result, whereas blue-ink
> extraction degraded CER on this page.

不可在只有一頁的情況下寫成「所有筆記皆改善」或「ensemble 一定有效」。

## 效度威脅與限制

- 目前正式數值只有一頁，樣本不足以支持普遍結論。
- Raw baseline 使用 PaddleOCR `lang=ch`，Ensemble 使用 `chinese_cht` 並調整 detection
  參數；整體系統比較合理，但不屬於只控制單一變因的 OCR model comparison。
- 單一 preprocessing ablation 使用相同 baseline OCR 設定，可比較 module-level
  contribution，但 variants 不是累加流程。
- 目前正式 Ensemble 使用 `--fast-ocr`，與程式完整預設 OCR 不同。
- Current Ensemble 在本頁未降低 CER，後續需調整 candidate fusion 或擴充資料驗證。
- LLM correction 雖有 grounded prompt 與長度 guard，仍偵測到 10 個 unsupported
  insertion candidates，需人工複核。
- Error analysis 依字串 alignment 自動分類，對複雜句段移位可能產生近似判定。
- OpenCC 會進行臺灣詞彙轉換，不只是字形繁簡轉換；ground truth policy 必須一致。
- 摘要與 Quiz 目前只驗證格式與來源限制，尚未以人工 rubric 評估內容品質。

## Final Experiment Checklist

提交論文前：

1. 凍結並人工複核所有 ground-truth `.txt`。
2. 為更多影像完成 baseline、ablation、ensemble、rule-based 與 LLM outputs。
3. 固定 PaddleOCR、OpenCV、OpenCC、Ollama 與 LLM model 版本。
4. 固定是否使用 `--fast-ocr`，不要混合不同 OCR 設定的結果。
5. 使用同一模型重跑 correction、summary 與 Quiz。
6. 重新產生 batch CER、ablation 與 error-analysis reports。
7. 人工複核 hallucination candidates。
8. 在論文中報告頁數、reference characters、micro/macro CER 與 exact command。
9. 對摘要與 Quiz 建立人工品質評分 rubric。
10. 將 final artifact metadata 與論文表格逐項核對。
