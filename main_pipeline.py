#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究主題：Intelligent Lecture Note Digitization and Summarization Using OCR and Large Language Models
主程式：main_pipeline.py
功能：自動化串接影像預處理、OCR辨識、LLM文字校正與自動摘要測驗生成之資料流
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="課堂筆記自動化數位化與摘要 Pipeline Pipeline")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="輸入的原始筆記圖片路徑 (例如: ./dataset/OCR_test_lin/sample.jpg)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output", 
        help="輸出結果的目標資料夾路徑 (預設: ./output)"
    )
    return parser.parse_args()

def run_step(step_name, command):
    """執行單一子模組腳本並處理異常"""
    print(f"\n==================================================")
    print(f"🚀 正在執行 Step: {step_name}")
    print(f"💻 指令: {' '.join(command)}")
    print(f"==================================================")
    
    try:
        # check=True 會在子程序回傳非 0 狀態碼時拋出 CalledProcessError
        # text=True 自動將標準輸出轉為字串
        result = subprocess.run(command, check=True, text=True)
        print(f"✅ {step_name} 執行成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} 執行失敗！錯誤碼: {e.returncode}", file=sys.stderr)
        if e.output:
            print(f"📋 錯誤輸出:\n{e.output}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ 找不到腳本或 Python 直譯器，請檢查模組檔案是否存在。", file=sys.stderr)
        return False

def main():
    args = parse_args()
    
    # 1. 解析輸入路徑與建立輸出路徑
    input_image = Path(args.input)
    if not input_image.exists():
        print(f"❌ 錯誤: 找不到輸入的圖片檔案：{input_image}")
        sys.exit(1)
        
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 取得不含副檔名的主檔名 (例如: OCR_test_lin)
    base_name = input_image.stem 
    
    # 2. 定義各階段資料流的中間產物路徑 (與你朋友的 GitHub 檔案命名對應)
    preprocessed_img = output_base / f"{base_name}_preprocessed_input.jpg"
    raw_ocr_txt      = output_base / f"{base_name}_merged_ocr.txt"
    corrected_txt    = output_base / f"{base_name}_corrected.txt"
    
    # 最終產物
    summary_md       = output_base / f"{base_name}_summary.md"
    quiz_md          = output_base / f"{base_name}_quiz.md"

    print(f"📝 啟動 Pipeline")
    print(f"🖼️  原始圖片: {input_image}")
    print(f"📂 輸出目錄: {output_base}")

    python_bin = sys.executable # 使用當前環境的 Python 直譯器執行子腳本

    # -------------------------------------------------------------------------
    # Step 1: 影像預處理 (Image Preprocessing)
    # -------------------------------------------------------------------------
    # 預期：讀取原始圖片，進行去線、二值化、銳化，並輸出預處理後的圖片
    step1_cmd = [
        python_bin, "image_preprocessing.py", 
        "--input", str(input_image), 
        "--output", str(preprocessed_img)
    ]
    if not run_step("1. 影像預處理", step1_cmd):
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: OCR 文字辨識 (OCR Recognition)
    # -------------------------------------------------------------------------
    # 預期：讀取預處理後的黑白/銳化圖片，執行 OCR，將結果合併輸出為純文字檔
    step2_cmd = [
        python_bin, "ocr_recognition.py", 
        "--input", str(preprocessed_img), 
        "--output", str(raw_ocr_txt)
    ]
    if not run_step("2. OCR 文字辨識", step2_cmd):
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 3: LLM 文字校正 (LLM Error Correction)
    # -------------------------------------------------------------------------
    # 預期：讀取帶有錯字、斷行混亂的原始 OCR 檔案，透過 LLM 清洗並輸出乾淨文本
    step3_cmd = [
        python_bin, "llm_correction.py", 
        "--input", str(raw_ocr_txt), 
        "--output", str(corrected_txt)
    ]
    if not run_step("3. LLM 文字校正", step3_cmd):
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 4: 筆記結構化摘要與測驗生成 (Note Summarization & Quiz Generation)
    # -------------------------------------------------------------------------
    # 預期：讀取校正後的文本，交由 LLM 生成精美 Markdown 摘要與課堂互動小測驗
    step4_cmd = [
        python_bin, "note_summarization.py", 
        "--input", str(corrected_txt), 
        "--output_summary", str(summary_md),
        "--output_quiz", str(quiz_md)
    ]
    if not run_step("4. 筆記摘要與測驗生成", step4_cmd):
        sys.exit(1)

    print(f"\n==================================================")
    print(f"🎉 Pipeline 全部執行成功！")
    print(f"📊 最終產物清單：")
    print(f"   1. 預處理影像: {preprocessed_img}")
    print(f"   2. 原始OCR文字: {raw_ocr_txt}")
    print(f"   3. LLM校正文字: {corrected_txt}")
    print(f"   4. 課堂重點摘要: {summary_md}")
    print(f"   5. 自動生成測驗: {quiz_md}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
