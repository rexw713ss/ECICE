"""Traditional Chinese output normalization shared by the OCR pipeline."""

from functools import lru_cache

from opencc import OpenCC


TRADITIONAL_CHINESE_INSTRUCTION = (
    "\n所有中文輸出一律使用臺灣繁體中文，不可輸出簡體字。"
    "若個人簡寫或特殊符號可由上下文確定代表某個中文字，請還原為對應繁體字；"
    "若無法確定，必須原樣保留，禁止臆測。數學符號、箭頭、編號、英數字與公式原樣保留。"
)

GROUNDED_CORRECTION_INSTRUCTION = (
    "\n只能根據原始 OCR、同區域候選文字與原文上下文做保守校正。"
    "不得補充、推論或新增原圖與 OCR 證據中不存在的事實、例子、法條、句子或段落。"
    "無法確定時必須原樣保留，禁止臆測或新增待核對標記。"
)


@lru_cache(maxsize=1)
def traditional_converter():
    return OpenCC("s2twp")


def to_traditional_chinese(text):
    """Convert Chinese text to Taiwan Traditional while preserving other symbols."""
    return traditional_converter().convert(str(text))
