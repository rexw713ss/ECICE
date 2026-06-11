import unittest
from unittest.mock import patch

from llm_correction import chat_completion, normalize_ocr_noise
from ocr_recognition import extract_candidates
from traditional_chinese import to_traditional_chinese


class TraditionalChineseTests(unittest.TestCase):
    def test_converts_simplified_chinese_to_taiwan_traditional(self):
        self.assertEqual(
            to_traditional_chinese("中华人民共和国与软件"),
            "中華人民共和國與軟體",
        )

    def test_preserves_special_symbols_formulas_and_private_use_characters(self):
        source = "笔记①：A→B，x²＋△與\ue000"
        converted = to_traditional_chinese(source)

        self.assertEqual(converted, "筆記①：A→B，x²＋△與\ue000")

    def test_ocr_cleanup_preserves_symbols_and_converts_chinese(self):
        self.assertEqual(normalize_ocr_noise("笔记①→软件"), "筆記①→軟體")

    def test_ocr_candidates_are_traditional_before_ensemble_merge(self):
        candidates = extract_candidates(
            {
                "rec_texts": ["简体软件①"],
                "rec_scores": [0.9],
                "rec_boxes": [[0, 0, 100, 20]],
            },
            "document_full",
        )

        self.assertEqual(candidates[0]["text"], "簡體軟體①")

    @patch("llm_correction.ollama_chat_completion")
    def test_all_llm_calls_receive_instruction_and_return_traditional(self, completion):
        completion.return_value = "简体答案①"

        output = chat_completion(
            [{"role": "system", "content": "校對"}, {"role": "user", "content": "內容"}]
        )

        self.assertEqual(output, "簡體答案①")
        sent_messages = completion.call_args.args[0]
        self.assertIn("所有中文輸出一律使用臺灣繁體中文", sent_messages[0]["content"])
        self.assertIn("特殊符號", sent_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
