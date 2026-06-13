import unittest
from unittest.mock import patch

from quiz_generation import generate_quiz


def numbered_items(prefix, count=5):
    return "\n".join(f"{index}. {prefix}{index}" for index in range(1, count + 1))


class QuizGenerationTests(unittest.TestCase):
    @patch("quiz_generation.chat_completion")
    def test_repairs_missing_short_answer_sections(self, completion):
        objective = (
            "### 單選題\n"
            + numbered_items("單選題")
            + "\n### 是非題\n"
            + numbered_items("是非題")
            + "\n### 單選題答案與解析\n"
            + numbered_items("單選題答案")
            + "\n### 是非題答案與解析\n"
            + numbered_items("是非題答案")
        )
        repaired_short = (
            "### 簡答題\n"
            + numbered_items("簡答題")
            + "\n### 簡答題答案與解析\n"
            + numbered_items("簡答題答案")
        )
        completion.side_effect = [objective, "缺少必要章節", repaired_short]

        result = generate_quiz("## 重點摘要\n內容", use_llm=True, question_count=5)

        self.assertTrue(result["used_llm"])
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["question_counts"]["short_answer"], 5)
        self.assertEqual(completion.call_count, 3)
        repair_messages = completion.call_args_list[2].args[0]
        self.assertIn("上一個輸出缺少必要章節", repair_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
