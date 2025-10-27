# src/evaluation/test.py
import unittest, importlib
import pandas as pd
import math

# Import module regardless of package root layout
try:
    mod = importlib.import_module("evaluation.offline_eval")
except ImportError:
    mod = importlib.import_module("src.evaluation.offline_eval")

time_split = getattr(mod, "time_split")
evaluate = getattr(mod, "evaluate")


class TestOfflineEval(unittest.TestCase):
    def test_time_split_basic(self):
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "movie_id": ["A", "B", "C", "D"],
            "rating": [5, 3, 4, 2],
            "rated_at": ["2025-01-05", "2025-01-12", "2025-01-18", "2025-01-25"],
        })
        tr, va, te = time_split(df, "2025-01-10", "2025-01-20")
        self.assertEqual((len(tr), len(va), len(te)), (1, 2, 1))

    def test_metrics_correctness(self):
        # user1 likes A,C; user2 likes B  (threshold=4.0)
        test_df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "movie_id": ["A", "C", "B"],
            "rating": [5.0, 4.0, 5.0],
            "rated_at": ["2025-02-01", "2025-02-01", "2025-02-02"],
        })
        # user1 -> ["A","X","Y"]; user2 -> ["Z","B","Q"]
        def rec(uid: int, k: int):
            return {1: ["A", "X", "Y"], 2: ["Z", "B", "Q"]}.get(uid, [])[:k]

        res = evaluate(rec, test_df, k=3, thr=4.0)

        # Expected averages across users:
        # P=1/3 ≈ 0.3333, R=0.75, HR=1.0, NDCG≈0.621, MAP=0.5, users=2
        self.assertAlmostEqual(res["precision"], 1/3, places=4)
        self.assertAlmostEqual(res["recall"], 0.75, places=4)
        self.assertAlmostEqual(res["hit_rate"], 1.0, places=4)
        self.assertTrue(0.60 <= res["ndcg"] <= 0.63)
        self.assertAlmostEqual(res["map"], 0.5, places=3)
        self.assertEqual(res["users"], 2)

    def test_threshold_effect(self):
        # Stricter threshold should not increase recall
        test_df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "movie_id": ["A", "C", "B"],
            "rating": [5.0, 4.0, 5.0],
            "rated_at": ["2025-02-01", "2025-02-01", "2025-02-02"],
        })
        def rec(uid: int, k: int):
            return {1: ["A", "C"], 2: ["B"]}.get(uid, [])[:k]

        low_thr = evaluate(rec, test_df, k=10, thr=4.0)
        high_thr = evaluate(rec, test_df, k=10, thr=5.0)
        self.assertGreaterEqual(low_thr["recall"], high_thr["recall"])

    def test_empty_predictions(self):
        # With relevant truth but empty preds, metrics should be zeros; users counted
        test_df = pd.DataFrame({
            "user_id": [1],
            "movie_id": ["A"],
            "rating": [5.0],
            "rated_at": ["2025-02-01"],
        })
        def rec(uid: int, k: int): return []
        res = evaluate(rec, test_df, k=5, thr=4.0)
        self.assertEqual(res["users"], 1)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["ndcg"], 0.0)
        self.assertEqual(res["hit_rate"], 0.0)
        self.assertEqual(res["map"], 0.0)


if __name__ == "__main__":
    unittest.main()
