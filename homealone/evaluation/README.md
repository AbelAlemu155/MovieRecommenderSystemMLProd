# Evaluation Module

This folder contains the offline evaluation logic and associated unit tests for the recommendation system.

## Files

- **offline_eval.py** – Core evaluation script used to test any recommender model offline.
- **test.py** – Unit tests verifying correctness of time-based splitting, metric calculations, and evaluator behavior.

## Purpose

The module supports **model-agnostic offline evaluation** — meaning any recommender exposing a `recommend(user_id, k)` method can be tested here.  
It compares predicted recommendations against user ratings from the database.

## Key Functions

- `time_split(df, train_end, val_end)` → Splits ratings by timestamp into train, validation, test sets.  
- `evaluate(fn, test, k, thr)` → Runs Precision@K, Recall@K, NDCG@K, Hit-Rate@K, and MAP@K metrics.

## Running Tests

From your project root:

RUN OFFLINE EVALUATION

python -m evaluation.offline_eval \
  --db src/db/home_alone.db \
  --train_end 2025-07-15 \
  --val_end 2025-10-15 \
  --k 20 \
  --rec src.models.ubcf.model:UBCFRecommender.recommend \
  --threshold 4.0

OUTPUT
{
  "precision": 0.33,
  "recall": 0.75,
  "ndcg": 0.62,
  "hit_rate": 1.0,
  "map": 0.50,
  "users": 2
}


RUN TESTS

```bash
conda activate myenv
export PYTHONPATH=$(pwd)/src
python -m unittest -v evaluation.test

OUTPUT tests

test_empty_predictions ... ok
test_metrics_correctness ... ok
test_threshold_effect ... ok
test_time_split_basic ... ok
----------------------------------------------------------------------
Ran 4 tests in 0.02s
OK
