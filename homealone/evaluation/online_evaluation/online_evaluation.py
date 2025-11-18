


# expects a pandas data frames
#  recommendation history: movie_id, and u_id recommended movies
# watch_history: u_id, movie_id, watch_time, and movie duration
#  rating history: u_id, movie_id , rating

import logging
from online_evaluation.snapshot_data import *
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_online_measure():
    logger.info("Starting online measure computation")

    recommendations_df, watch_rating_df = get_snapshot()
    logger.info(f"Retrieved {len(recommendations_df)} recommendations and {len(watch_rating_df)} watch/rating records")

    df = preprocess_snapshot(recommendations_df=recommendations_df,
                             watch_rating_df=watch_rating_df)
    logger.info(f"Preprocessed data: {len(df)} total records")
    df['recommended_successful'] = df.apply(
    lambda row: (
        pd.notnull(row['rating']) and 
        row['rating'] >= 4 and 
        pd.notnull(row['rated_at']) 
        and 
         row['rated_at'] >= row['timestamp']
    ) if pd.notnull(row['rating']) else (
        pd.notnull(row['completion_rate']) and
        row['completion_rate'] >= 0.7 
        and
        pd.notnull(row['last_watch']) and
        row['last_watch'] >= row['timestamp']
    ),
    axis=1
    )
    success_proportion = df['recommended_successful'].mean()
    successful_count = df['recommended_successful'].sum()

    logger.info(f"Success proportion: {success_proportion:.4f} ({successful_count}/{len(df)} recommendations)")
    logger.info(f"Successful recommendations:\n{df[df['recommended_successful']==True]}")

    return success_proportion


import pandas as pd

def _compute_online_measure_from_df(df: pd.DataFrame) -> float:
    """
    Same logic as compute_online_measure(), but operates directly on a provided DataFrame.
    Used for unit testing to bypass database access.
    """
    df["recommended_successful"] = df.apply(
        lambda row: (
            (pd.notnull(row["rating"]) and row["rating"] >= 4 and pd.notnull(row["rated_at"]) and row["rated_at"] >= row["timestamp"])
            or
            (pd.notnull(row["completion_rate"]) and row["completion_rate"] >= 0.7 and pd.notnull(row["last_watch"]) and row["last_watch"] >= row["timestamp"]))
        ,
        axis=1
    )
    success_proportion = df["recommended_successful"].mean()
    return float(success_proportion if not pd.isna(success_proportion) else 0.0)


if __name__== '__main__':
    compute_online_measure()



