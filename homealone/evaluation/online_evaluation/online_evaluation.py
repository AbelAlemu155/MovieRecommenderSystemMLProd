


# expects a pandas data frames
#  recommendation history: movie_id, and u_id recommended movies
# watch_history: u_id, movie_id, watch_time, and movie duration
#  rating history: u_id, movie_id , rating 

from homealone.evaluation.online_evaluation.snapshot_data import *
import pandas as pd

def compute_online_measure(treatment_group=None):
   
    recommendations_df, watch_rating_df = get_snapshot(treatment_group=treatment_group)
    df = preprocess_snapshot(recommendations_df=recommendations_df, 
                             watch_rating_df=watch_rating_df)
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
    print(f"Proportion of good recommendations: {success_proportion}")
    print(df[df['recommended_successful']==True])
    return success_proportion


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



