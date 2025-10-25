


# expects a pandas data frames
#  recommendation history: movie_id, and u_id recommended movies
# watch_history: u_id, movie_id, watch_time, and movie duration
#  rating history: u_id, movie_id , rating 

from homealone.evaluation.online_evaluation.snapshot_data import *
import pandas as pd
def compute_online_measure():
   
    recommendations_df, watch_rating_df = get_snapshot()
    df = preprocess_snapshot(recommendations_df=recommendations_df, watch_rating_df=watch_rating_df)
    df['recommended_successful'] = df.apply(
        lambda row: (row['rating'] >= 4) if pd.notnull(row['rating']) 
                    else (row['completion_rate'] >= 0.7 if pd.notnull(row['completion_rate']) else False),
        axis=1
    )
    success_proportion = df['recommended_successful'].mean()
    # print(f"Proportion of good recommendations: {success_proportion:.2%}")
    return success_proportion



