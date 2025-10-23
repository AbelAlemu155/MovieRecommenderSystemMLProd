


# expects a pandas data frames
#  recommendation history: movie_id, and u_id recommended movies
# watch_history: u_id, movie_id, watch_time, and movie duration
#  rating history: u_id, movie_id , rating 
import pandas as pd
def compute_online_measure(recommendation_history, watch_history, rating_history):
   


    df = recommendation_history.merge(rating_history, on=['user_id', 'movie_id'], how='left')
    df = df.merge(watch_history, on=['user_id', 'movie_id'], how='left')
    df['completion_rate'] = df['watch_time'] / df['movie_duration']
    df['recommended_successful'] = df.apply(
        lambda row: (row['rating'] >= 4) if pd.notnull(row['rating']) 
                    else (row['completion_rate'] >= 0.7 if pd.notnull(row['completion_rate']) else False),
        axis=1
    )
    success_proportion = df['recommended_successful'].mean()
    print(f"Proportion of good recommendations: {success_proportion:.2%}")
    return success_proportion



import pandas as pd
from io import StringIO
  # replace `your_module` with your actual module name


def test_compute_online_measure():
    # --- Sample data ---
    recommendation_history = pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'movie_id': [10, 11, 12, 13]
    })

    rating_history = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [10, 13],
        'rating': [5, 3]  # one good (>=4), one not good (<4)
    })

    watch_history = pd.DataFrame({
        'user_id': [1, 2, 2],
        'movie_id': [11, 12, 13],
        'watch_time': [70, 90, 30],
        'movie_duration': [100, 100, 100]
    })
    # Expected:
    #  movie 10: rating=5 → success ✅
    #  movie 11: completion=0.7 → success ✅
    #  movie 12: completion=0.9 → success ✅
    #  movie 13: rating=3, completion=0.3 → fail ❌
    # So, 3/4 successful → 0.75

    expected_proportion = 0.75

    result = compute_online_measure(recommendation_history, watch_history, rating_history)

    # Assert the result
    assert abs(result - expected_proportion) < 1e-6, f"Expected {expected_proportion}, got {result}"






