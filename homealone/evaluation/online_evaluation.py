


# expects a pandas data frames
#  recommendation history: movie_id, and u_id recommended movies
# watch_history: u_id, movie_id, watch_time, and movie duration
#  rating history: u_id, movie_id , rating 
import pandas as pd
def compute_online_measure(recommendation_history, watch_history, rating_history):
   
    recommendation_history['movie_id'] = recommendation_history['movie_ids'].str.split(',')
    recommendation_history = recommendation_history.explode('movie_id').drop(columns='movie_ids').reset_index(drop=True)

 
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



def test_all_high_ratings():
    """All recommendations rated >= 4 should yield success proportion = 1.0"""
    rec = pd.DataFrame({'user_id': [1], 'movie_ids': ['1,2']})
    ratings = pd.DataFrame({'user_id': [1,1], 'movie_id': ['1','2'], 'rating': [5,4]})
    watch = pd.DataFrame({'user_id': [1,1], 'movie_id': ['1','2'], 'watch_time': [100,100], 'movie_duration': [100,100]})
    
    result = compute_online_measure(rec, watch, ratings)
    assert result == 1.0


def test_some_low_ratings():
    """Some movies have ratings below 4"""
    rec = pd.DataFrame({'user_id': [1], 'movie_ids': ['1,2,3']})
    ratings = pd.DataFrame({'user_id': [1,1,1], 'movie_id': ['1','2','3'], 'rating': [5,2,3]})
    watch = pd.DataFrame({'user_id': [1,1,1], 'movie_id': ['1','2','3'], 'watch_time': [90,80,50], 'movie_duration': [100,100,100]})
    
    result = compute_online_measure(rec, watch, ratings)
    assert round(result, 2) == round(1/3, 2)  # only movie_id=1 is >=4


def test_no_ratings_but_high_completion():
    """Success based on completion rate ≥ 0.7"""
    rec = pd.DataFrame({'user_id': [1], 'movie_ids': ['10,11']})
    ratings = pd.DataFrame({'user_id': [], 'movie_id': [], 'rating': []})
    watch = pd.DataFrame({
        'user_id': [1,1],
        'movie_id': ['10','11'],
        'watch_time': [80,50],
        'movie_duration': [100,100]
    })
    result = compute_online_measure(rec, watch, ratings)
    assert result == 0.5  # one movie ≥0.7 completion rate


def test_missing_watch_and_rating_data():
    """Missing data should count as unsuccessful"""
    rec = pd.DataFrame({'user_id': [1], 'movie_ids': ['20,21']})
    ratings = pd.DataFrame({'user_id': [1], 'movie_id': ['20'], 'rating': [5]})
    watch = pd.DataFrame({'user_id': [1], 'movie_id': ['21'], 'watch_time': [30], 'movie_duration': [100]})
    
    result = compute_online_measure(rec, watch, ratings)
    # movie 20: rating >=4 → success
    # movie 21: completion = 0.3 <0.7 → fail
    assert result == 0.5


def test_multiple_users():
    """Test aggregation across multiple users"""
    rec = pd.DataFrame({'user_id': [1,2], 'movie_ids': ['1,2', '3,4']})
    ratings = pd.DataFrame({
        'user_id': [1,1,2,2],
        'movie_id': ['1','2','3','4'],
        'rating': [5,3,4,1]
    })
    watch = pd.DataFrame({
        'user_id': [1,1,2,2],
        'movie_id': ['1','2','3','4'],
        'watch_time': [90,90,90,90],
        'movie_duration': [100,100,100,100]
    })
    
    # Successful: movie 1 (rating=5), movie 3 (rating=4)
    result = compute_online_measure(rec, watch, ratings)
    assert result == 0.5

if __name__== "__main__":
    test_all_high_ratings()
    test_some_low_ratings()
    test_no_ratings_but_high_completion()
    test_missing_watch_and_rating_data()
    test_multiple_users()