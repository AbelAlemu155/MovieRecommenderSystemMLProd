

import requests

import os
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from homealone.experimentation.map_users_to_group import define_hash, pyhash

def request_recommendations():
    """Request recommendations for existing users from the local recommender service."""

    db_path = os.path.join(os.getcwd(), "homealone/db/home_alone.db")

    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT user_id FROM ratings")
        user_ids = [row[0] for row in cursor.fetchall()]

        for uid in user_ids:
            url = f"http://localhost:8082/recommend/{uid}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"User {uid}: Success -> {response}")
                else:
                    print(f"User {uid}: Failed with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"User {uid}: Error -> {e}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


def add_ratings_to_db():
    db_dir = "/home/abel/group-project-f25-home-alone/homealone/db"
    db_path = os.path.join(db_dir, "home_alone.db")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    recs_df = pd.read_sql_query("SELECT DISTINCT user_id FROM recommendations", conn)
    recs_df['pyhash'] = recs_df['user_id'].apply(pyhash)

   
    selected_users = recs_df[recs_df['pyhash'] % 100 < 10]['user_id'].tolist()[:50]
    print(selected_users[:20])

    movie_id = 'interstellar+2014'
    rating_value = 5
    timestamp = datetime.now()

    # UPSERT query
    query = """
    INSERT INTO ratings (user_id, movie_id, rating, rated_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(user_id, movie_id)
    DO UPDATE SET
        rating = excluded.rating,
        rated_at = excluded.rated_at;
    """

    # Prepare data for executemany
    data = [(u_id, movie_id, rating_value, timestamp) for u_id in selected_users]

    # Execute batch UPSERT
    conn.cursor().executemany(query, data)
    conn.commit()
    




if __name__== '__main__':
    # request_recommendations()
    add_ratings_to_db()