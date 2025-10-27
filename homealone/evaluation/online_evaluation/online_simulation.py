

import requests

from homealone.data_processing.db_writer import DBWriter
import os
import sqlite3
import requests

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
                    print(f"User {uid}: Success -> {response.json()}")
                else:
                    print(f"User {uid}: Failed with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"User {uid}: Error -> {e}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()




if __name__== '__main__':
    request_recommendations()