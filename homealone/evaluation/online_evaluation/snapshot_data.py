

import pandas as pd, os
import sqlite3

from experimentation.map_users_to_group import define_hash, pyhash
def get_snapshot(treatment_group=None):
    

    query = """
     select * from recommendations ;
    """
    # Use environment variable or default to docker path
    # db_dir = os.getenv("DB_PATH", "/app/db")

    # db_dir = "homealone/db"
    db_dir= "/app/db"
    db_path = os.path.join(db_dir, "home_alone.db")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    recommendations_df = pd.read_sql_query(query, conn,  parse_dates=['timestamp'] )
    if(treatment_group is None):

        query = """
            SELECT 
                wh.user_id,
                wh.movie_id,
                wh.start_time,
                wh.last_watch,
                r.rating,
                m.runtime,
                r.rated_at
            FROM watch_history AS wh
            LEFT JOIN ratings AS r
                ON wh.user_id = r.user_id AND wh.movie_id = r.movie_id
            LEFT JOIN movies AS m
                ON wh.movie_id = m.movie_id
            WHERE wh.user_id IN (
                SELECT user_id FROM recommendations 
            );
            """
    

    elif(treatment_group):
        print("treatement group is true")
        # Load recommendations table
        recs_df = pd.read_sql_query("SELECT DISTINCT user_id FROM recommendations", conn)
        recs_df['pyhash'] = recs_df['user_id'].apply(pyhash)
        selected_users = recs_df[recs_df['pyhash'] % 100 < 10]['user_id'].tolist()

        # Now filter in SQL with fixed list
        query = f"""
        SELECT 
            wh.user_id,
            wh.movie_id,
            wh.start_time,
            wh.last_watch,
            r.rating,
            m.runtime,
            r.rated_at
        FROM watch_history AS wh
        LEFT JOIN ratings AS r
            ON wh.user_id = r.user_id AND wh.movie_id = r.movie_id
        LEFT JOIN movies AS m
            ON wh.movie_id = m.movie_id
        WHERE wh.user_id IN ({','.join(map(str, selected_users))});
        """
        # print(len(selected_users))
    else:
        print("treatement group is False")
        
        # Load recommendations table
        recs_df = pd.read_sql_query("SELECT DISTINCT user_id FROM recommendations", conn)
        recs_df['pyhash'] = recs_df['user_id'].apply(pyhash)
        selected_users = recs_df[recs_df['pyhash'] % 100 >= 10]['user_id'].tolist()

        # Now filter in SQL with fixed list
        query = f"""
        SELECT 
            wh.user_id,
            wh.movie_id,
            wh.start_time,
            wh.last_watch,
            r.rating,
            m.runtime,
            r.rated_at
        FROM watch_history AS wh
        LEFT JOIN ratings AS r
            ON wh.user_id = r.user_id AND wh.movie_id = r.movie_id
        LEFT JOIN movies AS m
            ON wh.movie_id = m.movie_id
        WHERE wh.user_id IN ({','.join(map(str, selected_users))});
        """
        # print(len(selected_users))

    # Load result into pandas DataFrame
    watch_rating_df = pd.read_sql_query(query, conn, 
                                        parse_dates=["start_time", "last_watch", "rated_at"]
                                         )

    
    
    # print(watch_rating_df[['rated_at']].head)
    # Close the connection
    conn.close()
    return recommendations_df, watch_rating_df


def preprocess_snapshot(recommendations_df, watch_rating_df):
    recommendations_df["user_id"] = recommendations_df["user_id"].astype(int)
    watch_rating_df["user_id"] = watch_rating_df["user_id"].astype(int)
    recommendations_df.rename(columns={'recommended_items': 'movie_ids'}, inplace=True)
    
    recommendations_df['movie_id'] = recommendations_df['movie_ids'].str.split(',')
    recommendations_df = recommendations_df.explode('movie_id').drop(columns='movie_ids').reset_index(drop=True)
    #  [the size of the new dataframe will be 20 times, this is an important test condition ]
    #  merge the data 
    df = recommendations_df.merge(watch_rating_df, on=['user_id', 'movie_id'], how='left')
    df["watch_duration_minutes"] = (df["last_watch"] - df["start_time"]).dt.total_seconds() / 60
    df['completion_rate'] = df['watch_duration_minutes'] / df['runtime']
    return df    