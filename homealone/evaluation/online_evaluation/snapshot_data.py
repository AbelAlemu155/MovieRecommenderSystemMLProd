
from homealone.data_processing.db_writer import DBWriter
import pandas as pd
def get_snapshot():
    dbWriter= DBWriter()

    query = """
     select * from recommendations ;
    """

    recommendations_df = pd.read_sql_query(query, dbWriter.conn)

    query = """
    SELECT 
        wh.user_id,
        wh.movie_id,
        wh.duration,
        wh.watch_time, 
        r.rating
    FROM watch_history AS wh
    LEFT JOIN ratings AS r
        ON wh.user_id = r.user_id AND wh.movie_id = r.movie_id
    WHERE wh.user_id IN (
        SELECT user_id FROM recommendations
    );
    """

    # Load result into pandas DataFrame
    watch_rating_df = pd.read_sql_query(query, dbWriter.conn)

    # Close the connection
    dbWriter.conn.close()

    return recommendations_df, watch_rating_df


def preprocess_snapshot(recommendations_df, watch_rating_df):
    if 'recommended_items' in df.columns:
        recommendations_df.rename(columns={'recommended_items': 'movie_ids'}, inplace=True)
    
    recommendations_df['movie_id'] = recommendations_df['movie_ids'].str.split(',')
    recommendations_df = recommendations_df.explode('movie_id').drop(columns='movie_ids').reset_index(drop=True)
    #  the size of the new dataframe will be 20 times 
    #  merge the data 
    df = recommendations_df.merge(watch_rating_df, on=['user_id', 'movie_id'], how='left')

    df['completion_rate'] = df['watch_time'] / df['movie_duration']
    return df    