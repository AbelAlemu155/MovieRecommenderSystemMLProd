from data_processing.db_writer import DBWriter
import pandas as pd, math

# query is the sql query to fetch the data for tha model
def get_data(): 

    query = """
                SELECT 
                    WH.user_id,
                    WH.movie_id,
                    WH.total_watch_time,
                    WH.last_watch,
                    (julianday(WH.last_watch) - julianday(WH.start_time)) * 24*60 AS gap_watch,
                    R.rating, 
                    m.runtime
                FROM watch_history AS WH
                JOIN movies as m ON m.movie_id = WH.movie_id
                LEFT JOIN ratings AS R
                ON WH.user_id = R.user_id AND WH.movie_id = R.movie_id;
                    """
    #  split the data 80% to 20% rule using time into train and test data
    writer= DBWriter()
    df= pd.read_sql_query(query, writer.conn, parse_dates=["last_watch"])
    writer.conn.close()

    #  split the data 80% to 20% rule using time into train and test data
    df_sorted = df.sort_values(by='last_watch')

    train_idx = int(len(df_sorted) * 0.7)
    val_idx   = int(len(df_sorted) * 0.85)  # 70% + 15% = 85%

    # Split the data
    train_df = df_sorted.iloc[:train_idx].copy()
    val_df   = df_sorted.iloc[train_idx:val_idx].copy()
    test_df  = df_sorted.iloc[val_idx:].copy()

    return train_df, val_df,  test_df



    


    


    