from data_processing.db_writer import DBWriter
import pandas as pd, math

# query is the sql query to fetch the data for tha model
def get_data(query): 
    #  split the data 80% to 20% rule using time into train and test data
    writer= DBWriter()
    df= pd.read_sql_query(query, writer.conn, parse_dates=["last_watch"])
    writer.conn.close()

    #  split the data 80% to 20% rule using time into train and test data
    df_sorted = df.sort_values(by='last_watch')

    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    return train_df, test_df



    


    


    