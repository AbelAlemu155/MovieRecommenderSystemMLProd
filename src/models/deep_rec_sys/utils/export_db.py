import sqlite3, pandas as pd

conn = sqlite3.connect('src/db/home_alone.db')
pd.read_sql_query("SELECT * FROM watch_history", conn).to_csv("src/db/watch_history.csv", index=False)
pd.read_sql_query("SELECT * FROM ratings", conn).to_csv("src/db/ratings.csv", index=False)
pd.read_sql_query("SELECT * FROM movies", conn).to_csv("src/db/movies.csv", index=False)


conn.close()