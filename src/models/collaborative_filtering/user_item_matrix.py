import pandas as pd, numpy as np

class UserItemMatrix:
    def __init__(self, writer,alpha=0.6, beta=0.01, lambda_decay=0.01):
        self.writer=writer
        self.alpha= alpha
        self.beta= beta
        self.lambda_decay= lambda_decay
    
    def process_data(self):
      
        query = """
                SELECT 
                    WH.user_id,
                    WH.movie_id,
                    WH.total_watch_time,
                    WH.last_watch,
                    (julianday(WH.last_watch) - julianday(WH.start_time)) * 24*60 AS gap_watch,
                    R.rating
                FROM watch_history AS WH
                LEFT JOIN ratings AS R
                ON WH.user_id = R.user_id AND WH.movie_id = R.movie_id;
                    """
        self.df = pd.read_sql_query(query, self.writer.conn, parse_dates=["last_watch"])
    
    # TODO: this will be replaced with movie duration logic
    def get_median(self):
        assert self.df is not None, "data not preprocessed" 
        self.median= self.df['total_watch_time'].median()
        
    def compute_implicit_rating(self):
        assert self.median is not None, "median not computed"

        df= self.df.copy()
        df['median_watch'] = self.median

        df['rating']= df['rating'] / 5

        df['watch_engagement']= (df['total_watch_time'] / df['median_watch']).clip(upper=1.0)
        df.dropna(subset=['watch_engagement'])

        df['p_star'] = np.where(
        df['rating'].notna(), 
        self.alpha * df['rating'] + (1 - self.alpha) * df['watch_engagement'], 
        df['watch_engagement']
        )

        df['f_ui'] = 1.0 / (1.0 + self.beta * df['gap_watch'])


        now = pd.Timestamp.now()

        df['day_diff']=(now - df['last_watch']).dt.total_seconds() / (3600 * 24)
        # Difference in days
        df['recency'] =  np.exp(-self.lambda_decay * df['day_diff'])

        df['implicit_pref'] = df['p_star'] * df['f_ui'] * df['recency']

        return df['implicit_pref']
    
    def build_item_matrix(self):
        self.process_data()
        self.get_median()
        df= self.df[['user_id', 'movie_id']].copy()
        df['implicit_pref']= self.compute_implicit_rating()

        user_col= 'user_id'
        movie_col= 'movie_id'
        user_item_matrix = df.pivot_table(
            index=user_col,
            columns=movie_col,
            values='implicit_pref',
            aggfunc='mean'
            )
    
        self.writer.conn.close()

        # TODO: write the item matrix to a db
        return user_item_matrix
    
    





# if __name__ == '__main__':
#     import os, sqlite3
#     db_path = os.getenv('DB_PATH', 'src/db/home_alone.db')
    
#     conn = sqlite3.connect(db_path)
#     query = """
#                 SELECT 
#                     WH.user_id,
#                     WH.movie_id,
#                     WH.total_watch_time,
#                     WH.last_watch,
#                     (julianday(WH.last_watch) - julianday(WH.start_time)) * 24*60 AS gap_watch,
#                     R.rating
#                 FROM watch_history AS WH
#                 LEFT JOIN ratings AS R
#                 ON WH.user_id = R.user_id AND WH.movie_id = R.movie_id;
#                     """

#     df = pd.read_sql_query(query, conn, parse_dates=["last_watch"])



# def implicit_preference(
#     rating,                # float: 0-5, or None if missing
#     watch_time,            # float: total watched time in seconds/minutes
#     movie_duration,        # float: total duration of the movie in same unit as watch_time
#     start_time,            # float: timestamp of first watch (e.g., unix time or datetime in seconds)
#     last_watch,            # float: timestamp of last watch
#     now=None,              # float: current timestamp (default: now)
#     alpha=0.6,             # weight for rating vs completion
#     beta=0.5,              # span factor weight
#     lambda_decay=0.01      # recency decay rate
# ):
#     """
#     Computes an implicit user-item preference score.
#     Returns NaN if the user has not interacted (watched) the item.
#     """
#     # 0. Handle no interaction
#     if watch_time is None or watch_time == 0:
#         return np.nan  # no interaction

#     if now is None:
#         now = last_watch  # fallback if current time not provided

#     # 1. Normalized rating
#     r_ui = rating / 5.0 if rating is not None else 0.0

#     # 2. Completion ratio
#     c_ui = min(1.0, watch_time / movie_duration)

#     # 3. Watch duration multiplier (log-scaled)
#     w_ui = np.log1p(watch_time / movie_duration)  # log(1 + watch_ratio)

#     # 4. Base preference combining rating and completion
#     p_star = (alpha * r_ui + (1 - alpha) * c_ui) * w_ui

#     # 5. Completion span factor
#     span = last_watch - start_time
#     f_ui = 1.0 / (1.0 + beta * span / movie_duration)  # longer spans reduce confidence

#     # 6. Recency decay factor
#     recency = np.exp(-lambda_decay * (now - last_watch))

#     # 7. Final implicit preference
#     p_ui = p_star * f_ui * recency

#     return p_ui
