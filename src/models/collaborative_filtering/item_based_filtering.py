
from .helpers import compute_similarity, default_recommender
from data_processing.db_writer import DBWriter
import pandas as pd, numpy as np, joblib

class ItemBasedFiltering:

    def __init__(self, type='cosine', top_n= 20,alpha=0.6, beta=0.01, lambda_decay=0.01):
        self.type= type
        self.top_n = top_n
        self.alpha= alpha
        self.beta= beta
        self.lambda_decay= lambda_decay

        
    
    def load(self):
        writer= DBWriter()
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
        df= pd.read_sql_query(query, writer.conn, parse_dates=["last_watch"])
        writer.conn.close()
        return df
    def compute_implicit_rating(self):
      

        df= self.load()
        df['rating']= df['rating'] / 5

        df['watch_engagement']= (df['total_watch_time'] / df['runtime']).clip(upper=1.0)
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

        return df
    
    def build_item_matrix(self):
       
    
        df= self.compute_implicit_rating()[['user_id', 'movie_id', 'implicit_pref']]

        user_col= 'user_id'
        movie_col= 'movie_id'
        user_item_matrix = df.pivot_table(
            index=user_col,
            columns=movie_col,
            values='implicit_pref',
            aggfunc='mean'
            )
        # implicit rating
    
        # TODO: write the item matrix to a db
        return user_item_matrix
    
    def train(self):
        self.user_item_matrix = self.build_item_matrix()
        item_sim_df= compute_similarity(self.user_item_matrix.T)
        self.item_sim_df = item_sim_df

        # save the model 
        joblib.dump(self, './models/collaborative_filtering/saved_models/item_based_models.pkl')


    def recommend(self, user_id):
        # take transpose
        
        # load trained model
        itembasedF = joblib.load('./models/collaborative_filtering/saved_models/item_based_models.pkl')
        if user_id not in itembasedF.user_item_matrix.index:
            return default_recommender(itembasedF.user_item_matrix, self.top_n)
        
        user_ratings = itembasedF.user_item_matrix.loc[user_id].fillna(0)

        # Weighted sum of similarities
        scores = itembasedF.item_sim_df.dot(user_ratings) / (itembasedF.item_sim_df.sum(axis=1) + 1e-9)

        # Remove items already interacted with
        scores = scores.drop(user_ratings[user_ratings > 0].index, errors='ignore')

        return scores.sort_values(ascending=False).head(self.top_n).index.tolist()





if __name__== '__main__':

   
    itemBasedF= ItemBasedFiltering()
    # itemBasedF.train()

    u_id = 1576
    # u_id = 1728
    # u_id = 149900
    # u_id = -12
   
    
    import time
    start_time = time.perf_counter()
    print(itemBasedF.recommend(u_id))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")


    


    






    
