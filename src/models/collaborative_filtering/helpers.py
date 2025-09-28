
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd , numpy as np

# the matrix might be both item based and user based 
def compute_similarity(matrix, type='cosine'):
    c_matrix= matrix.fillna(0) 
    if(type== 'cosine'):
        similarity = cosine_similarity(c_matrix)
        sim_df = pd.DataFrame(similarity, 
                               index=matrix.index, 
                               columns=matrix.index)
        return sim_df
    # unimplemented cosine is the default used 
    return None

# recommend popular items
def default_recommender(user_item_matrix, top_n):
    popular_items = user_item_matrix.mean(axis=0, skipna=True).sort_values(ascending=False).head(top_n)

    return popular_items.index.tolist()

def compute_implicit_rating(df,alpha=0.6, beta=0.01, lambda_decay=0.01):
      

        df= df.copy()
        df['rating']= df['rating'] / 5

        df['watch_engagement']= (df['total_watch_time'] / df['runtime']).clip(upper=1.0)
        df.dropna(subset=['watch_engagement'])

        df['p_star'] = np.where(
        df['rating'].notna(), 
        alpha * df['rating'] + (1 - alpha) * df['watch_engagement'], 
        df['watch_engagement']
        )

        df['f_ui'] = 1.0 / (1.0 + beta * df['gap_watch'])


        now = pd.Timestamp.now()

        df['day_diff']=(now - df['last_watch']).dt.total_seconds() / (3600 * 24)
        # Difference in days
        df['recency'] =  np.exp(-lambda_decay * df['day_diff'])

        df['implicit_pref'] = df['p_star'] * df['f_ui'] * df['recency']

        return df

        
    

