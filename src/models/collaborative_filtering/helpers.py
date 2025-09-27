
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 

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

        
    

