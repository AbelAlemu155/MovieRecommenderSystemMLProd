
from evaluation.get_data import get_data
import numpy as np

import pandas as pd
from models.collaborative_filtering.helpers import compute_implicit_rating
from models.collaborative_filtering.item_based_filtering import ItemBasedFiltering
def evaluate_item_based():

    #  this is the query for item based
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
    
    train_df, val_df, test_df= get_data(query)

    top_k=20

    # get ground truth using implicit rating
    #  take the relevant for the gorundtruth
    test_df_with_rating= compute_implicit_rating(df=test_df)[['user_id', 'movie_id', 'implicit_pref']]
    user_col= 'user_id'
    movie_col= 'movie_id'
    test_user_item_matrix = test_df_with_rating.pivot_table(
            index=user_col,
            columns=movie_col,
            values='implicit_pref',
            aggfunc='mean'
            )
    
   
    

    #  first train the model
    model= ItemBasedFiltering(train_set= train_df, val_set=val_df)
    model.train()

    # evaluate on the test set 
   
 
    relevant_items = test_user_item_matrix.apply(
        lambda row: row.nlargest(top_k).index.tolist(), axis=1
    )

   
    predicted_items = [model.recommend(user_id) for user_id in test_user_item_matrix.index]

   
    relevant_matrix = np.array(relevant_items.tolist(), dtype=object)   # shape: (n_users, top_k)
    predicted_matrix = np.array(predicted_items, dtype=object)          # shape: (n_users, k_pred)

   
    precisions = np.array([
        np.isin(pred_row, rel_row).sum() / len(pred_row)
        if len(pred_row) > 0 else 0.0
        for pred_row, rel_row in zip(predicted_matrix, relevant_matrix)
    ])


    mean_precision = sum(precisions) / len(precisions)
    print(f'mean_precison is: {mean_precision}')

    return mean_precision



if __name__=='__main__':
    evaluate_item_based()




