

import numpy as np

import pandas as pd
from models.collaborative_filtering.item_based_filtering import ItemBasedFiltering
from models.collaborative_filtering.helpers import compute_implicit_rating
from .get_data import get_data, get_random_users, get__counts

import time


import requests

def evaluate_item_based():

    #  this is the query for item based
   
    
    train_df, val_df, test_df= get_data()

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



def evaluate_average_latency():
    

    users= get_random_users()  

    latency= 0
    
    for u_id in users:
        url = f"http://127.0.0.1:8082/recommend/{u_id}"
        params = {
        "userid": u_id,
        
        }
        start_time = time.time() 
        response = requests.get(url, params=params)
        end_time = time.time()

        latency+= end_time - start_time
    
    print(latency*1000/ len(users))
    #  return time in milliseconds
    return latency * 1000/ len(users) 
        


def evaluate_training_cost():
    # query the database to find the number of users, items, and interactions used for the models

    U, I , N = get__counts()
    top_k=20
    return top_k *(  U * I - N)


if __name__=='__main__':
    # evaluate_average_latency()
    evaluate_item_based()
   
    




