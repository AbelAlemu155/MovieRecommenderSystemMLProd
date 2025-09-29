# from src.models.content_based.content_based_rec import ContentBasedRecommender


# # ContentBasedRecommender().train()


# m = ContentBasedRecommender().recommend(user_id=int(56), k=20)
# print(m)




from .get_data import get_data
from models.collaborative_filtering.item_based_filtering import ItemBasedFiltering

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


    # get ground truth using implicit rating
    #  take the relevant for the gorundtruth


    #  first train the model
model= ItemBasedFiltering(train_set= train_df, val_set=val_df)
model.train()
print(model.recommend(user_id=56))
