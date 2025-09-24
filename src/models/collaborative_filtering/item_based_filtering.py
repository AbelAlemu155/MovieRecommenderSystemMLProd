
from .helpers import compute_similarity, default_recommender
from data_processing.db_writer import DBWriter
from .user_item_matrix import UserItemMatrix

class ItemBasedFiltering:

    def __init__(self, uimBuilder, type='cosine', top_n= 20):
        self.type= type
        self.top_n = top_n
        self.user_item_matrix= uimBuilder.build_item_matrix()
    

    def recommend_items(self, user_id):
        # take transpose
        item_sim_df= compute_similarity(self.user_item_matrix.T)
        self.item_sim_df = item_sim_df

        if user_id not in self.user_item_matrix.index:
            return default_recommender(self.user_item_matrix, self.top_n)
        
        user_ratings = self.user_item_matrix.loc[user_id].fillna(0)

        # Weighted sum of similarities
        scores = self.item_sim_df.dot(user_ratings) / (self.item_sim_df.sum(axis=1) + 1e-9)

        # Remove items already interacted with
        scores = scores.drop(user_ratings[user_ratings > 0].index, errors='ignore')

        return scores.sort_values(ascending=False).head(self.top_n).index.tolist()





if __name__== '__main__':

    writer = DBWriter()
    
    uimBuilder = UserItemMatrix(writer=writer)

    u_id = 1576
    u_id = 1728
    # u_id = 149900
    # u_id = -11
    itemF= ItemBasedFiltering(uimBuilder= uimBuilder)

    print(itemF.recommend_items(u_id))


    


    






    
