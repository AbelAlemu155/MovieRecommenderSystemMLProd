import sqlite3
from surprise import SVD, KNNBasic, NMF  # Import NMF
from surprise.model_selection import train_test_split
from surprise import dump     
import logging
from logging.handlers import RotatingFileHandler
import os
from flask import Flask, jsonify, abort, g
import pickle
import random
from telemetry import Telemetry
import time

# We load these ONCE at startup
model_svd = None
# model_knn = None
model_nmf = None
trainset = None
fallback_pool = []
# --- Define file paths ---
# Paths are absolute, as used in container environments
MODEL_DIR = "/app/models"
DATABASE = '/app/db/home_alone.db'
TRAINSET_PATH = os.path.join(MODEL_DIR, "trainset.dump")
SVD_MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.dump")
# KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.dump")
NMF_MODEL_PATH = os.path.join(MODEL_DIR, "nmf_model.dump")

EXTRACTED_MOVIES = [
"toy+story+1995" for _ in range(20),
]
MASTER_FALLBACK_MOVIES = EXTRACTED_MOVIES


def load_resources():
    """
    Load all models and the pre-built trainset from disk.
    This function is called once when the Flask app starts.
    """
    # global model_svd, model_knn, model_nmf, trainset, fallback_pool
    global model_svd, model_nmf, trainset, fallback_pool

    try:
        # 1. Load the pre-built trainset
        print(f"Loading trainset from {TRAINSET_PATH}...")
        # dump.load returns (predictions, algo, trainset)
        # We only need the third item.
        # _, _, loaded_trainset = dump.load(TRAINSET_PATH)
        trainset = pickle.load(open(TRAINSET_PATH, 'rb'))
        if trainset is None:
            raise Exception(f"Trainset not found in {TRAINSET_PATH}")
        print("Trainset loaded successfully.")

        # 2. Load the pre-trained models
        print(f"Loading SVD model from {SVD_MODEL_PATH}...")
        # dump.load returns (predictions, algo, trainset)
        # We only need the second item (algo).
        _, model_svd = dump.load(SVD_MODEL_PATH)
        print("SVD model loaded.")
        
        # print(f"Loading KNN model from {KNN_MODEL_PATH}...")
        # _, model_knn = dump.load(KNN_MODEL_PATH)
        # print("KNN model loaded.")
        
        print(f"Loading NMF model from {NMF_MODEL_PATH}...")
        _, model_nmf = dump.load(NMF_MODEL_PATH)
        print("NMF model loaded.")
        
        
        fallback_pool = prepare_fallback_pool(
            trainset, 
            min_ratings=1, 
            min_avg_rating=2
        )
                
        print("All resources loaded successfully.")
        

    except FileNotFoundError as e:
        print(f"Error: Model or trainset file not found.")
        print(f"Details: {e}")
        # Stop the server from starting if resources are missing
        raise
    except Exception as e:
        print(f"An error occurred during resource loading: {e}")
        raise

def prepare_fallback_pool(trainset, min_ratings=1, min_avg_rating=3.0):
    """
    Creates a list of high-quality movies from the trainset.
    """
    print(f"Building fallback pool: min_ratings={min_ratings}, min_avg_rating={min_avg_rating}")
    pool = []
    
    # trainset.ir is a dict: {inner_item_id: [(inner_user_id, rating), ...]}
    for inner_iid, ratings in trainset.ir.items():
        # print(f"Processing item {inner_iid} with {ratings[0][0]} ratings.")
        # print(f"Processing item {trainset.to_raw_uid(ratings[0][0])} with {len(ratings)} ratings.")
        # 1. Check for min_ratings
        num_ratings = len(ratings)
        if num_ratings >= min_ratings:
            
            # 2. Calculate average rating
            # We only need the rating (r) from each (u, r) tuple
            avg_rating = sum(r for u, r in ratings) / num_ratings
            
            # 3. Check for min_avg_rating
            if avg_rating >= min_avg_rating:
                
                # 4. Add to pool (convert inner ID back to raw ID)
                pool.append({
                    "movie_id": trainset.to_raw_iid(inner_iid),
                    "avg_rating": round(avg_rating, 2)
                })
                
    print(f"Built fallback pool with {len(pool)} movies.")
    return pool

def get_random_fallback(pool, n=20):
    """
    Gets N random movies from the pre-compiled pool.
    """
    num_to_sample = min(n, len(pool))

    if num_to_sample == 0:
        return ""

    # 'pool' is now a list of dicts, random.sample works directly
    random_fallback = random.sample(pool, num_to_sample)
    # sort avg_rating descending
    random_fallback.sort(key=lambda x: x['avg_rating'], reverse=True)
    # return comma separated movie_ids
    return ", ".join([str(movie['movie_id']) for movie in random_fallback])

def get_top_n(predictions, n=20):
    """
    Sort predictions and return the top-N in a clean list.
    """
    # 1. Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    
    top_n_preds = predictions[:n]
    
    top_n_list = [pred.iid for pred in top_n_preds]

    return ", ".join([str(iid) for iid in top_n_list])

def get_random_fallback_simple(n=20):
    """
    Get N random movies from master fallback list.
    This is faster than querying the database.
    """
    num_to_sample = min(n, len(MASTER_FALLBACK_MOVIES))
    
    if num_to_sample == 0:
        return ""
    
    # Random sample from master list
    random_movies = random.sample(MASTER_FALLBACK_MOVIES, num_to_sample)
    
    return ",".join(random_movies)

# --- Initialize the Flask App ---
app = Flask(__name__)

def get_db():
    """Get database connection for current request"""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, timeout=30.0)
        # Enable WAL mode for better concurrency
        g.db.execute('PRAGMA journal_mode=WAL')
        g.db.execute('PRAGMA synchronous=NORMAL')
    return g.db

@app.teardown_appcontext
def close_db(error):
    """Close database connection at end of request"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route('/')
def hello():
    return 'Home Alone Recommendation API is running!'

# --- Define the API Endpoint ---
@app.route('/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    """
    Generate recommendations with optimizations for speed.
    """
    start_time = time.time()
    logging.info(f"Received recommendation request for user_id: {user_id}")
    
    if not user_id.isdigit():
        return jsonify({"error": "user_id must be an integer."}), 400
    
    raw_user_id = int(user_id)

    # temporary fix for 400 thingy till we clear the db
    recommendation_response = get_random_fallback_simple(n=20)
    logging.warning(recommendation_response)
    return jsonify(recommendation_response)

    # conn = get_db()
    # telemetry = Telemetry(conn)
    
    # try:
    #     inner_uid = trainset.to_inner_uid(raw_user_id)
    # except ValueError:
    #     logging.warning(f"User {raw_user_id} not in trainset. Using fallback.")
    #     recommendation_response = get_random_fallback(fallback_pool, n=20)
    #     telemetry.log('recommendation', {
    #         'user_id': raw_user_id,
    #         'recommended_items': recommendation_response,
    #         'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    #     })
    #     logging.warning(recommendation_response)
    #     return jsonify(recommendation_response)
    
    # # Get unrated items
    # rated_items_inner = {iid for (iid, r) in trainset.ur[inner_uid]}
    # all_items_inner = set(trainset.all_items())
    # unrated_items_inner = all_items_inner - rated_items_inner
    
    # MAX_PREDICTIONS = 200
    # if len(unrated_items_inner) > MAX_PREDICTIONS:
    #     popular_items = {item['movie_id'] for item in fallback_pool[:500]}
    #     popular_unrated = unrated_items_inner & {trainset.to_inner_iid(iid) for iid in popular_items if trainset.knows_item(iid)}
        
    #     if len(popular_unrated) >= MAX_PREDICTIONS:
    #         unrated_items_inner = random.sample(list(popular_unrated), MAX_PREDICTIONS)
    #     else:
    #         remaining = MAX_PREDICTIONS - len(popular_unrated)
    #         random_items = random.sample(list(unrated_items_inner - popular_unrated), remaining)
    #         unrated_items_inner = list(popular_unrated) + random_items
    
    # # Create testset
    # testset_for_user = [
    #     (raw_user_id, trainset.to_raw_iid(iid), 4.0) 
    #     for iid in unrated_items_inner
    # ]
    
    # if not testset_for_user:
    #     recommendation_response = get_random_fallback(fallback_pool, n=20)
    #     telemetry.log('recommendation', {
    #         'user_id': raw_user_id,
    #         'recommended_items': recommendation_response,
    #         'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    #     })
    #     logging.warning(recommendation_response)
    #     return jsonify(recommendation_response)
    
    # svd_predictions = model_svd.test(testset_for_user)
    
    # # Get top 20
    # recommendation_response = get_top_n(svd_predictions, n=20)
    
    # # Log telemetry
    # elapsed = time.time() - start_time
    # logging.info(f"Generated recommendations in {elapsed*1000:.0f}ms")
    
    # telemetry.log('recommendation', {
    #     'user_id': raw_user_id,
    #     'recommended_items': recommendation_response,
    #     'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    #     'response_time_ms': int(elapsed * 1000)
    # })
    # logging.warning(recommendation_response)
    
    # return jsonify(recommendation_response)


if __name__ == '__main__':
    # Load all models and data before starting the server
    logging.info("Starting server, loading resources...")
    load_resources()
    
    # Run the app
    logging.info("Server ready. Access at http://127.0.0.1:8082/recommend/<user_id>")
    app.run(debug=False, port=8082, host='0.0.0.0')