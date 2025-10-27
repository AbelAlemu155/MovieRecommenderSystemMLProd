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
model_knn = None
model_nmf = None
trainset = None
fallback_pool = []
# --- Define file paths ---
# Paths are absolute, as used in container environments
MODEL_DIR = "/app/models"
DATABASE = '/app/db/home_alone.db'
TRAINSET_PATH = os.path.join(MODEL_DIR, "trainset.dump")
SVD_MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.dump")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.dump")
NMF_MODEL_PATH = os.path.join(MODEL_DIR, "nmf_model.dump")


def load_resources():
    """
    Load all models and the pre-built trainset from disk.
    This function is called once when the Flask app starts.
    """
    global model_svd, model_knn, model_nmf, trainset, fallback_pool

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
        
        print(f"Loading KNN model from {KNN_MODEL_PATH}...")
        _, model_knn = dump.load(KNN_MODEL_PATH)
        print("KNN model loaded.")
        
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
    return ",".join([str(movie['movie_id']) for movie in random_fallback])

def get_top_n(predictions, n=20):
    """
    Sort predictions and return the top-N in a clean list.
    """
    # 1. Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    
    top_n_preds = predictions[:n]
    
    top_n_list = [pred.iid for pred in top_n_preds]

    return ",".join([str(iid) for iid in top_n_list])


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
    Generate recommendations for a given user_id.
    """
    print(f"Received recommendation request for user_id: {user_id}")
    conn = get_db()
    telemetry = Telemetry(conn)
    try:
        # 1. Convert user_id from URL to the correct type.
        # Check if user_id is purely numeric first
        if user_id.isdigit():
            raw_user_id = int(user_id)
        else:
            return jsonify({"error": "user_id must be an integer."}), 400
    except ValueError:
        return jsonify({"error": "Invalid user_id format."}), 400

    try:
        inner_uid = trainset.to_inner_uid(raw_user_id)
        logging.info(f"User {raw_user_id} mapped to inner ID {inner_uid}.")
    except ValueError:
        logging.warning(f"User {raw_user_id} not found in trainset. Using fallback recommendations.")
        recommendation_response =  get_random_fallback(fallback_pool, n=20)
        telemetry.log('recommendation', {
            'user_id': raw_user_id,
            'recommended_items': recommendation_response,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        })
        return jsonify(recommendation_response)

    # 3. Find all items the user has NOT rated (efficiently)
    rated_items_inner = {iid for (iid, r) in trainset.ur[inner_uid]}
    all_items_inner = set(trainset.all_items())
    unrated_items_inner = all_items_inner - rated_items_inner
    
    # 4. Format this set as a "testset" for the predict() function
    testset_for_user = [
        (raw_user_id, trainset.to_raw_iid(iid), 4.0) 
        for iid in unrated_items_inner
    ]

    if not testset_for_user:
        recommendation_response =  get_random_fallback(fallback_pool, n=20)
        telemetry.log('recommendation', {
            'user_id': raw_user_id,
            'recommended_items': recommendation_response,
            'timestamp': None
        })
        return jsonify(recommendation_response), 200

    svd_predictions = model_svd.test(testset_for_user)
    # knn_predictions = model_knn.test(testset_for_user)
    # nmf_predictions = model_nmf.test(testset_for_user)

    recommendation_response = get_top_n(svd_predictions, n=20)
    telemetry.log('recommendation', {
        'user_id': raw_user_id,
        'recommended_items': recommendation_response,
        'timestamp': None
    })
    return jsonify(recommendation_response)



if __name__ == '__main__':
    # Load all models and data before starting the server
    logging.info("Starting server, loading resources...")
    load_resources()
    
    # Run the app
    logging.info("Server ready. Access at http://127.0.0.1:8082/recommend/<user_id>")
    app.run(debug=True, port=8082, host='0.0.0.0')