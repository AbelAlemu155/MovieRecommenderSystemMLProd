import sqlite3
from surprise import SVD, KNNBasic, NMF
from surprise.model_selection import train_test_split
from surprise import dump     
import logging
from logging.handlers import RotatingFileHandler
import os
from flask import Flask, jsonify, abort, g, request
import pickle
import random
from telemetry import Telemetry
import time
import json
import pandas as pd
from threading import Thread, Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Define file paths ---
MODEL_DIR = "/app/models"
DATABASE = '/app/db/home_alone.db'
TRAINSET_PATH = os.path.join(MODEL_DIR, "trainset.dump")
SVD_MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.dump")
NMF_MODEL_PATH = os.path.join(MODEL_DIR, "nmf_model.dump")
PROVENANCE_PATH = os.path.join(MODEL_DIR, "provenance_manifest.json")

# Global variables with thread safety
model_lock = Lock()
request_counter = 0
request_lock = Lock()
model_reload_in_progress = False
model_svd = None
model_nmf = None
trainset = None
fallback_pool = []
PROVENANCE = {}

MASTER_FALLBACK_MOVIES = [
    "being+john+malkovich+1999",
    "love+the+beast+2009",
    "rabbit+hole+2010",
    "9+2005",
    "a+special+day+1977",
    "angel+heart+1987",
    "big+star+nothing+can+hurt+me+2013",
    "burnt+by+the+sun+1994",
    "bye+bye+brasil+1980",
    "chocolat+1988",
    "citizen+ruth+1996",
    "clueless+1995",
    "cul-de-sac+1966",
    "dark+passage+1947",
    "dear+frankie+2004",
    "dirty+dingus+magee+1970",
    "drive_+he+said+1971",
    "east+is+east+1999",
    "farewell_+my+lovely+1975",
    "first+name+carmen+1983",
    "toy+story+1995"
]


class ModelFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_reload = 0
        self.debounce_seconds = 5
        self.replica_id = int(os.getenv('REPLICA_ID', '1'))
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        filename = os.path.basename(event.src_path)
        relevant_files = ["svd_model.dump", "nmf_model.dump", "trainset.dump"]
        
        if filename in relevant_files:
            current_time = time.time()
            if current_time - self.last_reload > self.debounce_seconds:
                # Replica 1 waits 0s, Replica 2 waits 60s, Replica 3 waits 120s
                delay = (self.replica_id - 1) * 60
                
                logging.info(f"Detected change in {filename}")
                logging.info(f"Replica {self.replica_id} waiting {delay}s before reload...")
                
                time.sleep(delay)
                
                self.last_reload = time.time()
                logging.info(f"Replica {self.replica_id} starting reload...")
                Thread(target=load_resources, daemon=True).start()

def get_random_fallback_simple(n=20):
    """
    Get N random movies from master fallback list.
    """
    num_to_sample = min(n, len(MASTER_FALLBACK_MOVIES))
    
    if num_to_sample == 0:
        return ""
    
    random.shuffle(MASTER_FALLBACK_MOVIES)
    random_movies = random.sample(MASTER_FALLBACK_MOVIES, num_to_sample)
    
    return ",".join(random_movies)


def load_resources():
    """
    Load all models and the pre-built trainset from disk.
    Thread-safe reload that swaps resources atomically.
    """
    global model_svd, model_nmf, trainset, fallback_pool, PROVENANCE

    try:
        logging.info("=" * 60)
        logging.info("Starting resource reload...")
        
        # Load into temporary variables first
        temp_trainset = None
        temp_svd = None
        temp_nmf = None
        temp_fallback = []
        temp_provenance = {}
        
        # 1. Load provenance
        try:
            with open(PROVENANCE_PATH, "r") as f:
                temp_provenance = json.load(f)
            logging.info(f"Loaded provenance - Model: {temp_provenance.get('model_version', 'unknown')[:8]}, "
                        f"Data: {temp_provenance.get('data_version', 'unknown')[:8]}, "
                        f"Git: {temp_provenance.get('git_commit', 'unknown')[:8]}")
        except Exception as e:
            logging.warning(f"Could not load provenance manifest: {e}")
        
        # 2. Load trainset
        logging.info(f"Loading trainset from {TRAINSET_PATH}...")
        temp_trainset = pickle.load(open(TRAINSET_PATH, 'rb'))
        if temp_trainset is None:
            raise Exception(f"Trainset not found in {TRAINSET_PATH}")
        logging.info(f"✓ Trainset loaded: {temp_trainset.n_users} users, {temp_trainset.n_items} items")

        # 3. Load SVD model
        logging.info(f"Loading SVD model from {SVD_MODEL_PATH}...")
        _, temp_svd = dump.load(SVD_MODEL_PATH)
        logging.info("✓ SVD model loaded")
        
        # 4. Load NMF model
        logging.info(f"Loading NMF model from {NMF_MODEL_PATH}...")
        _, temp_nmf = dump.load(NMF_MODEL_PATH)
        logging.info("✓ NMF model loaded")
        
        # 5. Prepare fallback pool
        temp_fallback = prepare_fallback_pool(
            temp_trainset, 
            min_ratings=1, 
            min_avg_rating=2
        )
        logging.info(f"✓ Fallback pool prepared: {len(temp_fallback)} movies")
        
        # Atomic swap with lock
        with model_lock:
            model_svd = temp_svd
            model_nmf = temp_nmf
            trainset = temp_trainset
            fallback_pool = temp_fallback
            PROVENANCE = temp_provenance
        
        logging.info("✓ All resources reloaded successfully")
        logging.info("=" * 60)

    except FileNotFoundError as e:
        logging.error(f"Error: Model or trainset file not found: {e}")
        if model_svd is None:  # First load failed
            raise
    except Exception as e:
        logging.error(f"Error during resource loading: {e}")
        if model_svd is None:  # First load failed
            raise


def prepare_fallback_pool(trainset, min_ratings=1, min_avg_rating=3.0):
    """
    Creates a list of high-quality movies from the trainset.
    """
    logging.info(f"Building fallback pool: min_ratings={min_ratings}, min_avg_rating={min_avg_rating}")
    pool = []
    
    for inner_iid, ratings in trainset.ir.items():
        num_ratings = len(ratings)
        if num_ratings >= min_ratings:
            avg_rating = sum(r for u, r in ratings) / num_ratings
            
            if avg_rating >= min_avg_rating:
                pool.append({
                    "movie_id": trainset.to_raw_iid(inner_iid),
                    "avg_rating": round(avg_rating, 2)
                })
                
    logging.info(f"Built fallback pool with {len(pool)} movies")
    return pool


def get_random_fallback(pool, n=20):
    """
    Gets N random movies from the pre-compiled pool.
    """
    num_to_sample = min(n, len(pool))

    if num_to_sample == 0:
        return ""

    random_fallback = random.sample(pool, num_to_sample)
    random_fallback.sort(key=lambda x: x['avg_rating'], reverse=True)
    return ",".join([str(movie['movie_id']) for movie in random_fallback])


def get_top_n(predictions, n=20):
    """
    Sort predictions and return the top-N in a clean list.
    """
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_preds = predictions[:n]
    top_n_list = [pred.iid for pred in top_n_preds]
    return ",".join([str(iid) for iid in top_n_list])


# --- Initialize the Flask App ---
app = Flask(__name__)

@app.before_request
def track_requests():
    """Track active requests"""
    global request_counter
    with request_lock:
        request_counter += 1


@app.after_request
def untrack_requests(response):
    """Untrack completed requests"""
    global request_counter
    with request_lock:
        request_counter -= 1
    return response


def get_db():
    """Get database connection for current request"""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, timeout=30.0)
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
    return jsonify({
        'status': 'running',
        'model_version': PROVENANCE.get('model_version', 'unknown')[:8],
        'data_version': PROVENANCE.get('data_version', 'unknown')[:8],
        'git_commit': PROVENANCE.get('git_commit', 'unknown')[:8],
        'mlflow_run_id': PROVENANCE.get('mlflow_run_id', 'unknown')
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with reload status"""
    if model_reload_in_progress:
        return jsonify({
            'status': 'reloading',
            'message': 'Model reload in progress'
        }), 503
    
    with model_lock:
        healthy = all([model_svd is not None, model_nmf is not None, trainset is not None])
    
    return jsonify({
        'status': 'ready' if healthy else 'unhealthy',
        'models_loaded': healthy,
        'active_requests': request_counter,
        'replica_id': os.getenv('REPLICA_ID', 'unknown'),
        'provenance': PROVENANCE
    }), 200 if healthy else 503

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Graceful model reload triggered by training service"""
    global model_reload_in_progress
    
    if model_reload_in_progress:
        return jsonify({"status": "error", "message": "Reload already in progress"}), 409
    
    model_reload_in_progress = True
    
    try:
        logging.info("=== Reload triggered by training service ===")
        
        # Wait for active requests to complete (max 30 seconds)
        logging.info(f"Draining {request_counter} active requests...")
        wait_start = time.time()
        while request_counter > 0 and (time.time() - wait_start) < 30:
            time.sleep(0.5)
        
        if request_counter > 0:
            logging.warning(f"Proceeding with {request_counter} active requests")
        
        # Use existing load_resources function
        load_resources()
        
        return jsonify({
            "status": "success",
            "message": "Models reloaded successfully",
            "provenance": PROVENANCE
        }), 200
        
    except Exception as e:
        logging.error(f"Failed to reload models: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    finally:
        model_reload_in_progress = False



@app.route('/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    """
    Generate recommendations with thread-safe model access.
    """
    start_time = time.time()
    logging.info(f"Received recommendation request for user_id: {user_id}")
    
    if not user_id.isdigit():
        return jsonify({"error": "user_id must be an integer."}), 400
    
    raw_user_id = int(user_id)

    conn = get_db()
    telemetry = Telemetry(conn)
    
    # Use lock for reading model state
    with model_lock:
        current_trainset = trainset
        current_svd = model_svd
        current_fallback = fallback_pool
        current_provenance = PROVENANCE.copy()
    
    try:
        inner_uid = current_trainset.to_inner_uid(raw_user_id)
    except ValueError:
        logging.warning(f"User {raw_user_id} not found in trainset. Using fallback recommendations.")
        recommendation_response = get_random_fallback(fallback_pool, n=20)
        response_time_ms = (time.time() - start_time) * 1000
        telemetry.log('recommendation', {
            'user_id': raw_user_id,
            'recommended_items': recommendation_response,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'model_version': current_provenance.get('model_version'),
            'data_version': current_provenance.get('data_version'),
            'git_commit': current_provenance.get('git_commit'),
            'mlflow_run_id': current_provenance.get('mlflow_run_id')
        })
        return recommendation_response
    
    # Get unrated items
    rated_items_inner = {iid for (iid, r) in current_trainset.ur[inner_uid]}
    all_items_inner = set(current_trainset.all_items())
    unrated_items_inner = all_items_inner - rated_items_inner
    
    MAX_PREDICTIONS = 200
    if len(unrated_items_inner) > MAX_PREDICTIONS:
        popular_items = {item['movie_id'] for item in current_fallback[:500]}
        popular_unrated = unrated_items_inner & {current_trainset.to_inner_iid(iid) for iid in popular_items if current_trainset.knows_item(iid)}
        
        if len(popular_unrated) >= MAX_PREDICTIONS:
            unrated_items_inner = random.sample(list(popular_unrated), MAX_PREDICTIONS)
        else:
            remaining = MAX_PREDICTIONS - len(popular_unrated)
            random_items = random.sample(list(unrated_items_inner - popular_unrated), remaining)
            unrated_items_inner = list(popular_unrated) + random_items
    
    # Create testset
    testset_for_user = [
        (raw_user_id, current_trainset.to_raw_iid(iid), 4.0) 
        for iid in unrated_items_inner
    ]
    
    if not testset_for_user:
        recommendation_response = get_random_fallback(fallback_pool, n=20)
        response_time_ms = (time.time() - start_time) * 1000
        telemetry.log('recommendation', {
            'user_id': raw_user_id,
            'recommended_items': recommendation_response,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'model_version': current_provenance.get('model_version'),
            'data_version': current_provenance.get('data_version'),
            'git_commit': current_provenance.get('git_commit'),
            'mlflow_run_id': current_provenance.get('mlflow_run_id')
        })
        return recommendation_response
    
    svd_predictions = current_svd.test(testset_for_user)
    
    # Get top 20
    recommendation_response = get_top_n(svd_predictions, n=20)
    response_time_ms = (time.time() - start_time) * 1000

    # Log telemetry
    elapsed = time.time() - start_time
    logging.info(f"Generated recommendations in {elapsed*1000:.0f}ms")
    

    telemetry.log('recommendation', {
        'user_id': raw_user_id,
        'recommended_items': recommendation_response,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model_version': current_provenance.get('model_version'),
        'data_version': current_provenance.get('data_version'),
        'git_commit': current_provenance.get('git_commit'),
        'mlflow_run_id': current_provenance.get('mlflow_run_id')
    })
    
    return recommendation_response


@app.route('/provenance', methods=['GET'])
def get_recommendation_provenance():
    user_id = request.args.get('user_id', type=str)
    dt = request.args.get('datetime', type=str)
    recs = request.args.get('recommendations', type=str)
    n = int(request.args.get('n', 1))

    if not user_id or not dt:
        return jsonify({'error': 'user_id and datetime are required.'}), 400
    
    try:
        query = '''
            SELECT * FROM recommendations
            WHERE user_id = ? AND timestamp <= ?
        '''
        params = [user_id, dt]
        if recs:
            query += ' AND recommended_items = ?'
            params.append(recs)
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(n)

        conn = sqlite3.connect(DATABASE)
        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return jsonify({'error': 'No matching recommendations found.'}), 404
        
        results = df.to_dict(orient='records')
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def start_file_watcher():
    """
    Start watching the models directory for changes.
    """
    event_handler = ModelFileHandler()
    observer = Observer()
    observer.schedule(event_handler, MODEL_DIR, recursive=False)
    observer.start()
    logging.info(f"Started file watcher on {MODEL_DIR}")
    return observer


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load all models and data before starting the server
    logging.info("Starting server, loading resources...")
    load_resources()
    
    # Start file watcher for automatic reloading
    observer = start_file_watcher()
    
    try:
        # Run the app
        port = int(os.getenv('PORT', 5000))
        logging.info(f"Server ready on port {port}")
        app.run(debug=False, port=port, host='0.0.0.0')
    finally:
        observer.stop()
        observer.join()
