from flask import Flask
from models.ubcf.model import UBCFRecommender
from src.models.deep_rec_sys.deep_rec_model import DeepRecSysModel


app = Flask(__name__)


@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    
    print(f"Received recommendation request for user ID: {userid}")
    # ucbf_model = UBCFRecommender()
    # recommended = ucbf_model.recommend(user_id=int(userid)).index.tolist()

    deep_rec_sys = DeepRecSysModel(base_path="src/models/deep_rec_sys")
    recommended = deep_rec_sys.recommend(user_id=int(userid))
    return deep_rec_sys.format(recommended)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)