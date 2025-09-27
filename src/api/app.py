from flask import Flask
from models.ubcf.model import UBCFRecommender

app = Flask(__name__)


@app.route('/recommend/<userid>', methods=['POST'])
def recommend(userid):
    
    print(f"Received recommendation request for user ID: {userid}")
    recommended = UBCFRecommender().recommend(user_id=int(userid)).index.tolist()
    return ','.join(map(str, recommended))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)