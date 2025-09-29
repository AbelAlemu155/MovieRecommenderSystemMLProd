import csv
import random
import json
from datetime import datetime, timedelta
from faker import Faker
import os

fake = Faker()
random.seed(42)
Faker.seed(42)

# --- Config ---
N_USERS = 500
N_MOVIES = 5000
N_HISTORY = 10000

# --- User Data ---
user_rows = []
for uid in range(1, N_USERS + 1):
    age = random.randint(15, 70)
    occupation = fake.job()
    gender = random.choice(["male", "female", "other"])
    user_rows.append([uid, age, occupation, gender])

fake_data_path = "src/models/deep_rec_sys/fake_data/"

os.makedirs(fake_data_path, exist_ok=True)
with open(os.path.join(fake_data_path, "user_data.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "age", "occupation", "gender"])
    writer.writerows(user_rows)

# --- Movie Data ---
GENRES = [
    "Action","Adventure","Animation","Comedy","Crime","Documentary",
    "Drama","Family","Fantasy","History","Horror","Music","Mystery",
    "Romance","Science Fiction","TV Movie","Thriller","War","Western"
]
STATUSES = ["Released", "Post Production", "Planned", "Rumored"]
LANGS = ["en","es","fr","de","it","pt","ru","ja","ko","zh","hi","ar","sw","rw","lg"]

movie_rows = []
base_date = datetime(1970, 1, 1)
for mid in range(1, N_MOVIES + 1):
    tmdb_id = random.randint(10000, 9999999)
    imdb_id = f"tt{random.randint(1000000, 99999999)}"
    adult = random.choice([True, False])
    genres = random.sample(GENRES, random.randint(1, 3))
    overview = fake.paragraph(nb_sentences=3)
    original_language = random.choice(LANGS)
    popularity = round(random.uniform(0, 200), 3)
    release_date = (base_date + timedelta(days=random.randint(0, 20000))).strftime("%Y-%m-%d")
    revenue = random.randint(0, 500_000_000)
    budget = random.randint(0, 200_000_000)
    runtime = random.randint(60, 180)
    spoken_languages = random.sample(LANGS, random.randint(1, 2))
    belongs_to_collection = json.dumps([f"{fake.word().title()} Collection"]) if random.random() < 0.3 else "[]"
    vote_average = round(random.uniform(0, 10), 1)
    vote_count = random.randint(0, 50000)
    video = random.choice([True, False])
    title = fake.sentence(nb_words=random.randint(2, 5)).rstrip(".")
    production_countries = [fake.country_code() for _ in range(random.randint(1, 2))]
    status = random.choice(STATUSES)
    tagline = fake.sentence(nb_words=random.randint(3, 7)).rstrip(".")

    movie_rows.append([
        mid, tmdb_id, imdb_id, adult,
        json.dumps(genres), overview, original_language, popularity, release_date,
        revenue, budget, runtime, json.dumps(spoken_languages),
        belongs_to_collection, vote_average, vote_count, video, title,
        json.dumps(production_countries), status, tagline
    ])


with open(os.path.join(fake_data_path, "movie_data.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "movie_id","tmdb_id","imdb_id","adult","genres","overview",
        "original_language","popularity","release_date","revenue","budget","runtime",
        "spoken_languages","belongs_to_collection","vote_average","vote_count",
        "video","title","production_countries","status","tagline"
    ])
    writer.writerows(movie_rows)

# --- Watch History Data ---
watch_rows = []
for _ in range(N_HISTORY):
    uid = random.randint(1, N_USERS)
    mid = random.randint(1, N_MOVIES)
    start_time = fake.date_time_between(start_date="-2y", end_date="now")
    duration = timedelta(minutes=random.randint(10, 180))
    last_watch = start_time + duration
    watch_rows.append([uid, mid, start_time.isoformat(), last_watch.isoformat()])


with open(os.path.join(fake_data_path, "watch_history_data.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "movie_id", "start_time", "last_watch"])
    writer.writerows(watch_rows)

# --- Rating Data ---
rating_rows = []
for _ in range(N_HISTORY):
    uid = random.randint(1, N_USERS)
    mid = random.randint(1, N_MOVIES)
    rating = round(random.uniform(1, 5), 1)
    rating_rows.append([uid, mid, rating])


with open(os.path.join(fake_data_path, "rating_data.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "movie_id", "rating"])
    writer.writerows(rating_rows)

print("Fake CSVs generated: user_data.csv, movie_data.csv, watch_history_data.csv, rating_data.csv")
