import sqlite3
import random
import json
from datetime import datetime, timedelta
from faker import Faker
from dotenv import load_dotenv
import os

load_dotenv()

fake = Faker()

# Initialize database
def init_database():
    conn = sqlite3.connect(os.getenv("DB_PATH", "test_movies.db"))
    cursor = conn.cursor()

    # Speed up bulk inserts for test data
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=OFF;")

    # Create movies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tmdb_id INTEGER UNIQUE,
        imdb_id TEXT UNIQUE,
        adult BOOLEAN,
        genres TEXT,
        overview TEXT,
        original_language TEXT,
        popularity REAL,
        production_companies TEXT,
        release_date TEXT,
        revenue INTEGER,
        budget INTEGER,
        runtime INTEGER,
        spoken_languages TEXT,
        belongs_to_collection TEXT,
        vote_average REAL,
        vote_count INTEGER,
        video BOOLEAN,
        title TEXT,
        production_countries TEXT,
        status TEXT,
        tagline TEXT
    )
    ''')
    conn.commit()
    return conn

# Genre data
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western"
]

# Language codes with weights (more common languages have higher weights)
LANGUAGES = [
    ("en", 0.4), ("es", 0.1), ("fr", 0.08), ("de", 0.06), ("it", 0.05),
    ("ja", 0.05), ("ko", 0.04), ("zh", 0.04), ("ru", 0.03), ("pt", 0.03),
    ("hi", 0.02), ("ar", 0.02), ("sv", 0.02), ("no", 0.01), ("da", 0.01),
    ("fi", 0.01), ("nl", 0.01), ("pl", 0.01), ("tr", 0.01), ("th", 0.01)
]

# Production companies
PRODUCTION_COMPANIES = [
    "Warner Bros.", "Universal Pictures", "20th Century Studios", "Sony Pictures",
    "Paramount Pictures", "Disney", "Netflix", "Amazon Studios", "A24",
    "Focus Features", "Lionsgate", "MGM", "STX Entertainment", "Blumhouse Productions",
    "Legendary Entertainment", "Working Title Films", "Plan B Entertainment",
    "Searchlight Pictures", "New Line Cinema", "Columbia Pictures"
]

# Countries
COUNTRIES = [
    "United States", "United Kingdom", "France", "Germany", "Italy", "Spain",
    "Japan", "South Korea", "China", "Canada", "Australia", "Brazil",
    "India", "Russia", "Mexico", "Argentina", "Sweden", "Norway", "Denmark"
]

# Movie status options
STATUS_OPTIONS = ["Released", "Post Production", "In Production", "Planned", "Canceled"]

def weighted_choice(choices):
    """Select from weighted choices"""
    total = sum(weight for choice, weight in choices)
    r = random.uniform(0, total)
    upto = 0
    for choice, weight in choices:
        if upto + weight >= r:
            return choice
        upto += weight
    return choices[-1][0]

def generate_movie_title():
    """Generate a realistic movie title"""
    title_patterns = [
        lambda: f"The {fake.word().title()}",
        lambda: f"{fake.word().title()} {fake.word().title()}",
        lambda: f"{fake.first_name()}'s {fake.word().title()}",
        lambda: f"The {fake.word().title()} of {fake.word().title()}",
        lambda: f"{fake.word().title()}: {fake.word().title()}",
        lambda: f"{fake.word().title()} {random.randint(2, 5)}",
        lambda: fake.catch_phrase(),
        lambda: f"{fake.word().title()} {fake.word().title()}: The Movie"
    ]
    return random.choice(title_patterns)()

def generate_overview():
    """Generate a movie overview/plot summary"""
    verbs = ["fight", "escape", "discover", "protect", "find", "save", "destroy", "overcome", "reveal", "stop"]
    nouns = ["truth", "mystery", "secret", "treasure", "enemy", "world", "family", "love", "past", "future"]
    templates = [
        f"A {fake.job().lower()} must {random.choice(verbs)} to save {random.choice(nouns)}.",
        f"When {fake.first_name()} discovers a {random.choice(nouns)}, everything changes.",
        f"In a world where {random.choice(nouns)} is forbidden, one person dares to {random.choice(verbs)}.",
        f"A group of {fake.word()}s embark on a journey to {random.choice(verbs)} the {random.choice(nouns)}.",
        f"After a tragic {random.choice(nouns)}, {fake.first_name()} must learn to {random.choice(verbs)} again.",
        f"Based on the true story of {fake.first_name()} who {random.choice(verbs)}ed against all odds.",
        f"A thrilling tale of {random.choice(nouns)}, {random.choice(nouns)}, and {random.choice(nouns)}.",
        f"When the {random.choice(nouns)} threatens humanity, only {fake.first_name()} can stop it."
    ]
    return random.choice(templates)

def generate_tagline():
    """Generate a movie tagline"""
    verbs = ["experience", "believe", "witness", "discover", "feel", "see", "live", "fight", "survive", "win"]
    year = fake.date_object().year  # returns datetime.date
    taglines = [
        "The adventure begins here.",
        "Nothing will ever be the same.",
        "Some secrets are worth dying for.",
        "The truth will set you free.",
        "Prepare for the unexpected.",
        "Every hero has a beginning.",
        "The journey of a lifetime.",
        "Love conquers all.",
        "Justice will be served.",
        "The final countdown begins.",
        f"This {year}, {random.choice(verbs)} like never before.",
        "Based on an incredible true story.",
        "The most anticipated film of the year.",
        "You won't believe your eyes."
    ]
    return random.choice(taglines)

def generate_unique_ids(seen_tmdb: set, seen_imdb: set):
    """Generate unique tmdb_id and imdb_id."""
    # TMDB: 5–6 digits
    while True:
        tmdb_id = random.randint(10000, 999999)
        if tmdb_id not in seen_tmdb:
            seen_tmdb.add(tmdb_id)
            break
    # IMDb: "tt" + 7–8 digits (wider range lowers collision odds)
    while True:
        imdb_id = f"tt{random.randint(10_000_000, 999_999_999)}"
        if imdb_id not in seen_imdb:
            seen_imdb.add(imdb_id)
            break
    return tmdb_id, imdb_id

def generate_fake_movie(seen_tmdb: set, seen_imdb: set):
    """Generate a single fake movie record with unique IDs"""
    tmdb_id, imdb_id = generate_unique_ids(seen_tmdb, seen_imdb)

    # Movie details
    adult = random.choice([True, False]) if random.random() < 0.1 else False
    title = generate_movie_title()

    # Genres
    num_genres = random.choices([1, 2, 3, 4], weights=[30, 40, 25, 5])[0]
    movie_genres = random.sample(GENRES, num_genres)
    genres_json = json.dumps([{"id": i+1, "name": genre} for i, genre in enumerate(movie_genres)])

    # Language and countries
    original_language = weighted_choice(LANGUAGES)

    # Spoken languages
    spoken_langs = [original_language]
    if random.random() < 0.3:
        additional_langs = random.sample([lang for lang, _ in LANGUAGES if lang != original_language],
                                         random.randint(1, 2))
        spoken_langs.extend(additional_langs)
    spoken_languages_json = json.dumps([{"iso_639_1": lang, "name": lang.upper()} for lang in spoken_langs])

    # Production details
    num_companies = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    prod_companies = random.sample(PRODUCTION_COMPANIES, num_companies)
    production_companies_json = json.dumps([{"id": i+1, "name": company} for i, company in enumerate(prod_companies)])

    num_countries = random.choices([1, 2, 3], weights=[70, 25, 5])[0]
    prod_countries = random.sample(COUNTRIES, num_countries)
    production_countries_json = json.dumps(
        [{"iso_3166_1": country[:2].upper(), "name": country} for country in prod_countries]
    )

    # Dates and financial data
    release_date = fake.date_between(start_date='-50y', end_date='today')

    budget_categories = [
        (0, 0.05), (1_000_000, 0.15), (10_000_000, 0.25),
        (50_000_000, 0.30), (100_000_000, 0.20), (200_000_000, 0.05)
    ]
    base_budget = random.choices([cat[0] for cat in budget_categories],
                                 weights=[cat[1] for cat in budget_categories])[0]
    budget = base_budget + random.randint(0, base_budget // 2) if base_budget > 0 else 0

    if budget > 0:
        if random.random() < 0.2:
            revenue = random.randint(0, budget // 2)
        else:
            multiplier = random.uniform(1.5, 8.0)
            revenue = int(budget * multiplier)
    else:
        revenue = random.randint(0, 50_000_000)

    runtime = max(80, min(200, int(random.normalvariate(110, 20))))
    popularity = max(0.1, min(100.0, random.expovariate(0.1)))
    vote_count = max(1, int(random.expovariate(0.001)))
    vote_average = max(1.0, min(10.0, random.normalvariate(6.5, 1.5)))

    belongs_to_collection = None
    if random.random() < 0.2:
        collection_name = f"{title.split()[0]} Collection"
        belongs_to_collection = json.dumps({
            "id": random.randint(1000, 9999),
            "name": collection_name
        })

    overview = generate_overview()
    tagline = generate_tagline() if random.random() < 0.8 else ""
    video = random.choice([True, False]) if random.random() < 0.05 else False
    status = random.choices(STATUS_OPTIONS, weights=[85, 5, 5, 3, 2])[0]

    return {
        'tmdb_id': tmdb_id,
        'imdb_id': imdb_id,
        'adult': adult,
        'genres': genres_json,
        'overview': overview,
        'original_language': original_language,
        'popularity': round(popularity, 2),
        'production_companies': production_companies_json,
        'release_date': release_date.strftime('%Y-%m-%d'),
        'revenue': revenue,
        'budget': budget,
        'runtime': runtime,
        'spoken_languages': spoken_languages_json,
        'belongs_to_collection': belongs_to_collection,
        'vote_average': round(vote_average, 1),
        'vote_count': vote_count,
        'video': video,
        'title': title,
        'production_countries': production_countries_json,
        'status': status,
        'tagline': tagline
    }

def insert_movies(conn, num_movies=30000):
    """Insert fake movies into database"""
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM movies")

    print(f"Generating {num_movies} fake movies...")

    # Track uniqueness within this run
    seen_tmdb, seen_imdb = set(), set()

    batch_size = 1000
    for batch_start in range(0, num_movies, batch_size):
        batch_end = min(batch_start + batch_size, num_movies)
        movies_batch = []

        for i in range(batch_start, batch_end):
            if i % 5000 == 0:
                print(f"Generated {i}/{num_movies} movies...")

            movie = generate_fake_movie(seen_tmdb, seen_imdb)
            movies_batch.append((
                movie['tmdb_id'], movie['imdb_id'], movie['adult'],
                movie['genres'], movie['overview'], movie['original_language'],
                movie['popularity'], movie['production_companies'],
                movie['release_date'], movie['revenue'], movie['budget'],
                movie['runtime'], movie['spoken_languages'],
                movie['belongs_to_collection'], movie['vote_average'],
                movie['vote_count'], movie['video'], movie['title'],
                movie['production_countries'], movie['status'], movie['tagline']
            ))

        cursor.executemany('''
            INSERT INTO movies (
                tmdb_id, imdb_id, adult, genres, overview, original_language,
                popularity, production_companies, release_date, revenue, budget,
                runtime, spoken_languages, belongs_to_collection, vote_average,
                vote_count, video, title, production_countries, status, tagline
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', movies_batch)

        conn.commit()

    print(f"Successfully generated and inserted {num_movies} fake movies!")

def main():
    """Main function to generate the fake movie data"""
    try:
        conn = init_database()
        insert_movies(conn, 30000)

        # Print some statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM movies")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM movies WHERE adult = 1")
        adult_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(vote_average), AVG(popularity), AVG(runtime) FROM movies")
        avg_stats = cursor.fetchone()

        print(f"\nDatabase Statistics:")
        print(f"Total movies: {total_count}")
        print(f"Adult movies: {adult_count}")
        print(f"Average rating: {avg_stats[0]:.2f}")
        print(f"Average popularity: {avg_stats[1]:.2f}")
        print(f"Average runtime: {avg_stats[2]:.1f} minutes")

        conn.close()
        print(f"\nData saved to home_alone.db")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
