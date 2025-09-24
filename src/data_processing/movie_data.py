from dotenv import load_dotenv
import os
import requests

class MovieDataProcessor:
    
    def __init__(self, movie_id):
        load_dotenv()
        self.movie_id = movie_id
        self.movie_url = f"{os.getenv('MOVIE_SERVICE_URL')}/{movie_id}"
    
    def _format_genres(self, genres):
        if isinstance(genres, list):
            return ",".join([str(genre["id"]) for genre in genres])
        return genres
    
    def _format_spoken_languages(self, spoken_languages):
        if isinstance(spoken_languages, list):
            return ",".join([str(lang["iso_639_1"]) for lang in spoken_languages])
        return spoken_languages
    
    def _format_production_companies(self, production_companies):
        if isinstance(production_companies, list):
            return ",".join([str(comp["id"]) for comp in production_companies])
        return production_companies
    
    def _format_production_countries(self, production_countries):
        if isinstance(production_countries, list):
            return ",".join([str(country["iso_3166_1"]) for country in production_countries])
        return production_countries
    
    def _format_origin_countries(self, origin_countries):
        if isinstance(origin_countries, list):
            return ",".join([str(country) for country in origin_countries])
        return origin_countries

    def fetch_movie_details(self):
        # Simulate fetching movie details from an external API
        
        keys = [
            'tmdb_id', 'imdb_id', "adult", "genres", "overview", "original_language", "popularity", "production_companies",
            "release_date", "revenue", "runtime", "spoken_languages", "spoken_languages", "vote_average", "vote_count"
        ]
        response = requests.get(self.movie_url)
        response.raise_for_status()
        movie_details = response.json()
        movie_details['genres'] = self._format_genres(movie_details.get('genres', ''))
        movie_details['spoken_languages'] = self._format_spoken_languages(movie_details.get('spoken_languages', ''))
        movie_details['production_companies'] = self._format_production_companies(movie_details.get('production_companies', ''))
        return {key: movie_details.get(key) for key in keys}
    
    def fetch_movie_details_imdb(self):
        # Simulate fetching movie details from IMDb
        imdb_details = {
            'movie_id': self.movie_id,
            'imdb_rating': 7.5,
            'imdb_votes': 15000
        }
        return imdb_details

    def fetch_movie_details_tmdb(self, tmdb_id):

        load_dotenv()
        
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=en-US"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {os.getenv('TMBD_READ_ACCESS_TOKEN')}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            tmdb_details = response.json()
            tmdb_details['genres'] = self._format_genres(tmdb_details.get('genres', ''))
            tmdb_details['spoken_languages'] = self._format_spoken_languages(tmdb_details.get('spoken_languages', ''))
            tmdb_details['production_companies'] = self._format_production_companies(tmdb_details.get('production_companies', ''))  
            tmdb_details['production_countries'] = self._format_production_countries(tmdb_details.get('production_countries', ''))
            tmdb_details['origin_country'] = self._format_origin_countries(tmdb_details.get('origin_country', ''))
        else:
            print(f"Failed to fetch TMDB details: {response.status_code}")
            print(response.text)
            tmdb_details = {}
        return tmdb_details
    
    
if __name__ == "__main__":
    
    processor = MovieDataProcessor(movie_id='about+a+boy+2002')
    
    details = processor.fetch_movie_details()
    tmdb_details = processor.fetch_movie_details_tmdb(details['tmdb_id'])

    print(details)
    
    print("----- TMDB Details -----")
    
    
    print(tmdb_details)