import pandas as pd

# ---------- Ratings ----------
ratings_cols = ["userId", "movieId", "rating", "timestamp"]

ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=ratings_cols
)

ratings.to_csv("data/ratings.csv", index=False)


# ---------- Movies ----------
movie_columns = [
    "movieId",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    names=movie_columns
)

genre_columns = movie_columns[5:]

movies["genres"] = movies[genre_columns].apply(
    lambda row: " ".join(
        [genre for genre in genre_columns if row[genre] == 1]
    ),
    axis=1
)

movies = movies[["movieId", "title", "genres"]]

movies.to_csv("data/movies.csv", index=False)

print("movies.csv and ratings.csv updated successfully!")