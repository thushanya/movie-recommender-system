from src.utils import load_data
from src.collaborative import train_model, predict
from src.content_based import build_similarity, recommend
import pandas as pd

movies, ratings = load_data()

# Collaborative Filtering
model = train_model(ratings)
print(predict(model, 1, 50))

# Content-Based Filtering
cosine_sim = build_similarity(movies)

indices = pd.Series(movies.index, index=movies['title'])

print(recommend("Toy Story (1995)", movies, cosine_sim, indices))