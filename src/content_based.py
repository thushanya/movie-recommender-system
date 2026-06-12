from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity(movies_df):

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def recommend(movie_title, movies_df, cosine_sim, indices):

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies_df.iloc[movie_indices][['title', 'genres']]