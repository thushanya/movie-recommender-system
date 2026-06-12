import numpy as np

def hybrid_recommendations(user_id, movies, ratings, model, cosine_sim, top_n=10):
    
    # Average movie popularity baseline
    movie_scores = []

    for idx, row in movies.iterrows():
        movie_id = row['movieId']

        # Collaborative Filtering score
        cf_score = model.predict(user_id, movie_id).est

        # Content-Based score (average similarity)
        cb_score = np.mean(cosine_sim[idx])

        # Hybrid score
        score = (0.6 * cf_score) + (0.4 * cb_score)

        movie_scores.append((row['title'], row['genres'], score))

    # Sort by score
    movie_scores.sort(key=lambda x: x[2], reverse=True)

    return movie_scores[:top_n]