import streamlit as st
import pandas as pd

from src.utils import load_data
from src.content_based import build_similarity, recommend
from src.collaborative import train_model
from src.hybrid import hybrid_recommendations


# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: #ffffff;
}

.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}

.stSelectbox {
    border-radius: 10px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)


# -----------------------
# Cached Data Loading
# -----------------------
@st.cache_data
def load_resources():
    movies, ratings = load_data()

    cosine_sim = build_similarity(movies)

    indices = pd.Series(
        movies.index,
        index=movies['title']
    ).drop_duplicates()

    return movies, ratings, cosine_sim, indices


# -----------------------
# Cached Model Loading (FIX)
# -----------------------
@st.cache_resource
def load_model(ratings):
    return train_model(ratings)


movies, ratings, cosine_sim, indices = load_resources()

model = load_model(ratings)


# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("ℹ️ About")

st.sidebar.info(
    """
    Movie Recommendation System

    Features:
    - Content-Based Filtering
    - Collaborative Filtering (SVD)
    - Hybrid Recommendation System

    Dataset:
    - MovieLens 100K

    Built with:
    - Python
    - Streamlit
    - Scikit-learn
    - Surprise
    """
)


# -----------------------
# Main Title
# -----------------------
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourites using Machine Learning."
)


# -----------------------
# Movie Selection
# -----------------------
selected_movie = st.selectbox(
    "Search or select a movie:",
    sorted(movies['title'].tolist())
)


# -----------------------
# Content-Based Recommendations
# -----------------------
if st.button("Get Recommendations"):

    with st.spinner("Finding similar movies..."):

        recommendations = recommend(
            selected_movie,
            movies,
            cosine_sim,
            indices
        )

    st.subheader(f"Because you liked: {selected_movie}")

    cols = st.columns(2)

    for idx, (_, row) in enumerate(recommendations.iterrows()):

        with cols[idx % 2]:
            st.markdown(f"""
            ### 🎥 {row['title']}

            **Genres:** {row['genres']}
            """)
            st.divider()


# -----------------------
# Top Rated Movies
# -----------------------
st.header("⭐ Most Popular Movies")

movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

top_movies = movie_stats[
    movie_stats['rating_count'] >= 50
].merge(
    movies,
    on='movieId'
).sort_values(
    by='avg_rating',
    ascending=False
).head(10)

st.dataframe(
    top_movies[
        ['title', 'avg_rating', 'rating_count']
    ].rename(
        columns={
            'title': 'Movie',
            'avg_rating': 'Average Rating',
            'rating_count': 'Number of Ratings'
        }
    ),
    use_container_width=True
)


# -----------------------
# Trending Movies
# -----------------------
st.header("🔥 Trending Movies")

def get_trending_movies(ratings, movies):

    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        count=('rating', 'count')
    ).reset_index()

    trending = stats[stats['count'] >= 50]

    trending = trending.merge(movies, on='movieId')

    trending = trending.sort_values(
        by=['avg_rating', 'count'],
        ascending=False
    ).head(10)

    return trending


trending_movies = get_trending_movies(ratings, movies)

st.dataframe(
    trending_movies[['title', 'avg_rating', 'count']].rename(
        columns={
            'title': 'Movie',
            'avg_rating': 'Rating',
            'count': 'Votes'
        }
    ),
    use_container_width=True
)


# -----------------------
# Hybrid Recommendations
# -----------------------
st.header("🤝 Hybrid Recommendations (Best of Both Models)")

user_id = st.number_input("Enter User ID", min_value=1, value=1)

if st.button("Get Hybrid Recommendations"):

    results = hybrid_recommendations(
        user_id,
        movies,
        ratings,
        model,
        cosine_sim
    )

    cols = st.columns(2)

    for i, (title, genres, score) in enumerate(results):

        with cols[i % 2]:

            st.markdown(f"""
            ### 🎬 {title}

            **Genres:** {genres}

            **Score:** `{round(score, 2)}`
            """)

            st.divider()