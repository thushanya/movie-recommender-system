from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def train_model(ratings_df):
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        ratings_df[['userId', 'movieId', 'rating']],
        reader
    )

    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    print("RMSE:", accuracy.rmse(predictions))

    return model


def predict(model, user_id, movie_id):
    return model.predict(user_id, movie_id).est