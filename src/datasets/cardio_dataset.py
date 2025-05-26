import numpy as np
from sklearn import preprocessing
from ucimlrepo import fetch_ucirepo

from src.datasets.dataset import MlpDataset, cache_dataset_to_file


@cache_dataset_to_file(name="cardio")
def fetch_cardio_dataset():
    cardio = fetch_ucirepo(id=193)

    # Prepare fetures
    cardio_X_df = cardio.data.features.astype("float32")

    cardio_scalable_values = ["AC", "FM", "UC", "DL", "DS", "DP"]
    scaler = preprocessing.RobustScaler()
    cardio_X_df[cardio_scalable_values] = scaler.fit_transform(
        cardio_X_df[cardio_scalable_values]
    )

    # cardio_X = preprocessing.normalize(cardio_df_X.values)
    cardio_X = cardio_X_df.values

    # Prepare targets
    cardio_y_df = cardio.data.targets
    cardio_class_y = cardio_y_df["CLASS"].astype("category").cat.codes.values

    return cardio_X, cardio_class_y


cardio_X, cardio_y = fetch_cardio_dataset()


class CardioDataset(MlpDataset):
    input_size: int = len(cardio_X[0])
    output_size: int = len(np.unique(cardio_y))

    @classmethod
    def get_xy(cls) -> tuple[np.ndarray, np.ndarray]:
        return cardio_X, cardio_y

    @classmethod
    def get_dataloaders(cls, batch_size: int | None = None):
        X, y = cls.get_xy()
        return cls.get_dataloaders_from_xy(X, y, batch_size)
