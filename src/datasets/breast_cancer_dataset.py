import numpy as np
from sklearn import preprocessing
from ucimlrepo import fetch_ucirepo

from src.constants import BATCH_SIZE
from src.datasets.dataset import MlpDataset, cache_to_file


@cache_to_file(name="breast_cancer")
def fetch_breast_cancer_dataset():
    breast_cancer = fetch_ucirepo(id=14)
    df = breast_cancer.data.features.copy(deep=True)
    y_df = breast_cancer.data.targets.copy(deep=True)

    # Account for NaN values
    indexes_with_NaN = df[df.isnull().any(axis=1)].index
    df = df.drop(indexes_with_NaN)
    y_df = y_df.drop(indexes_with_NaN)

    # Fix the dataset entry errors. 'tumor-size' and 'inv-nodes' columns have misformatted values.
    # Where `5-9` is expected, there is `9-May`, a date format.

    month_abbreviations = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    for col in ["tumor-size", "inv-nodes"]:
        for month_abbr, month_num in month_abbreviations.items():
            # Use str.endswith to avoid partial matches
            mask = df[col].str.endswith(month_abbr)
            # df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(month_abbr, str(month_num))
            df.loc[mask, col] = (
                df.loc[mask, col]
                .astype(str)
                .apply(
                    lambda s: s.split("-")[1].replace(month_abbr, str(month_num))
                    + "-"
                    + s.split("-")[0]
                )
            )

    breast_cancer_X_df = df.copy(deep=True)

    # Convert categorical features to one-hot encoded ones
    categorical_features = [
        "menopause",
        "node-caps",
        "breast",
        "breast-quad",
        "irradiat",
    ]
    oh = preprocessing.OneHotEncoder(sparse_output=False, drop="first")
    new_features = oh.fit_transform(breast_cancer_X_df[categorical_features])
    new_features_names = oh.get_feature_names_out()

    breast_cancer_X_df[new_features_names] = new_features
    breast_cancer_X_df.drop(columns=categorical_features, inplace=True)

    # Convert ordinal categorical features to floats

    # Converts `10-20` to (10 + 20) / 2 = 15
    range_to_average_func = (
        lambda s: float(int(s.split("-")[0]) + int(s.split("-")[0])) / 2
    )

    breast_cancer_X_df["age"] = breast_cancer_X_df["age"].apply(range_to_average_func)
    breast_cancer_X_df["tumor-size"] = breast_cancer_X_df["tumor-size"].apply(
        range_to_average_func
    )
    breast_cancer_X_df["inv-nodes"] = breast_cancer_X_df["inv-nodes"].apply(
        range_to_average_func
    )

    # Rescale data to <-1, 1>
    breast_cancer_X = (
        preprocessing.RobustScaler()
        .fit(breast_cancer_X_df)
        .transform(breast_cancer_X_df)
    )
    breast_cancer_X = preprocessing.normalize(breast_cancer_X)

    # Convert target
    breast_cancer_y_df = y_df.copy(deep=True)
    oh = preprocessing.OneHotEncoder(sparse_output=False)
    new_target = oh.fit_transform(breast_cancer_y_df[["Class"]])
    new_target_names = oh.get_feature_names_out()

    breast_cancer_y_df[new_target_names] = new_target
    breast_cancer_y_df.drop(columns=["Class"], inplace=True)

    return breast_cancer_X, breast_cancer_y_df.values


breast_cancer_X, breast_cancer_y_df = fetch_breast_cancer_dataset()


# https://archive.ics.uci.edu/dataset/14/breast+cancer
class BreastCancerDataset(MlpDataset):
    input_size: int = len(breast_cancer_X[0])
    output_size: int = breast_cancer_y_df.shape[1]

    @classmethod
    def get_dataloaders(cls, batch_size=BATCH_SIZE):
        X, y = breast_cancer_X, np.array(breast_cancer_y_df)
        return cls.get_dataloaders_from_xy(X, y, batch_size)
