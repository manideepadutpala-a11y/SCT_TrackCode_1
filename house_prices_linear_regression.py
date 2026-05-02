import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

BASIC_FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
ADVANCED_NUMERIC_FEATURES = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "FullBath",
    "TotalBathrooms",
    "TotalPorchSF",
    "TotalSF",
    "Age",
    "RemodelAge",
    "HasPool",
    "CentralAir",
]
ADVANCED_CATEGORICALS = ["MSZoning", "HouseStyle", "SaleCondition"]


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def create_advanced_features(df):
    df = df.copy()
    df["TotalBathrooms"] = df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
    df["TotalPorchSF"] = (
        df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )
    df["TotalSF"] = df["GrLivArea"].fillna(0) + df["TotalBsmtSF"].fillna(0)
    df["Age"] = df["YrSold"].fillna(df["YearBuilt"]) - df["YearBuilt"].fillna(df["YrSold"])
    df["RemodelAge"] = df["YrSold"].fillna(df["YearRemodAdd"]) - df["YearRemodAdd"].fillna(df["YrSold"])
    df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    df["CentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0}).fillna(0)

    for col in [
        "OverallQual",
        "OverallCond",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "GarageArea",
        "FullBath",
        "YearBuilt",
        "YearRemodAdd",
        "LotArea",
        "YrSold",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ["MSZoning", "HouseStyle", "SaleCondition"]:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    encoded = pd.get_dummies(df[ADVANCED_CATEGORICALS], drop_first=True)
    df = pd.concat([df, encoded], axis=1)
    return df


def prepare_features(df, feature_set="basic"):
    df = df.copy()
    if feature_set == "basic":
        df["TotalBathrooms"] = df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
        X = df[BASIC_FEATURES + ["TotalBathrooms"]].copy()
        return X.fillna(X.median())

    df = create_advanced_features(df)
    X = df[ADVANCED_NUMERIC_FEATURES + list(pd.get_dummies(df[ADVANCED_CATEGORICALS], drop_first=True).columns)]
    return X.fillna(X.median())


def prepare_target(df):
    if "SalePrice" not in df.columns:
        raise KeyError("Target column SalePrice is missing from the dataset.")
    return df["SalePrice"].values


def build_model(model_type="linear"):
    if model_type == "linear":
        return LinearRegression()
    if model_type == "gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=500, random_state=42)
    raise ValueError(f"Unknown model type: {model_type}")


def align_columns(train_df, test_df):
    train_columns = train_df.columns
    test_df = test_df.reindex(columns=train_columns, fill_value=0)
    return test_df


def train_and_evaluate(train_path, test_path=None, output_path=None, model_type="linear", feature_set="basic", save_model=None):
    train_df = load_dataset(train_path)
    X_train = prepare_features(train_df, feature_set=feature_set)
    y_train = prepare_target(train_df)

    model = build_model(model_type)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    rmse_scores = -cv_scores

    print("Training results")
    print("----------------")
    print(f"Model type: {model_type}")
    print(f"Feature set: {feature_set}")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"5-fold CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

    if hasattr(model, "coef_"):
        coefficients = dict(zip(X_train.columns, model.coef_))
        print("Coefficients:")
        for feature, coef in coefficients.items():
            print(f"  {feature}: {coef:.4f}")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        top_importances = importances.sort_values(ascending=False).head(20)
        print("Top feature importances:")
        for feature, importance in top_importances.items():
            print(f"  {feature}: {importance:.4f}")

    if save_model:
        joblib.dump(model, save_model)
        print(f"Saved trained model to {save_model}")

    if test_path:
        test_df = load_dataset(test_path)
        if "Id" not in test_df.columns:
            raise KeyError("Test dataset must contain an Id column for submission output.")
        X_test = prepare_features(test_df, feature_set=feature_set)
        if X_test.shape[1] != X_train.shape[1] or any(X_test.columns != X_train.columns):
            X_test = align_columns(X_train, X_test)
        predictions = model.predict(X_test)
        submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": np.maximum(predictions, 0)})
        if output_path:
            submission.to_csv(output_path, index=False)
            print(f"Saved predictions to {output_path}")
        else:
            print(submission.head())

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a house price regression model.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=False, help="Path to test.csv (optional)")
    parser.add_argument("--output", default="submission.csv", help="Path for output predictions")
    parser.add_argument(
        "--model",
        choices=["linear", "gradient_boosting"],
        default="linear",
        help="Model type to train.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["basic", "advanced"],
        default="basic",
        help="Feature set to use for training.",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional path to save the trained model as a .pkl file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        args.train,
        args.test,
        args.output if args.test else None,
        model_type=args.model,
        feature_set=args.feature_set,
        save_model=args.save_model,
    )
