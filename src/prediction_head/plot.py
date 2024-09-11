import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


def plot_regression_distribution(y, title: str, xlabel: str, ylabel: str):
    plt.hist(y, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_classification_distribution(x, y, title: str, xlabel: str, ylabel: str):
    # Performs PCA before plotting the data
    pca = PCA(n_components=2)
    prinicpalComponents = pca.fit_transform(x)
    pca_explained_variance = pca.explained_variance_ratio_
    pca_df: pd.DataFrame = pd.DataFrame(
        data=prinicpalComponents, columns=["PC1", "PC2"]
    )
    final_df: pd.DataFrame = pd.concat(
        [pca_df, pd.DataFrame(y, columns=["target"])], axis=1
    )
    scatter = plt.scatter(final_df["PC1"], final_df["PC2"], c=final_df["target"])
    plt.title(title)
    plt.xlabel(xlabel + " (" + str(pca_explained_variance[0] * 100) + "%)")
    plt.ylabel(ylabel + " (" + str(pca_explained_variance[1] * 100) + "%)")
    plt.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    plt.show()


def plot_ground_truth_vs_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, xlabel: str, ylabel: str
):
    # convert y_true and y_pred array to pandas dataframe
    df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
    # plot
    sns.jointplot(data=df, x="y_true", y="y_pred", kind="reg")
    # plot red line that is perfect prediction
    plt.plot(
        [df["y_true"].min(), df["y_true"].max()],
        [df["y_true"].min(), df["y_true"].max()],
        "r",
    )
    plt.suptitle(title, y=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
