"""
A7. Heatmap Plot:
Consider the first 20 observation vectors from the 'thyroid0387_UCI' dataset.
Calculate:
    - Jaccard Coefficient (JC)
    - Simple Matching Coefficient (SMC)
    - Cosine Similarity (COS)
between all pairs of these 20 vectors.

Visualization:
    - Create heatmaps for JC, SMC (binary features only), and COS (all features).
    - Use seaborn heatmaps for visual comparison.

Reference:
    - JC formula: JC = f11 / (f11 + f10 + f01)
    - SMC formula: SMC = (f11 + f00) / (f00 + f01 + f10 + f11)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityHeatmap:
    """Class to compute JC, SMC, and COS for dataset vectors and plot heatmaps."""

    def __init__(self, file_path: str, sheet_name: str):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None
        self.encoded_df = None
        self.sample_df = None

    def load_and_clean_data(self):
        """Load dataset and fill missing values."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            else:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
        print("✅ Data loaded and missing values handled.")

    def encode_categorical(self):
        """Label encode all categorical variables."""
        self.encoded_df = self.df.copy()
        for column in self.encoded_df.columns:
            if self.encoded_df[column].dtype == 'object':
                label_encoder = LabelEncoder()
                self.encoded_df[column] = label_encoder.fit_transform(
                    self.encoded_df[column].astype(str)
                )
        print("🔢 Categorical attributes encoded.")

    def select_sample(self, n: int = 20):
        """Select first n rows for similarity analysis."""
        self.sample_df = self.encoded_df.iloc[:n].reset_index(drop=True)
        print(f"📌 Selected first {n} rows for analysis.")

    @staticmethod
    def calculate_jc_smc(vector1, vector2):
        """Calculate Jaccard and SMC for two binary vectors."""
        f11 = np.sum((vector1 == 1) & (vector2 == 1))
        f00 = np.sum((vector1 == 0) & (vector2 == 0))
        f10 = np.sum((vector1 == 1) & (vector2 == 0))
        f01 = np.sum((vector1 == 0) & (vector2 == 1))

        jc_score = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
        smc_score = (f11 + f00) / (f00 + f01 + f10 + f11)
        return jc_score, smc_score

    def compute_similarity_matrices(self):
        """Compute JC, SMC, and COS matrices for the sample."""
        n = len(self.sample_df)

        # Binary feature filtering
        binary_columns = [
            col for col in self.sample_df.columns
            if self.sample_df[col].isin([0, 1]).all()
        ]

        jaccard_matrix = np.zeros((n, n))
        smc_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                v1 = self.sample_df.loc[i, binary_columns].astype(int).values
                v2 = self.sample_df.loc[j, binary_columns].astype(int).values
                jc, smc = self.calculate_jc_smc(v1, v2)
                jaccard_matrix[i, j] = jc
                smc_matrix[i, j] = smc

        # Cosine similarity for all features
        cosine_matrix = cosine_similarity(self.sample_df)

        return jaccard_matrix, smc_matrix, cosine_matrix

    @staticmethod
    def plot_heatmap(matrix, title, cmap):
        """Plot a seaborn heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=False, cmap=cmap)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        """Run the complete JC, SMC, and COS analysis with heatmaps."""
        jc_matrix, smc_matrix, cos_matrix = self.compute_similarity_matrices()

        # Plot all heatmaps
        self.plot_heatmap(cos_matrix, "🔵 Cosine Similarity Heatmap (First 20 Observations)", "coolwarm")
        self.plot_heatmap(jc_matrix, "🟢 Jaccard Coefficient Heatmap (Binary Features)", "viridis")
        self.plot_heatmap(smc_matrix, "🟡 Simple Matching Coefficient Heatmap (Binary Features)", "YlGnBu")


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    sim_heatmap = SimilarityHeatmap(
        file_path="Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )
    sim_heatmap.load_and_clean_data()
    sim_heatmap.encode_categorical()
    sim_heatmap.select_sample(20)
    sim_heatmap.run_analysis()
