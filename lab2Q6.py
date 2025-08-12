"""
A6. Cosine Similarity Measure:
Take the complete vectors for the first two observations (including all attributes).
Calculate the Cosine similarity between the documents by using the second 
feature vector for each document.

Steps:
    1. Load the dataset from 'thyroid0387_UCI'.
    2. Handle missing values:
       - For categorical columns, fill with mode.
       - For numeric columns, fill with mean.
    3. Encode categorical variables using Label Encoding.
    4. Extract the first two observation vectors.
    5. Calculate cosine similarity between them.

Formula:
    CosineSimilarity(A, B) = (A ⋅ B) / (||A|| * ||B||)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityCalculator:
    """Class to calculate cosine similarity between two dataset observations."""

    def __init__(self, file_path: str, sheet_name: str):
        """
        Initialize with dataset file path and sheet name.
        Args:
            file_path (str): Path to the Excel file.
            sheet_name (str): Worksheet name containing the data.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None
        self.encoded_df = None

    def load_and_clean_data(self):
        """Load dataset and handle missing values."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

        # Fill missing values based on column type
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            else:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
        print("✅ Data loaded and missing values handled.")

    def encode_categorical(self):
        """Apply label encoding to categorical variables."""
        self.encoded_df = self.df.copy()
        for column in self.encoded_df.columns:
            if self.encoded_df[column].dtype == 'object':
                label_encoder = LabelEncoder()
                self.encoded_df[column] = label_encoder.fit_transform(
                    self.encoded_df[column].astype(str)
                )
        print("🔢 Categorical attributes encoded.")

    def get_observation_vectors(self, index1: int, index2: int):
        """
        Retrieve two observation vectors by index.
        Args:
            index1 (int): Row index of first observation.
            index2 (int): Row index of second observation.
        Returns:
            tuple: Two reshaped vectors ready for similarity calculation.
        """
        vector_1 = self.encoded_df.iloc[index1].values.reshape(1, -1)
        vector_2 = self.encoded_df.iloc[index2].values.reshape(1, -1)
        return vector_1, vector_2

    def compute_cosine_similarity(self, vector_1, vector_2):
        """
        Calculate cosine similarity between two vectors.
        Args:
            vector_1: First vector (2D shape).
            vector_2: Second vector (2D shape).
        Returns:
            float: Cosine similarity score.
        """
        similarity_score = cosine_similarity(vector_1, vector_2)[0][0]
        print(f"📊 Cosine Similarity: {similarity_score:.4f}")
        return similarity_score


# ------------------ Main Program Execution ------------------
if __name__ == "__main__":
    # Initialize the calculator
    similarity_calculator = CosineSimilarityCalculator(
        file_path="Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )

    # Step 1: Load and clean data
    similarity_calculator.load_and_clean_data()

    # Step 2: Encode categorical attributes
    similarity_calculator.encode_categorical()

    # Step 3: Get first two observations
    obs1, obs2 = similarity_calculator.get_observation_vectors(0, 1)

    # Step 4: Compute cosine similarity
    similarity_calculator.compute_cosine_similarity(obs1, obs2)
