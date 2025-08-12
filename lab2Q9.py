"""
A9. Data Normalization / Scaling:
From the data study, identify the attributes which may need normalization.
Employ appropriate normalization techniques to create normalized set of data.

Steps:
    1. Identify numeric attributes that require normalization.
    2. Apply:
        - Min-Max Scaling
        - Z-score Standardization
    3. Output:
        - Range of each attribute before scaling
        - Scaled datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataNormalizer:
    """Class to perform Min-Max scaling and Z-score standardization."""

    def __init__(self, file_path: str, sheet_name: str):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None
        self.numeric_columns = []

    def load_data(self):
        """Load dataset and fill missing numeric values temporarily for scaling."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        # Fill missing numeric values with mean for scaling purposes
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print("✅ Data loaded and missing numeric values handled.")

    def show_ranges(self):
        """Display range of each numeric column before scaling."""
        print("\n📊 Column Ranges Before Scaling:")
        for column in self.numeric_columns:
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            value_range = max_val - min_val
            print(f"{column}: Min = {min_val}, Max = {max_val}, Range = {value_range}")

    def apply_minmax_scaling(self):
        """Apply Min-Max Scaling to numeric columns."""
        scaler = MinMaxScaler()
        scaled_df = self.df.copy()
        scaled_df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])
        print("\n✅ Min-Max Scaling Applied.")
        return scaled_df

    def apply_zscore_standardization(self):
        """Apply Z-score Standardization to numeric columns."""
        scaler = StandardScaler()
        scaled_df = self.df.copy()
        scaled_df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])
        print("✅ Z-score Standardization Applied.")
        return scaled_df

    def run_normalization(self):
        """Run normalization workflow and display samples."""
        self.show_ranges()
        minmax_df = self.apply_minmax_scaling()
        zscore_df = self.apply_zscore_standardization()

        print("\n📈 Sample - Min-Max Scaled Data:\n", minmax_df[self.numeric_columns].head())
        print("\n📉 Sample - Z-score Standardized Data:\n", zscore_df[self.numeric_columns].head())


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    normalizer = DataNormalizer(
        file_path="Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )
    normalizer.load_data()
    normalizer.run_normalization()
