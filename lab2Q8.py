"""
A8. Data Imputation:
Employ appropriate central tendencies to fill the missing values in the data variables.

Guidelines:
    • Mean → For numeric attributes with no outliers
    • Median → For numeric attributes containing outliers
    • Mode → For categorical attributes
Steps:
    1. Load the dataset.
    2. Identify numeric and categorical attributes.
    3. For numeric columns:
        - Detect outliers using the IQR method.
        - Use mean if no outliers, else median.
    4. For categorical columns:
        - Fill missing values with mode.
    5. Output the filled dataset.
"""

import pandas as pd
import numpy as np


class DataImputer:
    """Class for imputing missing values in a dataset based on attribute type and outliers."""

    def __init__(self, file_path: str, sheet_name: str):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None

    def load_data(self):
        """Load the dataset from Excel."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        print("✅ Data loaded successfully.")

    def impute_numeric(self):
        """Impute numeric columns using mean or median based on outliers."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        for column in numeric_columns:
            if self.df[column].isnull().sum() == 0:
                continue  # Skip if no missing values

            # Outlier detection using IQR
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            has_outliers = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).any()

            if has_outliers:
                imputed_value = self.df[column].median()
                strategy = "median"
            else:
                imputed_value = self.df[column].mean()
                strategy = "mean"

            self.df[column].fillna(imputed_value, inplace=True)
            print(f"{column}: Filled missing with {strategy} = {imputed_value}")

    def impute_categorical(self):
        """Impute categorical columns using mode."""
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()

        for column in categorical_columns:
            if self.df[column].isnull().sum() == 0:
                continue

            mode_value = self.df[column].mode()[0]
            self.df[column].fillna(mode_value, inplace=True)
            print(f"{column}: Filled missing with mode = {mode_value}")

    def run_imputation(self):
        """Run the full imputation process."""
        self.impute_numeric()
        self.impute_categorical()
        print("\n📊 Missing values after imputation:\n", self.df.isnull().sum())


# ------------------ Main Program ------------------
if __name__ == "__main__":
    imputer = DataImputer(
        file_path="Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )
    imputer.load_data()
    imputer.run_imputation()
