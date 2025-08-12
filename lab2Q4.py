"""
A4. Data Exploration:
Load the data from “thyroid0387_UCI” worksheet and perform the following:
    • Study each attribute and its values, and identify the datatype (nominal, ordinal, numeric, etc.).
    • For categorical attributes, identify the encoding scheme to be employed.
      (Hint: Label encoding for ordinal variables, One-Hot encoding for nominal variables)
    • Study the data range for numeric variables.
    • Identify missing values in each attribute.
    • Detect outliers in the data.
    • For numeric variables, calculate the mean and variance (or standard deviation).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class ThyroidDataExplorer:
    """Class to perform data exploration on the thyroid0387_UCI dataset."""

    def __init__(self, file_path: str, sheet_name: str):
        """
        Initialize with Excel file path and sheet name.
        Args:
            file_path (str): Path to Excel file.
            sheet_name (str): Worksheet name to load.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None

    def load_data(self):
        """Load the dataset."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        print("✅ Data loaded successfully. Preview:")
        print(self.df.head())

    def check_data_types(self):
        """Display data types for each attribute."""
        print("\n📊 Data Types:")
        print(self.df.dtypes)

    def check_missing_values(self):
        """Check and display missing values for each attribute."""
        print("\n❓ Missing Values:")
        print(self.df.isnull().sum())

    def detect_categorical_columns(self):
        """Identify categorical attributes and recommend encoding schemes."""
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        print("\n🗂️ Categorical Attributes:", categorical_columns)

        print("\n💡 Encoding Recommendation:")
        print("- Use Label Encoding for ordinal attributes (e.g., severity levels)")
        print("- Use One-Hot Encoding for nominal attributes (e.g., gender, class)")
        return categorical_columns

    def encode_categorical(self, categorical_columns):
        """Apply label encoding as placeholder for categorical attributes."""
        label_encoded_df = self.df.copy()
        for column in categorical_columns:
            label_encoder = LabelEncoder()
            try:
                label_encoded_df[column] = label_encoder.fit_transform(self.df[column].astype(str))
            except Exception as e:
                print(f"⚠️ Skipping encoding for {column}: {e}")
        return label_encoded_df

    def analyze_numeric_columns(self):
        """Calculate statistics for numeric attributes."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print("\n📈 Numeric Summary:")
        for column in numeric_columns:
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            var_val = self.df[column].var()
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            print(f"{column}: Mean = {mean_val:.2f}, Std = {std_val:.2f}, Var = {var_val:.2f}, Range = {min_val} to {max_val}")
        return numeric_columns

    def detect_outliers_boxplot(self, numeric_columns):
        """Visualize outliers using boxplots for numeric variables."""
        excluded_columns = ['Record ID', 'record_id', 'id']
        plot_numeric_columns = [col for col in numeric_columns if col not in excluded_columns]

        for col in plot_numeric_columns:
            plt.figure(figsize=(6, 4))
            self.df[col].plot.box()
            plt.title(f"📦 Boxplot of {col}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def detect_outliers_age_filtered(self):
        """Plot a filtered boxplot for age to better visualize outliers."""
        if 'age' in self.df.columns:
            plt.figure(figsize=(6, 4))
            self.df[self.df['age'] < 200]['age'].plot.box()
            plt.title("📦 Boxplot of Age (Filtered for age < 200)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


# ------------------ Main Program Execution ------------------
if __name__ == "__main__":
    explorer = ThyroidDataExplorer(file_path="Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    # Step 1: Load Data
    explorer.load_data()

    # Step 2: Data Type Check
    explorer.check_data_types()

    # Step 3: Missing Values
    explorer.check_missing_values()

    # Step 4: Categorical Data & Encoding
    categorical_cols = explorer.detect_categorical_columns()
    label_encoded_df = explorer.encode_categorical(categorical_cols)

    # Step 5: Numeric Data Analysis
    numeric_cols = explorer.analyze_numeric_columns()

    # Step 6: Outlier Detection
    explorer.detect_outliers_boxplot(numeric_cols)
    explorer.detect_outliers_age_filtered()
