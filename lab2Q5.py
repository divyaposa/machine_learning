"""
A5. Similarity Measure:
Take the first 2 observation vectors from the dataset. Consider only the attributes 
(direct or derived) with binary values for these vectors (ignore other attributes). 

Tasks:
    • Calculate the Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) 
      between the document vectors.
    • Use first vector for each document for this.
    • Compare the values for JC and SMC and judge the appropriateness of each.

Formulas:
    JC = (f11) / (f01 + f10 + f11)
    SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

Where:
    f11 = number of attributes where value is 1 in both vectors.
    f00 = number of attributes where value is 0 in both vectors.
    f10 = attributes where first vector is 1 and second is 0.
    f01 = attributes where first vector is 0 and second is 1.
"""

import pandas as pd


class BinarySimilarity:
    """Class to compute similarity measures for binary attributes."""

    def __init__(self, file_path: str, sheet_name: str):
        """
        Initialize dataset path and sheet name.

        Args:
            file_path (str): Path to Excel file.
            sheet_name (str): Worksheet name containing the data.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None
        self.binary_columns = []

    def load_and_prepare_data(self):
        """Load dataset and fill missing categorical values with mode."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        # Fill missing values with most frequent value for each column
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        print("✅ Data loaded and missing values handled.")

    def identify_binary_columns(self):
        """Identify attributes that have only binary values (0 or 1)."""
        self.binary_columns = [
            col for col in self.df.columns
            if self.df[col].dropna().value_counts().index.isin([0, 1]).all()
        ]
        print(f"🗂️ Binary attributes identified: {self.binary_columns}")
        return self.binary_columns

    def get_first_two_vectors(self):
        """Select the first two observation vectors using binary attributes."""
        if not self.binary_columns:
            raise ValueError("Binary columns not identified. Run identify_binary_columns() first.")

        vector_1 = self.df.loc[0, self.binary_columns].astype(int)
        vector_2 = self.df.loc[1, self.binary_columns].astype(int)
        return vector_1, vector_2

    def calculate_similarity_measures(self, vector_1, vector_2):
        """Compute f11, f00, f10, f01, JC, and SMC."""
        f11 = ((vector_1 == 1) & (vector_2 == 1)).sum()
        f00 = ((vector_1 == 0) & (vector_2 == 0)).sum()
        f10 = ((vector_1 == 1) & (vector_2 == 0)).sum()
        f01 = ((vector_1 == 0) & (vector_2 == 1)).sum()

        # Calculate Jaccard Coefficient
        jaccard_coeff = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0

        # Calculate Simple Matching Coefficient
        smc_coeff = (f11 + f00) / (f00 + f01 + f10 + f11)

        print(f"\n🔢 Similarity Components:\n f11 = {f11}, f00 = {f00}, f10 = {f10}, f01 = {f01}")
        print(f"📊 Jaccard Coefficient = {jaccard_coeff:.4f}")
        print(f"📊 Simple Matching Coefficient = {smc_coeff:.4f}")

        return jaccard_coeff, smc_coeff

    def interpret_results(self, jaccard_coeff, smc_coeff):
        """Interpret the comparison between JC and SMC."""
        print("\n💡 Interpretation:")
        if abs(jaccard_coeff - smc_coeff) > 0.1:
            print("Jaccard Coefficient focuses on shared presence (1s), "
                  "better when 1s are more meaningful.")
        else:
            print("Both JC and SMC are similar, but JC is preferred for sparse binary data.")


# ------------------ Main Program Execution ------------------
if __name__ == "__main__":
    # Create object
    similarity_checker = BinarySimilarity(file_path="Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    # Step 1: Load and prepare data
    similarity_checker.load_and_prepare_data()

    # Step 2: Identify binary columns
    similarity_checker.identify_binary_columns()

    # Step 3: Get first two vectors
    vec1, vec2 = similarity_checker.get_first_two_vectors()

    # Step 4: Calculate similarity measures
    jc, smc = similarity_checker.calculate_similarity_measures(vec1, vec2)

    # Step 5: Interpret results
    similarity_checker.interpret_results(jc, smc)
