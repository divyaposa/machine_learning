"""
A1. Please refer to the “Purchase Data” worksheet of Lab Session Data.xlsx.
Please load the data and segregate them into 2 matrices A & C (following the nomenclature of AX = C).
Do the following activities:
    • What is the dimensionality of the vector space for this data?
    • How many vectors exist in this vector space?
    • What is the rank of Matrix A?
    • Using Pseudo-Inverse, find the cost of each product available for sale.
      (Suggestion: If you use Python, you can use numpy.linalg.pinv() function to get a pseudo-inverse.)
"""

import pandas as pd
import numpy as np

class PurchaseAnalysis:
    """Class to perform purchase data analysis based on given requirements."""

    def __init__(self, file_path: str, sheet_name: str):
        """
        Initialize with Excel file path and sheet name.

        Args:
            file_path (str): Path to the Excel file.
            sheet_name (str): Name of the worksheet to read.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.purchase_df = None
        self.product_matrix = None
        self.payment_vector = None

    def load_data(self):
        """Load purchase data from the Excel sheet."""
        self.purchase_df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        print("Data loaded successfully. Preview:")
        print(self.purchase_df.head())  # Display first few rows for verification

    def create_matrices(self):
        """
        Create:
        - Matrix A: Product quantities (Candies, Mangoes, Milk Packets)
        - Vector C: Payment amounts
        """
        self.product_matrix = self.purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
        self.payment_vector = self.purchase_df['Payment (Rs)'].values

    def analyze_vector_space(self):
        """
        Calculate:
        - Dimensionality of the vector space
        - Number of vectors
        - Rank of Matrix A
        """
        vector_space_dim = np.linalg.matrix_rank(self.product_matrix)  # Rank is also dimensionality here
        num_vectors = self.product_matrix.shape[0]  # Number of rows in Matrix A

        print(f"Dimensionality of the vector space: {vector_space_dim}")
        print(f"Number of vectors in the vector space: {num_vectors}")
        print(f"Rank of Matrix A: {vector_space_dim}")

        return vector_space_dim, num_vectors

    def calculate_cost_per_product(self):
        """
        Use pseudo-inverse to solve AX = C for X,
        where X is the cost vector for the products.
        """
        product_matrix_pinv = np.linalg.pinv(self.product_matrix)  # Pseudo-inverse of A
        cost_vector = np.dot(product_matrix_pinv, self.payment_vector)  # Solve for X

        print("\nCost of each product (in Rs):")
        print(f"Candies: {cost_vector[0]:.2f} Rs")
        print(f"Mangoes: {cost_vector[1]:.2f} Rs")
        print(f"Milk Packets: {cost_vector[2]:.2f} Rs")

        return cost_vector

# ------------------ Main Program Execution ------------------

if __name__ == "__main__":
    # Create an instance of PurchaseAnalysis with the file path and sheet name
    analysis = PurchaseAnalysis(file_path="Lab Session Data.xlsx", sheet_name="Purchase data")

    # Step 1: Load the purchase data
    analysis.load_data()

    # Step 2: Create Matrices A and C
    analysis.create_matrices()

    # Step 3: Analyze vector space properties
    analysis.analyze_vector_space()

    # Step 4: Calculate the cost of each product
    analysis.calculate_cost_per_product()
