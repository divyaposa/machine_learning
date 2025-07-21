import pandas as pd
import numpy as np

# Load the purchase data
purchase_df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

# Preview the data
print(purchase_df.head())

# Create Matrix A (features: candies, mangoes, milk)
product_matrix = purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values

# Create Vector C (target: payment)
payment_vector = purchase_df['Payment (Rs)'].values

# Rank and dimension analysis
vector_space_dim = np.linalg.matrix_rank(product_matrix)
num_vectors = product_matrix.shape[0]

print(f"The dimensionality of the vector space is: {vector_space_dim}")
print(f"The number of vectors in the vector space is: {num_vectors}")
print(f"The rank of Matrix A is: {vector_space_dim}")

# Calculate pseudo-inverse and solve for cost vector
product_matrix_pinv = np.linalg.pinv(product_matrix)
cost_vector = np.dot(product_matrix_pinv, payment_vector)

# Output cost of each product
print("Cost of each product (Candies, Mangoes, Milk Packets) in Rs:")
print(f"Candies: {cost_vector[0]:.2f} Rs")
print(f"Mangoes: {cost_vector[1]:.2f} Rs")
print(f"Milk Packets: {cost_vector[2]:.2f} Rs")
