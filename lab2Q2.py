import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the purchase data
purchase_df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

# Add label: RICH if payment > 200, else POOR
purchase_df['customer_label'] = purchase_df['Payment (Rs)'].apply(lambda amt: 'RICH' if amt > 200 else 'POOR')

# Select features and target
features = purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
labels = purchase_df['customer_label'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)

# Make predictions
predicted_labels = classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Predict labels for full dataset
purchase_df['predicted_label'] = classifier.predict(scaler.transform(purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]))

# Show final result
print("\n📊 Updated DataFrame with Predictions:")
print(purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'customer_label', 'predicted_label']])
