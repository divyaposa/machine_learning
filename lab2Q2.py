"""
A2. Mark all customers (in the “Purchase Data” table) with payments above Rs. 200 as RICH 
and others as POOR. Develop a classifier model to categorize customers into RICH or POOR 
based on purchase behavior.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


class CustomerClassifier:
    """Class to classify customers as RICH or POOR based on purchase data."""

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
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(random_state=42)

    def load_data(self):
        """Load purchase data from the Excel sheet."""
        self.purchase_df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        print("✅ Data loaded successfully. Preview:")
        print(self.purchase_df.head())

    def label_customers(self):
        """Label customers as RICH if payment > 200, otherwise POOR."""
        self.purchase_df['customer_label'] = self.purchase_df['Payment (Rs)'].apply(
            lambda amt: 'RICH' if amt > 200 else 'POOR'
        )

    def prepare_data(self):
        """
        Prepare features (X) and labels (y) for training.
        Returns:
            X_train, X_test, y_train, y_test: Split and scaled datasets.
        """
        features = self.purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
        labels = self.purchase_df['customer_label'].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train the Random Forest classifier."""
        self.classifier.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance and print results."""
        predicted_labels = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predicted_labels)

        print(f"\n📈 Model Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_labels))

    def predict_full_dataset(self):
        """Predict labels for the entire dataset and append to DataFrame."""
        features_scaled = self.scaler.transform(
            self.purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
        )
        self.purchase_df['predicted_label'] = self.classifier.predict(features_scaled)

    def display_results(self):
        """Display DataFrame with actual and predicted labels."""
        print("\n📊 Updated DataFrame with Predictions:")
        print(self.purchase_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 
                                'Payment (Rs)', 'customer_label', 'predicted_label']])


# ------------------ Main Program Execution ------------------
if __name__ == "__main__":
    # Create instance of classifier
    model = CustomerClassifier(file_path="Lab Session Data.xlsx", sheet_name="Purchase data")

    # Step 1: Load and label data
    model.load_data()
    model.label_customers()

    # Step 2: Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test = model.prepare_data()

    # Step 3: Train the model
    model.train_model(X_train_scaled, y_train)

    # Step 4: Evaluate the model
    model.evaluate_model(X_test_scaled, y_test)

    # Step 5: Predict for the full dataset
    model.predict_full_dataset()

    # Step 6: Show final DataFrame
    model.display_results()
