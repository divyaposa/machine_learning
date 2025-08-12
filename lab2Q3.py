"""
A3. Please refer to the data present in the “IRCTC Stock Price” data sheet of the Excel file.
Perform the following:
    • Calculate the mean and variance of the Price data (column D).
    • Select the price data for all Wednesdays and calculate the sample mean. Compare with the population mean.
    • Select the price data for the month of April and calculate the sample mean. Compare with the population mean.
    • From Chg% (column I) find the probability of making a loss.
    • Calculate the probability of making a profit on Wednesday.
    • Calculate the conditional probability of making a profit, given that today is Wednesday.
    • Make a scatter plot of Chg% against the day of the week.
"""

import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


class IRCTCStockAnalysis:
    """Class to perform statistical analysis on IRCTC stock price data."""

    def __init__(self, file_path: str, sheet_name: str):
        """
        Initialize with Excel file path and sheet name.

        Args:
            file_path (str): Path to the Excel file.
            sheet_name (str): Name of the worksheet to read.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None

    def load_data(self):
        """Load stock price data from the Excel sheet."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        print("✅ Data loaded successfully. Preview:")
        print(self.df.head())

    def calculate_population_stats(self):
        """Calculate mean and variance of the population (Price column)."""
        mean_price = statistics.mean(self.df['Price'])
        variance_price = statistics.variance(self.df['Price'])

        print(f"📊 Mean of population: {mean_price:.2f}")
        print(f"📊 Variance of population: {variance_price:.2f}")
        return mean_price, variance_price

    def analyze_wednesday_prices(self, population_mean, population_variance):
        """Analyze Wednesday price statistics and compare with population."""
        wed_prices = self.df[self.df['Day'] == 'Wed']['Price']
        mean_wed = statistics.mean(wed_prices)
        variance_wed = statistics.variance(wed_prices)

        print(f"📅 Mean of prices on Wednesday: {mean_wed:.2f}")
        print(f"📅 Variance of prices on Wednesday: {variance_wed:.2f}")

        if mean_wed > population_mean:
            print("Observation: Wednesday mean is higher than population mean.")
        else:
            print("Observation: Wednesday mean is lower than population mean.")

        if variance_wed > population_variance:
            print("Observation: Wednesday variance is higher than population variance.")
        else:
            print("Observation: Wednesday variance is lower than population variance.")

    def analyze_april_prices(self, population_mean, population_variance):
        """Analyze April price statistics and compare with population."""
        april_prices = self.df[self.df['Month'] == 'Apr']['Price']
        mean_april = statistics.mean(april_prices)
        variance_april = statistics.variance(april_prices)

        print(f"📅 Mean of prices in April: {mean_april:.2f}")
        print(f"📅 Variance of prices in April: {variance_april:.2f}")

        if mean_april > population_mean:
            print("Observation: April mean is higher than population mean.")
        else:
            print("Observation: April mean is lower than population mean.")

        if variance_april > population_variance:
            print("Observation: April variance is higher than population variance.")
        else:
            print("Observation: April variance is lower than population variance.")

    def probability_of_loss(self):
        """Calculate probability of making a loss (Chg% < 0)."""
        self.df['Loss'] = self.df['Chg%'].apply(lambda x: 1 if x < 0 else 0)
        prob_loss = self.df['Loss'].mean()
        print(f"📉 Probability of making a loss: {prob_loss:.4f}")

    def probability_profit_wednesday(self):
        """Calculate probability of making a profit on Wednesday."""
        wed_data = self.df[self.df['Day'] == 'Wed']
        prob_profit_wed = (wed_data['Chg%'] > 0).mean()
        print(f"📈 Probability of making a profit on Wednesday: {prob_profit_wed:.4f}")

    def conditional_probability_profit_given_wed(self):
        """Calculate conditional probability of profit given Wednesday."""
        wed_data = self.df[self.df['Day'] == 'Wed']
        conditional_prob = (wed_data['Chg%'] > 0).mean()
        print(f"📊 Conditional probability of profit given Wednesday: {conditional_prob:.4f}")

    def plot_chg_vs_day(self):
        """Create scatter plot of Chg% vs Day of the week."""
        order_of_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        self.df['Day'] = pd.Categorical(self.df['Day'], categories=order_of_days, ordered=True)

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='Day', y='Chg%', data=self.df)
        plt.title("Scatter Plot of Chg% vs Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Change %")
        plt.show()


# ------------------ Main Program Execution ------------------
if __name__ == "__main__":
    # Create analysis object
    analysis = IRCTCStockAnalysis(file_path="Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

    # Step 1: Load data
    analysis.load_data()

    # Step 2: Population statistics
    pop_mean, pop_var = analysis.calculate_population_stats()

    # Step 3: Wednesday analysis
    analysis.analyze_wednesday_prices(pop_mean, pop_var)

    # Step 4: April analysis
    analysis.analyze_april_prices(pop_mean, pop_var)

    # Step 5: Probability calculations
    analysis.probability_of_loss()
    analysis.probability_profit_wednesday()
    analysis.conditional_probability_profit_given_wed()

    # Step 6: Plot results
    analysis.plot_chg_vs_day()
