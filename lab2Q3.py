import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("Lab Session Data.xlsx", sheet_name = "IRCTC Stock Price")
print("Mean of population: ",statistics.mean(df['Price']))
print("Variance of population: ",statistics.variance(df['Price']))

print("Mean of prices on wednesday: ",statistics.mean(df[df['Day']=='Wed']['Price']))
print("Variance of prices on wednesday: ",statistics.variance(df[df['Day']=='Wed']['Price']))
print("Mean & Variance of population is more than mean & variance of wednesdays")

print("Mean of prices in April: ",statistics.mean(df[df['Month']=='Apr']['Price']))
print("Variance of prices in April: ",statistics.variance(df[df['Month']=='Apr']['Price']))
print("Mean of population is less than mean in April, and Variance of population is less than variance in April")

df['Loss'] = df['Chg%'].apply(lambda x: 1 if x < 0 else 0)
print(f"Probability of making a loss: {df['Loss'].mean():.4f}")

df['LossOnWed'] = df[df['Day']=='Wed']['Chg%'].apply(lambda x: 1 if x < 0 else 0)
print(f"Probability of making a loss on wednesday: {df['LossOnWed'].mean():.4f}")

conditional_probability_profit = (df[df['Day'] == 'Wed']['Chg%'] > 0).mean()
print(f"Conditional probability profit: {conditional_probability_profit}")


order_of_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
df['Day'] = pd.Categorical(df['Day'], categories=order_of_days, ordered=True)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Day', y='Chg%', data=df)
plt.show()
