import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

print(df.head())

print("\nЛипсващи стойности по колони:\n", df.isnull().sum())

print("\nСтатистика:\n", df.describe())

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["Close"], label="Цена при затваряне (Close)", color='blue')
plt.title("Цена на акцията във времето")
plt.xlabel("Дата")
plt.ylabel("Цена (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["Volume"], label="Обем на търговията", color='orange')
plt.title("Обем на търговията (Volume) във времето")
plt.xlabel("Дата")
plt.ylabel("Обем")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Корелация между показателите")
plt.tight_layout()
plt.show()
