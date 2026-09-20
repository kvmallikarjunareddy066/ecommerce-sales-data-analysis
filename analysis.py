"""
ApexPlanet Data Analytics Internship
Task 1: Foundational Setup & Exploratory Data Analysis

Run:
    python scripts/analysis.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_sales_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_ecommerce_sales.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_PATH)

# Clean data
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

df["Customer_Name"] = df["Customer_Name"].fillna("Unknown Customer")
df["City"] = df["City"].fillna("Unknown City")
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median()).round().astype(int)
df["Sales"] = df["Sales"].fillna(df["Sales"].median())
df["Profit"] = df["Profit"].fillna(df["Profit"].median())

before_duplicates = len(df)
df = df.drop_duplicates().reset_index(drop=True)
df.to_csv(PROCESSED_PATH, index=False)

print("Cleaned dataset saved to:", PROCESSED_PATH)
print("Rows after cleaning:", len(df))
print("Duplicates removed:", before_duplicates - len(df))
print("\nStatistical summary:")
print(df[["Quantity", "Sales", "Profit"]].describe())

# Create charts
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
df["Category"].value_counts().plot(kind="bar")
plt.title("Number of Orders by Category")
plt.xlabel("Category")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(REPORT_DIR / "orders_by_category.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(REPORT_DIR / "sales_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Sales"])
plt.title("Sales Boxplot")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(REPORT_DIR / "sales_boxplot.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Category")
plt.title("Sales vs Profit")
plt.tight_layout()
plt.savefig(REPORT_DIR / "sales_vs_profit.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 6))
numeric_cols = ["Quantity", "Sales", "Profit"]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
plt.show()

print("\nTop category by sales:")
print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False).head(1))

print("\nTop city by sales:")
print(df.groupby("City")["Sales"].sum().sort_values(ascending=False).head(1))
