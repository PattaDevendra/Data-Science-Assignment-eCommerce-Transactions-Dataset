# Data-Science-Assignment-eCommerce-Transactions-Dataset
import pandas as pd

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

print(customers.head())
print(products.head())
print(transactions.head())
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

customers.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)


customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(products['Price'], kde=True)
plt.title('Product Price Distribution')
plt.show()

print(customers.describe())
print(products.describe())
print(transactions.describe())
region_sales = transactions.groupby(customers['Region']).agg({'TotalValue': 'sum'}).reset_index()
sns.barplot(x='Region', y='TotalValue', data=region_sales)
plt.title('Sales by Region')
plt.show()
transactions['TransactionMonth'] = transactions['TransactionDate'].dt.to_period('M')
monthly_transactions = transactions.groupby('TransactionMonth').size()
monthly_transactions.plot(kind='line', figsize=(10, 6))
plt.title('Monthly Transactions')
plt.ylabel('Number of Transactions')
plt.show()
