#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("DrugSalesData.csv", sep=';', low_memory=False)
print("FIRST 5 RECORDS")
print(df.head())
print("\nLAST 5 RECORDS")
print(df.tail())
print("\nSHAPE OF DATASET")
print(df.shape)
print("\nCOLUMN NAMES")
print(df.columns)
print("\nDATASET INFORMATION")
df.info()
print("\nSTATISTICAL SUMMARY")
print(df.describe(include='all'))
print("\nDATATYPES")
print(df.dtypes)
print("\nMISSING VALUES")
print(df.isnull().sum())
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())
print("\nTOTAL NULL VALUES")
print(df.isnull().sum().sum())
print("\nUNIQUE VALUES")
print(df.nunique())
print("\nMEMORY USAGE")
print(df.memory_usage(deep=True))


# In[13]:


print("Before Cleaning :", df.shape)

df = df.drop_duplicates()
df = df.dropna(how='all')
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()
df.replace('', np.nan, inplace=True)
print("\nMissing Values")
print(df.isnull().sum())
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['int64','float64']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

print("\nAfter Cleaning")
print(df.isnull().sum())

print("\nDataset Shape")
print(df.shape)

print("\nFirst Five Rows")
print(df.head())


# In[21]:


num_cols = df.select_dtypes(include=['int64','float64']).columns
print("Numeric Columns")
print(num_cols)
for col in num_cols:
    print(f"\nPositive values from {col}")
    print(df[df[col] > 0].head())
filtered = df.dropna()
print("\nFiltered Dataset Shape")
print(filtered.shape)
print(filtered.head())


# In[20]:


print("\n========== GROUPING ==========\n")

object_cols = df.select_dtypes(include='object').columns

if len(object_cols) > 0:
    group_col = object_cols[0]
    print("Grouping based on:", group_col)

    grouped = df.groupby(group_col).size().reset_index(name='Count')
    print(grouped)

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    if len(numeric_cols) > 0:
        grouped_mean = df.groupby(group_col)[numeric_cols].mean()
        print("\nMean Values")
        print(grouped_mean)

        grouped_sum = df.groupby(group_col)[numeric_cols].sum()
        print("\nSum Values")
        print(grouped_sum)

        grouped_max = df.groupby(group_col)[numeric_cols].max()
        print("\nMaximum Values")
        print(grouped_max)

        grouped_min = df.groupby(group_col)[numeric_cols].min()
        print("\nMinimum Values")
        print(grouped_min)
else:
    print("No Object Columns Found")
print("\n========== SORTING ==========\n")

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_cols:

    print(f"\nAscending Order : {col}")
    print(df.sort_values(by=col).head())

    print(f"\nDescending Order : {col}")
    print(df.sort_values(by=col, ascending=False).head())
print("\n========== AGGREGATION ==========\n")

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_cols:

    print("\n--------------------------------")
    print("Column :", col)

    print("Minimum :", df[col].min())

    print("Maximum :", df[col].max())

    print("Mean :", df[col].mean())

    print("Median :", df[col].median())

    print("Mode :")
    print(df[col].mode())

    print("Variance :", df[col].var())

    print("Standard Deviation :", df[col].std())

    print("Count :", df[col].count())

    print("Sum :", df[col].sum())


# In[16]:


print("\n========== QUERY 1 ==========")
print("Total Records :", len(df))

print("\n========== QUERY 2 ==========")
print("Total Columns :", len(df.columns))

print("\n========== QUERY 3 ==========")
print(df['Drug Name'].value_counts().head(10))

print("\n========== QUERY 4 ==========")
print(df['Drug Group'].value_counts())

print("\n========== QUERY 5 ==========")
print(df['Pharmacy'].value_counts().head(10))

print("\n========== QUERY 6 ==========")
print(df['Season'].value_counts())

print("\n========== QUERY 7 ==========")
print(df['Company Name'].nunique())

print("\n========== QUERY 8 ==========")
print(df['DoctorNameSurname'].nunique())

print("\n========== QUERY 9 ==========")
print(df['Patient name'].nunique())

print("\n========== QUERY 10 ==========")
print(df['Reporting'].value_counts())

print("\n========== QUERY 11 ==========")
print(df['Drug Name'].mode())

print("\n========== QUERY 12 ==========")
print(df['Pharmacy'].mode())

print("\n========== QUERY 13 ==========")
print(df.groupby('Season').size())

print("\n========== QUERY 14 ==========")
print(df.groupby('Drug Group').size().sort_values(ascending=False).head(10))

print("\n========== QUERY 15 ==========")
print(df.groupby('Pharmacy').size().sort_values(ascending=False).head(10))

print("\n========== QUERY 16 ==========")
print(df.groupby('Company Name').size().sort_values(ascending=False).head(10))

print("\n========== QUERY 17 ==========")
print(df.groupby('DoctorNameSurname').size().sort_values(ascending=False).head(10))

print("\n========== QUERY 18 ==========")
print(df['RecipeDate'].value_counts().head(10))

print("\n========== QUERY 19 ==========")
print(df['Receipt Date'].value_counts().head(10))

print("\n========== QUERY 20 ==========")
print(df['Drug Name'].value_counts().tail(10))


# In[17]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
df['Drug Group'].value_counts().head(10).plot(kind='bar')
plt.title("Top Drug Groups")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10,5))
df['Pharmacy'].value_counts().head(10).plot(kind='bar')
plt.title("Top Pharmacies")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(6,6))
df['Season'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title("Season Distribution")
plt.ylabel("")
plt.show()
plt.figure(figsize=(6,4))
df['Reporting'].value_counts().plot(kind='bar')
plt.title("Reporting")
plt.show()
plt.figure(figsize=(10,5))
df['Company Name'].value_counts().head(10).plot(kind='bar')
plt.title("Top Companies")
plt.show()
plt.figure(figsize=(10,5))
df['DoctorNameSurname'].value_counts().head(10).plot(kind='bar')
plt.title("Top Doctors")
plt.show()
plt.figure(figsize=(10,5))
df['Drug Name'].value_counts().head(10).plot(kind='bar')
plt.title("Top Drugs")
plt.show()
plt.figure(figsize=(10,5))
df['Patient name'].value_counts().head(10).plot(kind='bar')
plt.title("Top Patients")
plt.show()
df['Company Name'].plot(kind='hist',bins=20)
plt.title("Company Distribution")
plt.show()
plt.figure(figsize=(5,5))
plt.boxplot(df['Company Name'])
plt.title("Company Name Boxplot")
plt.show()


# In[ ]:





# In[ ]:




