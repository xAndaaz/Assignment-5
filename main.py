import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats


df_train = pd.read_csv('Dataset/train.csv')
print(df_train.head())

#Exploratory Data Analysis
print("Skewness and Kurtosis of SalePrice")
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# Plotting the distribution of SalePrice
plt.figure(figsize=(10, 6))
sns.distplot(df_train['SalePrice'], fit=norm, color='blue', label=f"Skewness: {(df_train['SalePrice'].skew()):.2f}\n Kurtosis: {(df_train['SalePrice'].kurt()):.2f}")
plt.title('Distribution of SalePrice (Original)')
plt.show()

# Plotting the probability plot
fig = plt.figure(figsize=(10, 6))
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.title('Probability Plot of SalePrice (Original)')
plt.show()

# log transformation to handle right skewness
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

# transformed SalePrice
plt.figure(figsize=(10, 6))
sns.distplot(df_train['SalePrice'], fit=norm, color='green')
plt.title('Distribution of SalePrice (Log Transformed)')
plt.show()

# probability plot of the transformed SalePrice
fig = plt.figure(figsize=(10, 6))
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.title('Probability Plot of SalePrice (Log Transformed)')
plt.show()


# Feature Engineering and Data Cleaning
# Ccombined square footage of different areas into a single feature
df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']

# Combining all bathroom 
df_train['TotalBath'] = df_train['FullBath'] + (0.5 * df_train['HalfBath']) + df_train['BsmtFullBath'] + (0.5 * df_train['BsmtHalfBath'])

# Combining all porch 
df_train['TotalPorchSF'] = df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + df_train['3SsnPorch'] + df_train['ScreenPorch']


# handling missing values 
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    df_train[col] = df_train[col].fillna('None')

# for numerical features fill missing values with 0, as this is the most logical value in these cases
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    df_train[col] = df_train[col].fillna(0)

# For LotFrontage, we'll fill missing values with the median of the neighborhood
df_train['LotFrontage'] = df_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


#one hot encoding for categorical features
df_train = pd.get_dummies(df_train)
print("\ntransformed data")
print(df_train.head())


#  feature selection

# correlation analysis

corrmat = df_train.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap='viridis')
plt.title('Overall Correlation Matrix')
plt.show()

# Let's zoom in on the top 10 most correlated features.
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap='plasma')
plt.title('Top 10 Correlated Features with SalePrice')
plt.show()


# prediction model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X = df_train.drop(['SalePrice', 'Id'], axis=1)
y = df_train['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)

# evalution on validation set
y_pred = ridge.predict(X_val)

#rmse error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"\nModel Performance-------")
print(f"Validation RMSE: {rmse:.4f}")

# predictions vs actual values plot
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.75, color='indigo')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--', color='red', linewidth=2)
plt.xlabel('Actual SalePrice (Log)')
plt.ylabel('Predicted SalePrice (Log)')
plt.title('Actual vs. Predicted SalePrice on Validation Set')
plt.show()


