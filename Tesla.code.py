#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv("C:/Users/premt/OneDrive/Desktop/Data Analytics course/Python Projects/tesla/tesla_data.csv")


# 

# In[20]:


import numpy as np

# Numeric statistics
numeric_stats = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                    'Cumulative Open', 'Price Change']].describe()

# Range values
range_values = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                   'Cumulative Open', 'Price Change']].max() - df[['Open', 'High', 'Low', 
                                                                  'Close', 'Adj Close', 'Volume', 
                                                                  'Cumulative Open', 'Price Change']].min()

# Function to calculate Median Absolute Deviation (MAD)
def mad(data):
    median = np.median(data)
    mad_value = np.median(np.abs(data - median))
    return mad_value

# MAD for 'Close' and 'Volume'
mad_close = mad(df['Close'])
mad_volume = mad(df['Volume'])


# In[13]:


print("Basic Statistics:")
print(numeric_stats)
print("\nRange Values:")
print(range_values)
print("\nMedian Absolute Deviation (MAD):")
print(f"MAD for 'Close': {mad_close:.2f}")
print(f"MAD for 'Volume': {mad_volume:.2f}")


# In[14]:


X = df[['Volume']]
y = df['Close']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)


# In[16]:


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r_squared:.2f}")


# In[17]:


plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Volume')
plt.ylabel('Close')
plt.legend()
plt.title('Linear Regression Analysis')
plt.show()


# In[18]:


rolling_mean = df['Close'].rolling(window=30).mean()
rolling_std = df['Close'].rolling(window=30).std()


# In[19]:


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
plt.plot(rolling_mean.index, rolling_mean, label='30-Day Rolling Mean',
color='orange')
plt.plot(rolling_std.index, rolling_std, label='30-Day Rolling Std Dev',
color='green')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Tesla Stock Price with Rolling Statistics')
plt.legend()
plt.grid(True)
plt.show()


# # Autocorrelation and Partial Autocorrelation Plots:
# • Plot autocorrelation and partial autocorrelation functions to identify autocorrelation patterns in the time series
# 

# In[23]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Close'], lags=40, ax=ax1)
plot_pacf(df['Close'], lags=40, ax=ax2)
ax1.set_title('Autocorrelation Plot')
ax2.set_title('Partial Autocorrelation Plot')
plt.show()


# In[25]:


# Select the relevant columns for numerical analysis
numeric_df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                 'Cumulative Open', 'Price Change']]

# Descriptive statistics
numeric_stats = numeric_df.describe()

# Range values
range_values = numeric_df.max() - numeric_df.min()

# Function to calculate Median Absolute Deviation (MAD)
def mad(data):
    median = data.median()
    mad_value = (data - median).abs().median()
    return mad_value

# MAD for 'Close' and 'Volume'
mad_close = mad(numeric_df['Close'])
mad_volume = mad(numeric_df['Volume'])


# In[26]:


print("Basic Statistics:")
print(numeric_stats)
print("\nRange Values:")
print(range_values)
print("\nMedian Absolute Deviation (MAD):")
print(f"MAD for 'Close': {mad_close:.2f}")
print(f"MAD for 'Volume': {mad_volume:.2f}")


# In[27]:


numeric_df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
'Cumulative Open', 'Price Change']]
correlation_matrix = numeric_df.corr()


# In[28]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[29]:


close_data = df['Close']


# In[30]:


Q1 = close_data.quantile(0.25)
Q3 = close_data.quantile(0.75)
IQR = Q3 - Q1


# In[31]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[32]:


outliers = close_data[(close_data < lower_bound) | (close_data > upper_bound)]
print("Outliers in 'Close' column:")
print(outliers)


# In[33]:


plt.boxplot(close_data, vert=False)
plt.title('Box Plot of Close Prices (with Outliers)')
plt.xlabel('Close Prices')
plt.show()


# In[34]:


lag = 1
df['Volume_lagged'] = df['Volume'].shift(lag)


# In[36]:


correlation = df[['Volume_lagged', 'Close']].corr().iloc[0, 1]


# In[37]:


plt.figure(figsize=(12, 6))
plt.scatter(df['Volume_lagged'], df['Close'], alpha=0.5)
plt.title(f'Lag Analysis (Lag={lag}) - Correlation: {correlation:.2f}')
plt.xlabel('Volume (Lagged)')
plt.ylabel('Close')
plt.grid(True)
plt.show()


# In[38]:


events = [
{'date': '2023-01-15', 'event': 'Product Announcement'},
{'date': '2023-05-10', 'event': 'Earnings Release'},
]


# In[39]:


df['Date'] = pd.to_datetime(df['Date'])


# In[40]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')


# In[42]:


for event in events:
    event_date = pd.to_datetime(event['date'])
    plt.axvline(event_date, color='red', linestyle='--', label=event['event'])

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Tesla Stock Price and Events')
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


sma_10 = df['Close'].rolling(window=10).mean()


# In[44]:


sma_50 = df['Close'].rolling(window=50).mean()


# In[45]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['Date'], sma_10, label='10-day SMA', linestyle='--')
plt.plot(df['Date'], sma_50, label='50-day SMA', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Tesla Stock Price with Moving Averages')
plt.legend()
plt.grid(True)
plt.show()


# In[46]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Close'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.title('Histogram of Close Price')
plt.subplot(1, 2, 2)
plt.hist(df['Volume'], bins=20, color='green', alpha=0.7)
plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.title('Histogram of Volume')
plt.tight_layout()
plt.show()


# In[47]:


close_prices = df['Close']
# Calculate daily returns
returns = close_prices.pct_change().dropna()


# In[48]:


confidence_level = 0.95
investment_amount = 100000
var = np.percentile(returns, 100 * (1 - confidence_level))
potential_loss = var * investment_amount


# In[50]:


print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence: {var:.2%}")
print(f"Potential loss for an investment of ${investment_amount:,.0f}: "
     f"${potential_loss:,.2f}")


# In[51]:


plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, alpha=0.7, color='b')
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.show()


# In[53]:


eps_df.to_csv('earnings_per_share.csv', index=False)
print("EPS data saved to earnings_per_share.csv")


# In[55]:


get_ipython().system('pip install yfinance')


# In[62]:


file_path = "C:/Users/premt/OneDrive/Desktop/Data Analytics course/Python Projects/tesla/tesla_data.csv"
print(f"File saved to: {file_path}")



# In[63]:


merged_df = pd.merge(df, eps_df, left_on='Date', right_on='date', how='inner')
merged_df['PE_ratio'] = merged_df['Close'] / merged_df['EPS']
print(merged_df[['date', 'PE_ratio']])


# In[64]:


merged_df['date'] = pd.to_datetime(merged_df['date'])


# In[66]:


plt.figure(figsize=(12, 6))
plt.plot(merged_df['date'], merged_df['PE_ratio'], marker='o', linestyle='-',
color='b')
plt.title('Tesla P/E Ratio Over Time (2022)')
plt.xlabel('Date')
plt.ylabel('P/E Ratio')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[67]:


window_10 = 10
window_50 = 50


# In[68]:


df['MA_10'] = df['Close'].rolling(window=window_10).mean()
df['MA_50'] = df['Close'].rolling(window=window_50).mean()


# In[69]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.plot(df['Date'], df['MA_10'], label='10-day MA', color='orange')
plt.plot(df['Date'], df['MA_50'], label='50-day MA', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Technical Analysis - Moving Averages')
plt.legend()
plt.grid(True)
plt.xticks(df['Date'][::15], rotation=45)
plt.tight_layout()
plt.show()


# In[70]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.title('Tesla Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(df['Date'][::30], rotation=45)
plt.legend()
plt.grid(True)


# In[71]:


plt.figure(figsize=(8, 6))
sns.histplot(df['Close'], bins=20, kde=True, color='green')
plt.title('Histogram of Closing Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')


# In[72]:


plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Close'], color='orange')
plt.title('Box Plot of Closing Prices')


# In[73]:


numeric_columns = df.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
# Show the heatmap
plt.show()


# In[74]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Volume'], y=df['Close'], color='purple')
plt.title('Volume vs. Closing Price')
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.tight_layout()
plt.show()


# In[75]:


df['Date'] = pd.to_datetime(df['Date'])


# In[76]:


df.set_index('Date', inplace=True)
tsla_close = df['Close']


# In[77]:


train_size = int(len(tsla_close) * 0.8)
train, test = tsla_close[:train_size], tsla_close[train_size:]


# In[78]:


import statsmodels.api as sm


# In[79]:


model = sm.tsa.arima.ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()


# In[80]:


predictions = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse}")


# In[81]:


plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.legend()
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show() # Add this line to display the plot


# In[ ]:




