#!/usr/bin/env python
# coding: utf-8

# In[99]:


from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[100]:


# Assuming you've imported pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/lime/Downloads/TSLAF.csv')

# Display the first few rows of the DataFrame
print(df.head())


# In[101]:


print(df[df['Date'].astype(str).str.startswith('2022-12-25')])


# In[102]:


# Standard correlation plot using matplotlib
corrMatrix = df.corr()

# Standard correlation plot using matplotlib
plt.matshow(corrMatrix)
plt.colorbar()


plt.title('Correlation Matrix', pad=20)
plt.show()

sns.heatmap(corrMatrix, annot=True, cmap='coolwarm')

# Save the plot as a picture
plt.savefig('correlation_matrix.png')


# In[103]:


df['Date'] = pd.to_datetime(df['Date'])

# Add a 'Weekday' column
df['Weekday'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6

# Filter out weekends
df = df[df['Weekday'] < 5]  # Keep only weekdays

# Create a scatter plot
plt.figure(figsize=(12, 8))

plt.scatter(df['Date'], df['Close'], label='Close', alpha=0.6)
plt.scatter(df['Date'], df['Open'], label='Open', alpha=0.6)
plt.scatter(df['Date'], df['Low'], label='Low', alpha=0.6)
plt.scatter(df['Date'], df['High'], label='High', alpha=0.6)

plt.title('Scatter Plot of Date vs Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()
plt.show()


# In[104]:


model = RandomForestRegressor()




# In[105]:


train_data = df.iloc[:-2].copy()
test_data = df.iloc[-2:-1].copy()  # The second last day

df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame to only include data up to October 23, 2023
df_filtered = df[df['Date'] <= '2023-10-23']
# Creates a new column 'Close_Lag1' that contains the 'Close' values shifted down by one row.
df_filtered['Close_Lag1'] = df_filtered['Close'].shift(1)

# Drop NaNs
df_filtered.dropna(inplace=True)
#Filters data to only be Close in a new row included up to October 23, 2023 

# Set up the features and target for the training data
X_train = df_filtered[['Close_Lag1']]
y_train = df_filtered['Close']

# Initialize and train the model
model = RandomForestRegressor() 
model.fit(X_train, y_train) // #df



# In[98]:


X_last_day = df_filtered[df_filtered['Date'] == '2023-10-23'][['Close_Lag1']].values.reshape(1, -1) # last day before prediction

# Predict using the model
predicted_close = model.predict(X_last_day) #use the 

print(f"Predicted Close value for October 24, 2023: {predicted_close[0]}")


# In[107]:


df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame to only include data up to October 23, 2023
df_filtered = df[df['Date'] <= '2023-10-23']

# Creates a new column 'Close_Lag1' that contains the 'Close' values shifted down by one row.
df_filtered['Close_Lag1'] = df_filtered['Close'].shift(1)

# Drop NaNs
df_filtered.dropna(inplace=True)


#Filters data to only be Close in a new row included up to October 23, 2023 


# Set up the features and target for the training data
X_train = df_filtered[['Close_Lag1']]
y_train = df_filtered['Close']

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[109]:


# Predict using the Linear Regression model
lr_predicted_close = lr_model.predict(X_last_day)


print(f"Linear Regression - Predicted Close value for October 24, 2023: {lr_predicted_close[0]}")


# In[121]:


if len(df_filtered['Date']) == len(df_filtered['Close']):
    dates_to_plot = df_filtered['Date']
else:
    # Drop the first date to match the length of df_filtered['Close']
    dates_to_plot = df_filtered['Date'].iloc[1:]

plt.figure(figsize=(12, 6))
plt.plot(dates_to_plot, df_filtered['Close'], label='Actual Close Price', color='blue')

# Add the predicted value for October 24, 2023
predicted_date = pd.Timestamp('2023-10-24')
plt.scatter(predicted_date, lr_predicted_close[0], color='red', zorder=5)
plt.annotate(f'Predicted: {lr_predicted_close[0]}', (predicted_date, lr_predicted_close[0]), textcoords="offset points", xytext=(0,10), ha='center')


plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price vs Date')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




