#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[2]:


btc_data = pd.read_csv('BTC.csv')
btc_data.head()


# In[3]:


btc_data.info()


# In[4]:


btc_data.info()


# In[9]:


btc_data.describe()


# In[10]:


btc_data.isnull().sum()


# In[11]:


# Handle missing values in 'tradecount' column
btc_data['tradecount'].fillna(btc_data['tradecount'].mean(), inplace=True) 
btc_data.isnull().sum()


# In[12]:


btc_data.dtypes


# In[13]:


# convert date format
btc_data['date'] = pd.to_datetime(btc_data['date'], format="%Y-%m-%d", errors='coerce')


# In[16]:


# Visualize data
# Time series plot
plt.figure(figsize=(12, 6))
plt.plot(btc_data['date'], btc_data['close'], label='Closing Price') 
plt.title('BTC Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.show()


# In[17]:


#Plot the trading volume over time.

plt.figure(figsize=(12, 6))
plt.bar(btc_data['date'], btc_data['Volume BTC'], color='green', alpha=0.5)
plt.title('BTC Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume BTC')
plt.show()


# In[18]:


#Histograms
colors = ['skyblue', 'salmon', 'lightgreen', 'orange']
plt.figure(figsize=(12, 8))

for i, column in enumerate(['open', 'high', 'low', 'close']): 
    plt.hist(btc_data[column], bins=20, alpha=0.7, color=colors[i], label=column)
    
plt.title('Histograms of BTC Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[19]:


# Select only numeric columns for correlation matrix
num_col = btc_data.select_dtypes(include=['float64']).columns
corr = btc_data[num_col].corr()


# In[20]:


#Plot the correlation heatmap
import seaborn as sns 
plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f') 
plt.title('Correlation Heatmap')
plt.show()


# In[21]:


# Feature scaling using Min-Max scaling
scaler = MinMaxScaler()

fs = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT', 'tradecount']

# Apply Min-Max scaling to selected features
btc_data[fs] = scaler.fit_transform(btc_data[fs])


# In[22]:


# Feature engineering

window_size = 10
btc_data['SMA'] = btc_data['close'].rolling(window=window_size).mean()


# In[23]:


# Display the preprocessed data
print("Preprocessed Data:")
print(btc_data.head())


# In[24]:


gamma = 0.9 # Discount factor
alpha = 0.1 # Learning rate
epsilon = 0.1 # Exploration-exploitation trade-off


# In[25]:


def q_learning(Q, state, action, reward, next_state):
    current_value = Q[state, action]
    max_future_value = np.max(Q[next_state, :])
    new_value = (1 - alpha) * current_value + alpha * (reward + gamma *max_future_value)
    Q[state, action] = new_value 
    return Q


# In[26]:


train_data, test_data = train_test_split(btc_data, test_size=0.2, random_state=42)


# In[27]:


# Initialize Q-table
num_states = 100
num_actions = 2

Q = np.zeros((num_states, num_actions))


# In[28]:


for index, row in train_data.iterrows():
    state = int(index % num_states)
    action = 1 if row['close'] < row['SMA'] else 0 # Buy if the price is below the moving average, else sell
    reward = row['close'] - row['SMA'] # Reward is the difference from the␣ moving average
    next_state = int((index + 1) % num_states) # Simple next state representation, adjust accordingly
    Q = q_learning(Q, state, action, reward, next_state)


# In[29]:


# Testing the Q-learning model
total_reward = 0
actions_taken = []

for index, row in test_data.iterrows(): 
    state = int(index % num_states) 
    action = np.argmax(Q[state, :]) 
    actions_taken.append(action) 
    reward = row['close'] - row['SMA'] 
    total_reward += reward


# In[30]:


# Visualize actions taken during testing with annotations
plt.figure(figsize=(8, 6))
plt.plot(test_data['date'], test_data['close'], label='Closing Price')

# Filter buy signals
buy_signals = test_data['date'][np.array(actions_taken) == 1]
plt.scatter(buy_signals, test_data['close'].loc[test_data['date'].isin(buy_signals)],
            color='green', label='Buy Signal', marker='^')

# Filter sell signals
sell_signals = test_data['date'][np.array(actions_taken) == 0]
plt.scatter(sell_signals, test_data['close'].loc[test_data['date'].isin(sell_signals)],
            color='red', label='Sell Signal', marker='v')

# Annotate Buy signals
for date, price in zip(buy_signals, test_data['close'].loc[test_data['date'].isin(buy_signals)]):
    plt.annotate('Buy', (date, price), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='green')

# Annotate Sell signals
for date, price in zip(sell_signals, test_data['close'].loc[test_data['date'].isin(sell_signals)]):
    plt.annotate('Sell', (date, price), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='red')

plt.title('BTC Closing Price and Trading Signals with Annotations')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.show()

