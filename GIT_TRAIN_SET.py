import pandas as pd
import numpy as np


sp_price = pd.read_csv('data/raw/sp500.csv')
sp_price.index = pd.DatetimeIndex(sp_price['Date'])
sp_price = sp_price[['Close']]
sp_price.columns = ['close']
sp_price = sp_price[sp_price.index >= '2000-01-01']
sp_ret = np.log(sp_price/sp_price.shift(1)).dropna()



r_length = len(sp_ret[sp_ret.index<='2014-01-01'])
i=0
for reb_date in pd.date_range(start = '2014-01-01', end='2023-09-01', freq='3MS'):
    train_samples = []
    tempdat = sp_ret[sp_ret.index <= reb_date]
    tempdat = tempdat.iloc[-r_length:]

    train_samples = tempdat - (tempdat.mean()[0])
    train_samples = train_samples / train_samples.std()

    train_samples.to_csv(('data/trainset/train' + str(i) + '.csv'))
    i +=1