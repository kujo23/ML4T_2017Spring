import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import seaborn as sns
from util import get_data, plot_data

def indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['AAPL'], n=10, gen_plot=False, verbose=False):

    original_sd = sd
    sd = sd+timedelta(days=-3*n)
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values
    df = prices.copy()
    df[1:] = (df[1:]/df[0:-1].values)

    days = prices.shape[0]
    for i in range(1,days):
        df.ix[i,:] = df.ix[i,:] *df.ix[i-1,:]
    df['momentum']= (df.ix[2*n:,0]/df.ix[0:-2*n:,0].values)-1.
    df['sma']= (df.ix[:,0] / df.ix[:,0].rolling(window=n,center=False).mean())-1
    df['normal_price']= (df.ix[:,0]/df.ix[0,0])
    df['sma_raw']=  (df.ix[:,'normal_price'].rolling(window=n,center=False).mean())
    df['price']= df.ix[:,0]

    #ema
    ema_multiplier = 2/(n+1.)
    df['sma_actual']=  (df.ix[:,0].rolling(window=n,center=False).mean())
    df.ix[n-1,'ema_raw'] =  df.ix[n-1,'sma_actual']
    df.ix[n-1,'ema_raw_normal'] =  df.ix[n-1,'sma_raw']
    for i in range(n,days):
        df.ix[i,'ema_raw'] = (df.ix[i,0] - df.ix[i-1,'ema_raw'])*ema_multiplier +  df.ix[i-1,'ema_raw']
        df.ix[i,'ema_raw_normal'] = (df.ix[i,'normal_price'] - df.ix[i-1,'ema_raw_normal'])*ema_multiplier +  df.ix[i-1,'ema_raw_normal']
    df['ema'] = df.ix[n-1:,0]/df.ix[n-1:,'ema_raw']-1.
    df = df.fillna(0)
    df = df.ix[original_sd:,:]

    #zscore
    df['sma_zscore'] =  (df.sma-df.sma.mean())/df.sma.std()
    df['ema_zscore'] =  (df.ema-df.ema.mean())/df.ema.std()
    df['momentum_zscore'] =  (df.momentum-df.momentum.mean())/df.momentum.std()
    df['normal_price']= (df.ix[:,'normal_price']/df.ix[0,'normal_price'])

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        plt.rcParams["figure.figsize"] = [15,6]

        plt.title('Momentum', fontsize=24, fontweight='bold')
        plt.ylabel('Normalised Values', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        line_price, = plt.plot(df['normal_price'])
        line_indicator, = plt.plot(df['momentum'])
        plt.legend([line_price, line_indicator], ['Normalise Price', '10 Days Momentum'], fontsize=16, loc=0)
        plt.plot()
        plt.savefig('Momentum.png', bbox_inches='tight')


        plt.title('SMA', fontsize=24, fontweight='bold')
        plt.ylabel('Normalised Values', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        line_price, = plt.plot(df['normal_price'])
        line_indicator, = plt.plot(df['sma'])
        line_indicator_raw, = plt.plot(df['sma_raw'])
        plt.legend([line_price,line_indicator_raw, line_indicator], ['Normalise Price','SMA','Normalise Price/SMA'], fontsize=16, loc=0)
        plt.plot()
        plt.savefig('SMA.png', bbox_inches='tight')

        plt.title('EMA', fontsize=24, fontweight='bold')
        plt.ylabel('Normalised Values', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        line_price, = plt.plot(df['normal_price'])
        line_indicator, = plt.plot(df['ema'])
        line_indicator_raw, = plt.plot(df['ema_raw_normal'])
        plt.legend([line_price,line_indicator_raw, line_indicator], ['Normalise Price','EMA','Normalise Price/EMA'], fontsize=16, loc=0)
        plt.plot()
        plt.savefig('EMA.png', bbox_inches='tight')

    if verbose:
        print df

    return df[['price','normal_price','sma','ema','momentum','sma_zscore','ema_zscore','momentum_zscore']]

def test_code():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['AAPL']
    df = indicators(sd = start_date, ed = end_date,\
        syms = symbols,gen_plot = False,n=10,verbose=False)
    print df

if __name__ == "__main__":
    test_code()
