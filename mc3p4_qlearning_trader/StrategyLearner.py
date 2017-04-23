"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut

class StrategyLearner(object):

    def author(self):
        return 'llee81'

    def indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1),syms = ['AAPL'], n=10, gen_plot=False, verbose=False):
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

        return df[['price','normal_price','sma','ema','momentum','sma_zscore','ema_zscore','momentum_zscore']]

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        #Actions: BUY, SELL, NOTHING - 0,1,2
        self.learner  = ql.QLearner(num_states=1000,\
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.98, \
        radr = 0.999, \
        dyna = 0, \
        verbose=False) #initialize the learner

        # convert the discretize values
        def get_discrete(thresholds,my_value):
            positions = np.where(thresholds>my_value)[0]
            return len(thresholds)-1 if len(positions)==0 else positions[0]

        #convert indicator value to a state
        def get_state(idx,my_holding):
            ema = df_ema.ix[idx,0]
            momentum = df_momentum.ix[idx,0]
            return get_discrete(thres_ema,ema)*100 + get_discrete(thres_momentum,momentum)*10 + my_holding

        def make_trade(data,my_idx,my_holding,my_action):
            #Action BUY, SELL, NOTHING - 0,1,2
            #Holding BUY, SELL, NOTHING - 1,2,3
            reward=None
            new_holding = None

            if my_holding==1: # OWN LONG
                if my_action==0 or my_action==2: #BUY,Hold
                    reward = data.ix[i,0]-data.ix[i-1,0]
                    new_holding = 1
                else: #sell close
                    reward = 0
                    new_holding = 3
            elif my_holding==2: #OWN SHORT
                if my_action==1 or my_action==2: #sell,Hold
                    reward = data.ix[i-1,0] - data.ix[i,0]
                    new_holding = 2
                else: #buy close
                    reward = 0
                    new_holding = 3
            elif my_holding==3: # OWN NOTHING
                if my_action==0: #buy
                    reward = data.ix[i,0]-data.ix[i-1,0]
                    new_holding = 1
                elif my_action==1: #sell
                    reward = data.ix[i-1,0] - data.ix[i,0]
                    new_holding = 2
                else: #hold
                    reward = 0
                    new_holding = 3

            return new_holding,reward

        # convert the discretize values
        def discretize(values,level=10):
            step_size = values.shape[0]/level
            df_sort = values.sort_values(by=values.columns[0])
            threshold = np.zeros(level)
            for i in range(level):
                threshold[i] = df_sort.ix[step_size*(i+1)-1,0]
            return threshold

        #GET PRICE
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

        #GET INDICATOR
        df_indicators = indicators(sd = sd, ed = ed,syms =[symbol],gen_plot = False,n=10,verbose=False)
        df_ema = pd.DataFrame(data=df_indicators.ema,index=df_indicators.index)
        df_momentum = pd.DataFrame(data=df_indicators.momentum,index=df_indicators.index)
        thres_ema = discretize(df_ema)
        thres_momentum = discretize(df_momentum)

        # each iteration involves one trip to the goal
        iterations = 50
        scores = np.zeros((iterations,1))
        for iteration in range(iterations):
            total_reward = 0
            current_hold = 3 #current_hold BUY, SELL, NOTHING - 1,2,3
            state = get_state(trades.index[0],current_hold)  #XYZ, XY are indicators
            action = self.learner.querysetstate(state) #action BUY, SELL, NOTHING - 0,1,2

            for i in range(1,trades.shape[0]):
                trades.ix[trades.index[i-1],'action']=action #last action
                trades.ix[trades.index[i-1],'hold']=current_hold #last hold
                trades.ix[trades.index[i-1],'state']=state

                current_hold,r= make_trade(trades,trades.index[i],current_hold,action)
                state = get_state(trades.index[i],current_hold)
                total_reward = total_reward + r
                trades.ix[trades.index[i],'reward']=r
                action = self.learner.query(state,r)

            if self.verbose: print total_reward
            scores[iteration] = total_reward
            if iteration>2 and np.absolute(scores[iteration]-scores[iteration-1])<0.005:
                if self.verbose: print 'total run: ',iteration
                break

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "AAPL", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 100000):

        #GET PRICE
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

        # convert the discretize values
        def discretize(values,level=10):
            step_size = values.shape[0]/level
            df_sort = values.sort_values(by=values.columns[0])
            threshold = np.zeros(level)
            for i in range(level):
                threshold[i] = df_sort.ix[step_size*(i+1)-1,0]
            return threshold

        # convert the discretize values
        def get_discrete(thresholds,my_value):
            positions = np.where(thresholds>my_value)[0]
            return len(thresholds)-1 if len(positions)==0 else positions[0]

        #convert indicator value to a state
        def get_state(idx,my_holding):
            ema = df_ema.ix[idx,0]
            momentum = df_momentum.ix[idx,0]
            return get_discrete(thres_ema,ema)*100 + get_discrete(thres_momentum,momentum)*10 + my_holding

        def make_trade(data,my_idx,my_holding,my_action):
            #Action BUY, SELL, NOTHING - 0,1,2
            #Holding BUY, SELL, NOTHING - 1,2,3
            reward=None
            new_holding = None

            if my_holding==1: # OWN LONG
                if my_action==0 or my_action==2: #BUY,Hold
                    reward = data.ix[i,0]-data.ix[i-1,0]
                    new_holding = 1
                else: #sell close
                    reward = 0
                    new_holding = 3
            elif my_holding==2: #OWN SHORT
                if my_action==1 or my_action==2: #sell,Hold
                    reward = data.ix[i-1,0] - data.ix[i,0]
                    new_holding = 2
                else: #buy close
                    reward = 0
                    new_holding = 3
            elif my_holding==3: # OWN NOTHING
                if my_action==0: #buy
                    reward = data.ix[i,0]-data.ix[i-1,0]
                    new_holding = 1
                elif my_action==1: #sell
                    reward = data.ix[i-1,0] - data.ix[i,0]
                    new_holding = 2
                else: #hold
                    reward = 0
                    new_holding = 3

            return new_holding,reward

        # here we build a fake set of trades
        #GET INDICATOR
        df_indicators = indicators(sd = sd, ed = ed,syms =[symbol],gen_plot = False,n=10,verbose=False)
        df_ema = pd.DataFrame(data=df_indicators.ema,index=df_indicators.index)
        df_momentum = pd.DataFrame(data=df_indicators.momentum,index=df_indicators.index)
        thres_ema = discretize(df_ema)
        thres_momentum = discretize(df_momentum)

        current_hold = 3 #current_hold BUY, SELL, NOTHING - 1,2,3
        state = get_state(trades.index[0],current_hold)  #XYZ, XY are indicators
        action = self.learner.querysetstate(state) #action BUY, SELL, NOTHING - 0,1,2

        hold_amount = 0
        for i in range(0,trades.shape[0]):
            trades.ix[trades.index[i-1],'action']=action #last action
            trades.ix[trades.index[i-1],'hold']=current_hold #last hold
            trades.ix[trades.index[i-1],'state']=state

            if action==0 and hold_amount!=200:
                hold_amount = hold_amount + 200
                trades.ix[trades.index[i],'order'] = 200
            elif action==1 and hold_amount!=-200:
                hold_amount = hold_amount - 200
                trades.ix[trades.index[i],'order'] = -200
            else:
                trades.ix[trades.index[i],'order'] = 0

            current_hold,r= make_trade(trades,trades.index[i],current_hold,action)
            state = get_state(trades.index[i],current_hold)
            action = self.learner.querysetstate(state)

        return pd.DataFrame(data=trades.order,index=trades.index)


if __name__=="__main__":
    print "One does not simply think up a strategy"
