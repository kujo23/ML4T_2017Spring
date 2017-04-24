"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def author():
    return 'llee81'


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    # GET PRICES OF ALL USED SYMBOLS
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    list_symbols = [i for i in orders_df.Symbol.unique()]
    all_symbols = get_data(list_symbols, pd.date_range(start_date, end_date))
    orders_df_full = pd.DataFrame(index=all_symbols.index)  # SAVE ALL DAYS DATA

    # ADD ALL SYMBOLS IN orders_df
    for sym in list_symbols:
        orders_df.ix[0, sym] = "0"
        orders_df_full.ix[0, sym] = "0"

    # POPULATE THE
    orders_df.ix[0, "cash"] = start_val
    orders_df.ix[0, "value"] = start_val
    orders_df_full.ix[0, "cash"] = start_val
    orders_df_full.ix[0, "Symbol"] = ''
    orders_df_full.ix[0, "Order"] = ''
    orders_df_full.ix[0, "Shares"] = ''
    orders_df_full.ix[0, "stock_price"] = ''
    orders_df_full.ix[0, "cash_used"] = ''
    orders_df_full.ix[0, "value"] = ''
    orders_df_full.ix[0, "leverage"] = ''
    orders_df_full = orders_df_full.fillna(0)
    orders_df = orders_df.fillna(0)

    # SAME DAY ORDER
    for i in range(orders_df.shape[0]):
        if i > 0:
            if orders_df.index[i] == orders_df.index[i - 1] and orders_df.ix[i, "Symbol"] == orders_df.ix[
                        i - 1, "Symbol"]:
                sym = orders_df.ix[i, "Symbol"]
                prev_add_amt = float(orders_df.ix[i - 1, "Shares"]) if orders_df.ix[
                                                                           i - 1, 'Order'] == 'BUY' else -float(
                    orders_df.ix[i - 1, "Shares"])
                add_amt = float(orders_df.ix[i, "Shares"]) if orders_df.ix[i, 'Order'] == 'BUY' else -float(
                    orders_df.ix[i, "Shares"])

                orders_df.ix[i - 1, "Shares"] = 0
                final_amt = add_amt + prev_add_amt
                orders_df.ix[i, "Shares"] = np.absolute(final_amt)
                orders_df.ix[i, "Order"] = 'BUY' if final_amt >= 0 else 'SELL'

    # COMPRESS ORDER
    for i in range(orders_df.shape[0]):
        sym = orders_df.ix[i, "Symbol"]
        add_amt = float(orders_df.ix[i, "Shares"]) if orders_df.ix[i, 'Order'] == 'BUY' else -float(
            orders_df.ix[i, "Shares"])
        orders_df.ix[orders_df.index[i], sym] = add_amt

    # REMOVE DUPLICATE DAYS
    orders_df = orders_df.groupby(orders_df.index).first()
    orders_df.ix[:, 'date1'] = orders_df.index
    orders_df_full.ix[:, 'date1'] = orders_df_full.index

    # COUNT ORDER VALUE
    for i in range(orders_df.shape[0]):
        stock_value = 0
        leverage_stocks = 0
        for sym in list_symbols:
            stock_value = stock_value + float(orders_df.ix[i, sym]) * float(
                all_symbols.ix[orders_df.index[i], sym]) * -1
            leverage_stocks = leverage_stocks + np.absolute(float(orders_df.ix[i, sym])) * float(
                all_symbols.ix[orders_df.index[i], sym])
            orders_df.ix[i, sym + '_p'] = float(orders_df.ix[i, sym]) * float(
                all_symbols.ix[orders_df.index[i], sym]) * -1

        orders_df.ix[i, 'cash_impact'] = stock_value
        orders_df.ix[i, 'leverage_stocks'] = leverage_stocks

        # print orders_df

    #orders_df.to_csv("orders_df.csv")

    def get_order_values(df_row):
        stock_value = 0
        leverage_stocks = 0
        for sym in list_symbols:
            if pd.isnull(df_row[sym]) == False:
                stock_value = stock_value + float(df_row[sym]) * float(all_symbols.ix[df_row.ix['date1'], sym])
                leverage_stocks = leverage_stocks + np.absolute(
                    float(df_row[sym]) * float(all_symbols.ix[df_row.ix['date1'], sym]))
        return stock_value, leverage_stocks

    leverage = 0
    stock_value = 0
    cash = start_val

    # init first row
    orders_df_full.ix[0, :] = orders_df.ix[0, :]
    current_stock_value, current_leverage = get_order_values(orders_df_full.loc[orders_df_full.index[0], :])
    orders_df_full.ix[0, "cash"] = orders_df_full.ix[0, "cash"] + orders_df.ix[0, 'cash_impact']
    orders_df_full.ix[0, "value"] = current_stock_value + orders_df_full.ix[0, "cash"]
    orders_df_full.ix[0, "leverage"] = current_leverage / orders_df_full.ix[0, "value"]
    if orders_df_full.ix[0, "leverage"] > 1.5:
        for sym in list_symbols:
            orders_df_full.ix[0, sym] = 0
        orders_df_full.ix[0, 'overleverage'] = orders_df_full.ix[0, "leverage"]
        orders_df_full.ix[0, "cash"] = start_val

    for i in range(1, orders_df_full.shape[0]):

        # copy down all symbol holdings
        for sym in list_symbols:
            orders_df_full.ix[i, sym] = orders_df_full.ix[i - 1, sym]
            orders_df_full.ix[i, "cash"] = orders_df_full.ix[i - 1, "cash"]

        # ADD NEW ORDER STOCKS
        if orders_df_full.index[i] in orders_df.index:
            for sym in list_symbols:
                orders_df_full.ix[i, sym] = float(orders_df_full.ix[i, sym]) + float(
                    orders_df.ix[orders_df_full.index[i], sym])
            orders_df_full.ix[i, "cash"] = orders_df_full.ix[i, "cash"] + orders_df.ix[
                orders_df_full.index[i], 'cash_impact']
            # check overleverage
            current_stock_value, current_leverage = get_order_values(orders_df_full.loc[orders_df_full.index[i], :])
            order_total_value = current_stock_value + orders_df_full.ix[i, "cash"]
            order_total_leverage = current_leverage / order_total_value
            if order_total_leverage > 1.5:
                for sym in list_symbols:
                    orders_df_full.ix[i, sym] = orders_df_full.ix[i - 1, sym]
                orders_df_full.ix[i, 'overleverage'] = order_total_leverage
                orders_df_full.ix[i, "cash"] = orders_df_full.ix[i - 1, "cash"]

        current_stock_value, current_leverage = get_order_values(orders_df_full.loc[orders_df_full.index[i], :])
        orders_df_full.ix[i, "value"] = current_stock_value + orders_df_full.ix[i, "cash"]
        orders_df_full.ix[i, "leverage"] = current_leverage / orders_df_full.ix[i, "value"]

    #orders_df_full.to_csv("orders_df_full.csv")
    return orders_df_full.value


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX.
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
