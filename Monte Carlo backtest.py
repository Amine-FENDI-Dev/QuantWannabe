import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
import random as rd
from sklearn.model_selection import train_test_split


# color list used only to create gradient effect in bar chart
global colorlist
colorlist = ['#fffb77','#fffa77','#fff977','#fff876','#fff776','#fff676','#fff576','#fff475',
'#fff375','#fff275','#fff175','#fff075','#ffef74','#ffef74','#ffee74','#ffed74',
'#ffec74','#ffeb73','#ffea73','#ffe973','#ffe873','#ffe772','#ffe672','#ffe572',
'#ffe472','#ffe372','#ffe271','#ffe171','#ffe071','#ffdf71','#ffde70','#ffdd70',
'#ffdc70','#ffdb70','#ffda70','#ffd96f','#ffd86f','#ffd76f','#ffd66f','#ffd66f',
'#ffd56e','#ffd46e','#ffd36e','#ffd26e','#ffd16d','#ffd06d','#ffcf6d','#ffce6d',
'#ffcd6d','#ffcc6c','#ffcb6c','#ffca6c','#ffc96c','#ffc86b','#ffc76b','#ffc66b',
'#ffc56b','#ffc46b','#ffc36a','#ffc26a','#ffc16a','#ffc06a','#ffbf69','#ffbe69',
'#ffbd69','#ffbd69','#ffbc69','#ffbb68','#ffba68','#ffb968','#ffb868','#ffb768',
'#ffb667','#ffb567','#ffb467','#ffb367','#ffb266','#ffb166','#ffb066','#ffaf66',
'#ffad65','#ffac65','#ffab65','#ffa964','#ffa864','#ffa763','#ffa663','#ffa463',
'#ffa362','#ffa262','#ffa062','#ff9f61','#ff9e61','#ff9c61','#ff9b60','#ff9a60',
'#ff9860','#ff975f','#ff965f','#ff955e','#ff935e','#ff925e','#ff915d','#ff8f5d',
'#ff8e5d','#ff8d5c','#ff8b5c','#ff8a5c','#ff895b','#ff875b','#ff865b','#ff855a',
'#ff845a','#ff8259','#ff8159','#ff8059','#ff7e58','#ff7d58','#ff7c58','#ff7a57',
'#ff7957','#ff7857','#ff7656','#ff7556','#ff7455','#ff7355','#ff7155','#ff7054',
'#ff6f54','#ff6d54','#ff6c53','#ff6b53','#ff6953','#ff6852','#ff6752','#ff6552',
'#ff6451','#ff6351','#ff6250','#ff6050','#ff5f50','#ff5e4f','#ff5c4f','#ff5b4f',
'#ff5a4e','#ff584e','#ff574e','#ff564d','#ff544d','#ff534d','#ff524c','#ff514c',
'#ff4f4b','#ff4e4b','#ff4d4b','#ff4b4a','#ff4a4a']


def monte_carlo(data, testsize=0.5, simulation=100, **kwargs):

    # Monte Carlo idea:
    # 1. Learn average return and volatility from historical data
    # 2. Generate many random future paths using those statistics
    # 3. Compare all simulated paths to history
    # 4. Pick the one that fits history best

    df, test = train_test_split(data, test_size=testsize, shuffle=False, **kwargs)
    forecast_horizon = len(test)

    df = df.loc[:, ['Close']]

    # compute log returns
    returnn = np.log(df['Close'].iloc[1:] / df['Close'].shift(1).iloc[1:])

    # drift is the average expected return
    drift = returnn.mean() - returnn.var() / 2

    # dictionary to store all simulated price paths
    d = {}

    # repeat simulation many times
    for counter in range(simulation):

        # start simulation from first real price
        d[counter] = [df['Close'].iloc[0]]

        # simulate both training and testing period
        for i in range(len(df) + forecast_horizon - 1):

            # generate random shock from standard normal distribution
            random_shock = rd.gauss(0, 1)

            # geometric brownian motion formula
            sde = drift + returnn.std() * random_shock

            # next price = previous price * exponential growth
            next_price = d[counter][-1] * np.exp(sde)

            d[counter].append(next_price.item())

    # choose simulation that matches training data best
    # best = smallest standard deviation from real data
    std = float('inf')
    pick = 0

    for counter in range(simulation):

        temp = np.std(np.subtract(
            d[counter][:len(df)], df['Close']))

        if temp < std:
            std = temp
            pick = counter

    return forecast_horizon, d, pick


def plot(df, forecast_horizon, d, pick, ticker):

    # plot all simulations on training data
    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(int(len(d))):
        if i != pick:
            ax.plot(df.index[:len(df)-forecast_horizon],
                    d[i][:len(df)-forecast_horizon],
                    alpha=0.05)

    # highlight best fitted path
    ax.plot(df.index[:len(df)-forecast_horizon],
            d[pick][:len(df)-forecast_horizon],
            c='#5398d9', linewidth=5, label='Best Fitted')

    df['Close'].iloc[:len(df)-forecast_horizon].plot(
        c='#d75b66', linewidth=5, label='Actual')

    plt.title(f'Monte Carlo Simulation\nTicker: {ticker}')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()

    # compare full best path (train + test) with real data
    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(d[pick], label='Best Fitted', c='#edd170')
    plt.plot(df['Close'].tolist(), label='Actual', c='#02231c')

    # vertical line separates training and testing
    plt.axvline(len(df)-forecast_horizon, linestyle=':', c='k')

    plt.title(f'Training vs Testing\nTicker: {ticker}')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('T+Days')
    plt.show()


def test(df, ticker, simu_start=100, simu_end=1000, simu_delta=100, **kwargs):

    table = pd.DataFrame()
    table['Simulations'] = np.arange(simu_start, simu_end+simu_delta, simu_delta)
    table.set_index('Simulations', inplace=True)
    table['Prediction'] = 0

    # test whether increasing number of simulations improves direction prediction
    for i in np.arange(simu_start, simu_end+1, simu_delta):

        forecast_horizon, d, pick = monte_carlo(df, simulation=i, **kwargs)

        # actual direction (up or down)
        actual_return = np.sign(
            df['Close'].iloc[len(df)-forecast_horizon] - df['Close'].iloc[-1])

        # predicted direction
        best_fitted_return = np.sign(
            d[pick][len(df)-forecast_horizon] - d[pick][-1])

        # 1 = correct direction, -1 = wrong direction
        table.at[i, 'Prediction'] = np.where(
            actual_return == best_fitted_return, 1, -1)

    # visualize whether more simulations increase accuracy
    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_visible(False)

    plt.barh(np.arange(1, len(table)*2+1, 2),
             table['Prediction'],
             color=colorlist[0::int(len(colorlist)/len(table))])

    plt.xticks([-1, 1], ['Failure', 'Success'])
    plt.yticks(np.arange(1, len(table)*2+1, 2), table.index)
    plt.xlabel('Prediction Accuracy')
    plt.ylabel('Times of Simulation')
    plt.title(f"Does more simulation improve prediction?\nTicker: {ticker}")
    plt.show()


def main():

    stdate = '2016-01-15'
    eddate = '2019-01-15'
    ticker = 'GE'

    # download historical price data
    df = yf.download(ticker, start=stdate, end=eddate)
    df.index = pd.to_datetime(df.index)

    forecast_horizon, d, pick = monte_carlo(df)

    plot(df, forecast_horizon, d, pick, ticker)

    test(df, ticker)


if __name__ == '__main__':
    main()
