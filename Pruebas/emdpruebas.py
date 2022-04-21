'''Pruebas de estrategias de pares utilizando
la EMD'''

import pandas as pd
import numpy as np
from pandas import read_csv
from numpy import cumsum, log, polyfit, sqrt, std, subtract, var, seterr
from numpy.random import randn
import matplotlib.pyplot as plt
from PyEMD import EMD
from PyEMD import CEEMDAN
import statsmodels.tsa.stattools as sts
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as smf
import decimal
import quandl
from numpy import linalg as LA

quandl.ApiConfig.api_key = "8y1PUCdzbufUBxucxqqn"
from arch.unitroot import VarianceRatio
from hurst import compute_Hc
import statsmodels.tsa.vector_ar.vecm as vm
import datetime
import http.client
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from numpy.matlib import repmat
import math
from numpy.linalg import inv, eig, cholesky as chol
import csv
from statsmodels.regression.linear_model import OLS
import statsmodels.tsa.tsatools as tsat
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pandas_ods_reader import read_ods


def create_new_time_series_emd(symbol1, symbol2):
    """
    Create a new time serie substracting the IMFs of two time series.
    Those IMFs were exstract from EMD

    PARAMETER
    ---------
    symbol1 : string
    symbol2 : string
        Symbols of two stock

    RETURN
    ------
        A numpy
        A new time serie
    """

    path = '/home/angel/Desktop/Datos/sp500-1day/'
    stock1 = pd.read_csv(path + symbol1 + ".csv")
    stock2 = pd.read_csv(path + symbol2 + ".csv")
    first_date = max([min(stock1["date"]), min(stock2["date"])])
    last_date = min([max(stock1["date"]), max(stock2["date"])])

    stock1 = np.array(stock1['close'][np.where(
        np.array(stock1['date']) == first_date)[0][0]:np.where(
            np.array(stock1['date']) == last_date)[0][0]])
    stock2 = np.array(stock2['close'][np.where(
        np.array(stock2['date']) == first_date)[0][0]:np.where(
            np.array(stock2['date']) == last_date)[0][0]])
    emd1 = EMD()
    imfs1 = emd1(stock1)
    emd2 = EMD()
    imfs2 = emd2(stock2)
    # print(imfs1)
    return np.sum(imfs1[:len(imfs1) - 1], axis=0) - np.sum(
        imfs2[:len(imfs2) - 1], axis=0)


def create_new_time_series_ceemdan(symbol1, symbol2):
    """
    Create a new time serie substracting the IMFs of two time series.
    Those IMFs were exstract from CEEMDAN

    PARAMETER
    ---------
    symbol1 : string
    symbol2 : string
        Symbols of two stock

    RETURN
    ------
        A numpy
        A new time serie
    """

    path = '/home/angel/Desktop/Datos/sp500-1day/'
    stock1 = pd.read_csv(path + symbol1 + ".csv")
    stock2 = pd.read_csv(path + symbol2 + ".csv")
    first_date = max([min(stock1["date"]), min(stock2["date"])])
    last_date = min([max(stock1["date"]), max(stock2["date"])])

    stock1 = np.array(stock1['close'][np.where(
        np.array(stock1['date']) == first_date)[0][0]:np.where(
            np.array(stock1['date']) == last_date)[0][0]])
    stock2 = np.array(stock2['close'][np.where(
        np.array(stock2['date']) == first_date)[0][0]:np.where(
            np.array(stock2['date']) == last_date)[0][0]])
    ceemdan1 = CEEMDAN()
    c_imfs1 = ceemdan1(stock1)
    ceemdan2 = CEEMDAN()
    c_imfs2 = ceemdan2(stock2)
    # print(imfs1)
    return np.sum(c_imfs1[:len(c_imfs1) - 1], axis=0) - np.sum(
        c_imfs2[:len(c_imfs2) - 1], axis=0)


def graph_of_imfs_emd(symbol, flag=False):
    """
    Show the graphs IMFs of a stock and the graph of the stock

    PARAMETER
    ---------
    symbol : string
        Symbols of a stock

    RETURN
    ------
        None
        Show the graphs of IMFs
    """
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    stock = np.array(pd.read_csv(path + symbol + ".csv")['close'])
    emd = EMD()
    imfs = emd(stock)
    axis_x = np.arange(len(stock))
    plt.plot(axis_x, stock)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(symbol)
    plt.show()

    if flag:
        for imf in imfs:
            axis_x = np.arange(len(imf))
            plt.plot(axis_x, imf)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('IMF')
            plt.show()


def graph_of_imfs_ceemdan(symbol, flag=False):
    """
    Show the graphs IMFs of a stock and the graph of the stock

    PARAMETER
    ---------
    symbol : string
        Symbols of a stock

    RETURN
    ------
        None
        Show the graphs of IMFs
    """
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    stock = np.array(pd.read_csv(path + symbol + ".csv")['close'])
    ceemdan = CEEMDAN()
    c_imfs = ceemdan(stock)
    axis_x = np.arange(len(stock))
    plt.plot(axis_x, stock)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(symbol)
    plt.show()

    if flag:
        for imf in c_imfs:
            axis_x = np.arange(len(imf))
            plt.plot(axis_x, imf)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('IMF')
            plt.show()


def hurst_exponent_adftest(stock):
    """
    print the hurst exponent of a stock

    PARAMETER
    ---------
    symbol : string
        Symbols of a stock

    RETURN
    ------
        None
        print the hurst exponent
    """

    stock = pd.DataFrame(stock)
    # X = series.Value
    Y = sts.adfuller(stock, maxlag=1)
    # series.plot()
    print(type(stock))
    # test ADF con sus valores criticos
    print('test estadístico ADF : %f' % Y[0])
    print('Valores criticos:')
    for key, value in Y[4].items():
        print('\t%s: %.3f' % (key, value))

    # Exponente de Hurst
    H, c, data = compute_Hc(stock, kind='change', simplified=True)
    print('Exponente de Hurst:' "{:.4f}".format(H))

    # test de ratio de varianza
    # vr = VarianceRatio(log(stock))

    # print(stock)
    # print(log(stock))

    # print('xxxxxxxxxxxxxxxxxxxxxxxx')

    # print(vr.summary().as_text())

    # Vida media
    # stock['Value_lagged'] = stock['Value'].shift(1)
    # stock['delta'] = stock['Value'] - stock['Value_lagged']

    # results = smf.ols('delta ~ Value_lagged', data=stock).fit()
    # lam = results.params['Value_lagged']
    # print(lam)

    # halflife = -np.log(2) / lam
    # print('la vida media es de %f dias' % halflife)


def cadftest(series1, series2, flag=False):
    """
    Return the cadf test of two series and print crtical values

    PARAMETER
    ---------
    series1 : numpy array
        series of stock 1

    series2 : numpy array
        series of stock 2

    RETURN
    ------
        p-values of the cadf test
        Show the graphs of IMFs
    """

    series1 = pd.Series(series1)
    series2 = pd.Series(series2)
    coint_t, pvalue, crit_value = ts.coint(series1, series2)
    if flag:
        print('t-statistic = %f' % coint_t)
        print('pvalue = %f' % pvalue)
        print('valores criticos del test CADF')
        print(crit_value)
    return coint_t


def hurst_exponent_adftest(stock):
    stock = pd.DataFrame(stock)
    # X = series.Value
    Y = sts.adfuller(stock, maxlag=1)
    # series.plot()
    print(type(stock))
    # test ADF con sus valores criticos
    print('test estadístico ADF : %f' % Y[0])
    print('Valores criticos:')
    for key, value in Y[4].items():
        print('\t%s: %.3f' % (key, value))

    # Exponente de Hurst
    H, c, data = compute_Hc(stock, kind='change', simplified=True)
    print('Exponente de Hurst:' "{:.4f}".format(H))

    # test de ratio de varianza
    # vr = VarianceRatio(log(stock))

    # print(stock)
    # print(log(stock))

    # print('xxxxxxxxxxxxxxxxxxxxxxxx')

    # print(vr.summary().as_text())

    # Vida media
    # stock['Value_lagged'] = stock['Value'].shift(1)
    # stock['delta'] = stock['Value'] - stock['Value_lagged']

    # results = smf.ols('delta ~ Value_lagged', data=stock).fit()
    # lam = results.params['Value_lagged']
    # print(lam)

    # halflife = -np.log(2) / lam
    # print('la vida media es de %f dias' % halflife)


def fistdate_lastdate():
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    symbols = pd.read_csv(path + "sp500_symbols.csv")
    symbols = symbols['symbol'][:100]
    first_dates = []
    last_dates = []

    for symbol in symbols:
        # print(new_nodes[node1],node1)
        if symbol not in ['BF B', 'OGN', 'VTRS', 'CARR', 'OTIS']:
            dates = np.array(pd.read_csv(path + symbol + ".csv")["date"])
            first_dates.append(min(dates))
            last_dates.append(max(dates))
    return max(first_dates), min(last_dates)


def compute_ceemdan():
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    symbols = pd.read_csv(path + "sp500_symbols.csv")
    symbols = symbols['symbol'][:100]
    first_date, last_date = fistdate_lastdate()
    for symbol in symbols:
        print(symbol)
        stock = pd.read_csv(path + symbol + ".csv")
        stock_close = np.array(stock['close'][np.where(
            np.array(stock['date']) == first_date)[0][0]:np.where(
                np.array(stock['date']) == last_date)[0][0]])
        print(len(stock_close))
        ceemdan = CEEMDAN()
        c_imfs = ceemdan(stock_close)
        # print(c_imfs)
        c_imfs = pd.DataFrame(c_imfs).transpose()
        # print(np.array(c_imfs[5]))
        c_imfs.to_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol + '.csv')


def range_cociente():
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    cocientes = {}
    symbols = pd.read_csv(path + "sp500_symbols.csv")['symbol'][:100]
    for symbol1 in symbols:
        for symbol2 in symbols:
            series1 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' +
                                  symbol1 + '.csv')
            series2 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' +
                                  symbol2 + '.csv')
            series1 = np.array(series1[str(len(series1.columns) - 2)])
            series2 = np.array(series2[str(len(series2.columns) - 2)])
            series = series1 / series2
            cocientes[symbol1, symbol2] = max(list(series)) - min(list(series))
    np.save('cocientes.npy', cocientes)


def short_range():
    cocientes = np.load('cocientes.npy', allow_pickle='TRUE').item()
    for alpha in [0.001, 0.01, 0.05]:
        print(alpha)
        short = []
        for key in cocientes.keys():
            if cocientes[key] <= alpha:
                short.append(key)
        short = pd.DataFrame(short)
        short.to_csv('/home/angel/Desktop/Sectorizacion/' + str(alpha) +
                     '.csv')


def cociente(symbol1, symbol2):
    series1 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol1 +
                          '.csv')
    series2 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol2 +
                          '.csv')
    residuo1 = np.array(series1[str(len(series1.columns) - 2)])
    residuo2 = np.array(series2[str(len(series2.columns) - 2)])
    return residuo1 / residuo2


def amplitud(symbol1, symbol2):
    series = cociente(symbol1, symbol2)
    return max(list(series)) - min(list(series))


def new_strategy(symbol1, symbol2, flag=False):
    # print(symbol1 + '-' + symbol2)
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    first_date, last_date = fistdate_lastdate()
    stock1 = pd.read_csv(path + symbol1 + ".csv")
    stock2 = pd.read_csv(path + symbol2 + ".csv")
    stock1 = np.array(stock1['close'][np.where(
        np.array(stock1['date']) == first_date)[0][0]:np.where(
            np.array(stock1['date']) == last_date)[0][0]])
    stock2 = np.array(stock2['close'][np.where(
        np.array(stock2['date']) == first_date)[0][0]:np.where(
            np.array(stock2['date']) == last_date)[0][0]])
    beta = cociente(symbol1, symbol2)
    corr = np.corrcoef(stock1, beta * stock2)[0][1]
    # print(corr)
    axisx = np.arange(len(stock1 - stock2))
    if flag:
        cadftest(stock1, beta * stock2, flag)
        plt.plot(axisx, stock1 - corr * (beta * stock2))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(symbol1 + '-' + symbol2)
        plt.show()
    series1 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol1 +
                          '.csv')
    series2 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol2 +
                          '.csv')
    residuo1 = np.array(series1[str(len(series1.columns) - 2)])
    residuo2 = np.array(series2[str(len(series2.columns) - 2)])
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df[symbol1] = stock1
    df[symbol2] = beta * stock2
    df1[symbol1] = residuo1
    df1[symbol2] = residuo2
    df2[symbol1] = stock1
    df2[symbol2] = stock2
    return df, df1, df2


def best_pvalues():
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    symbols = pd.read_csv(path + "sp500_symbols.csv")['symbol'][:100]
    best = []
    for symbol1 in symbols:
        for symbol2 in symbols:
            if symbol1 != symbol2:
                if new_strategy(symbol1, symbol2) <= 0.05:
                    best.append((symbol1, symbol2))
    best = pd.DataFrame(np.array(best))
    best.to_csv('/home/angel/Desktop/Sectorizacion/best_pvalues.csv')


def verificacion():
    # cocientes = np.load('cocientes.npy', allow_pickle='TRUE').item()
    best_pvalues_x = pd.read_csv(
        "/home/angel/Desktop/Sectorizacion/best_pvalues.csv")['0']
    best_pvalues_y = pd.read_csv(
        "/home/angel/Desktop/Sectorizacion/best_pvalues.csv")['1']
    crit = new_strategy(best_pvalues_x[0], best_pvalues_y[0])
    for ind in range(1, len(pd.DataFrame(best_pvalues_x).index)):
        if np.all(
                crit != new_strategy(best_pvalues_x[ind], best_pvalues_y[ind])
        ):
            return False
    return True


def price_spread(symbol1, symbol2, flag=False):
    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2, flag)
    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df
    yport = np.sum(yport, 1)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(symbol1 + "-" + symbol2)
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')
    # axis_x = np.arange(len(imf))
    # plt.plot(axis_x, imf)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('IMF')
    # plt.show()

    # plt.savefig('USO_GLD.png')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    # SE PROCEDE A CALCULAR EL INDICE DE SHARPE Y EL APR

    # SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # RAZON DE SHARPE Y APR
    APR1 = np.prod(1 + ret)**(252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    # print('La razon de Sharpe es de: ' + str(sharpe))

    # print(APR1)

    #############################################

    hedgeratio = np.ones(
        (len(df), 1)
    )  # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        res = sm.OLS(
            df.apply(log)[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df.apply(log)[symbol1][i - lookback + 1:i +
                                                   1])).fit()
        hedgeratio[i] = res.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df.apply(log)
    yport = np.sum(yport, 1)
    hedgeratio[0:20] = np.zeros((20, 1))
    yport[0:20] = np.zeros(20)
    s = (lookback, 2)
    df.values[0:20, :] = np.zeros(s)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(''' Spread = USO-hedgeratio*GLD (usando logaritmo) \n''')
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # RAZON DE SHARPE Y APR
    APR2 = np.prod(1 + ret)**(252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    return [APR1, APR2]


def price_spread_new_corr(symbol1, symbol2, flag=False):
    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2, flag)
    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df1[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df1[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df
    yport = np.sum(yport, 1)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(symbol1 + "-" + symbol2)
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')
    # axis_x = np.arange(len(imf))
    # plt.plot(axis_x, imf)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('IMF')
    # plt.show()

    # plt.savefig('USO_GLD.png')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    # SE PROCEDE A CALCULAR EL INDICE DE SHARPE Y EL APR

    # SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # RAZON DE SHARPE Y APR
    APR1 = np.prod(1 + ret)**(252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    # print('La razon de Sharpe es de: ' + str(sharpe))

    # print(APR1)

    #############################################

    hedgeratio = np.ones(
        (len(df), 1)
    )  # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        res = sm.OLS(
            df1.apply(log)[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df1.apply(log)[symbol1][i - lookback + 1:i +
                                                    1])).fit()
        hedgeratio[i] = res.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df.apply(log)
    yport = np.sum(yport, 1)
    hedgeratio[0:20] = np.zeros((20, 1))
    yport[0:20] = np.zeros(20)
    s = (lookback, 2)
    df.values[0:20, :] = np.zeros(s)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(''' Spread = USO-hedgeratio*GLD (usando logaritmo) \n''')
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # RAZON DE SHARPE Y APR
    APR2 = np.prod(1 + ret)**(252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    return [APR1, APR2]


def price_spread_n(symbol1, symbol2, flag=False):
    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2, flag)
    df = df2
    # df1 = df2
    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df1[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df1[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df
    yport = np.sum(yport, 1)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(symbol1 + "-" + symbol2)
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')
    # axis_x = np.arange(len(imf))
    # plt.plot(axis_x, imf)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('IMF')
    # plt.show()

    # plt.savefig('USO_GLD.png')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    # SE PROCEDE A CALCULAR EL INDICE DE SHARPE Y EL APR

    # SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # RAZON DE SHARPE Y APR
    APR1 = np.prod(1 + ret)**(252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    # print('La razon de Sharpe es de: ' + str(sharpe))

    # print(APR1)

    #############################################

    hedgeratio = np.ones(
        (len(df), 1)
    )  # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        res = sm.OLS(
            df1.apply(log)[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df1.apply(log)[symbol1][i - lookback + 1:i +
                                                    1])).fit()
        hedgeratio[i] = res.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df.apply(log)
    yport = np.sum(yport, 1)
    hedgeratio[0:20] = np.zeros((20, 1))
    yport[0:20] = np.zeros(20)
    s = (lookback, 2)
    df.values[0:20, :] = np.zeros(s)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(''' Spread = USO-hedgeratio*GLD (usando logaritmo) \n''')
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # RAZON DE SHARPE Y APR
    APR2 = np.prod(1 + ret)**(252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    return [APR1, APR2]


def std_stocks():
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    std_cocientes = {}
    symbols = pd.read_csv(path + "sp500_symbols.csv")['symbol'][:100]
    for symbol1 in symbols:
        for symbol2 in symbols:
            if symbol1 != symbol2:
                std_cocientes[symbol1,
                              symbol2] = np.std(cociente(symbol1, symbol2))
    np.save('std_cocientes.npy', std_cocientes)


def normas(symbol1, symbol2):
    series1 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol1 +
                          '.csv')
    series2 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol2 +
                          '.csv')
    series1 = np.array(series1[str(len(series1.columns) - 2)])
    series2 = np.array(series2[str(len(series2.columns) - 2)])
    return [
        np.std((series1 / LA.norm(series1)) / (series2 / LA.norm(series2))),
        np.std(
            (series1 / sum(list(series1))) / (series2 / sum(list(series2)))),
        np.std((series1 / max(list(series1))) / (series2 / max(list(series2))))
    ]


def correlaciones(symbol1, symbol2):
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    first_date, last_date = fistdate_lastdate()
    stock1 = pd.read_csv(path + symbol1 + ".csv")
    stock2 = pd.read_csv(path + symbol2 + ".csv")
    stock1 = np.array(stock1['close'][np.where(
        np.array(stock1['date']) == first_date)[0][0]:np.where(
            np.array(stock1['date']) == last_date)[0][0]])
    stock2 = np.array(stock2['close'][np.where(
        np.array(stock2['date']) == first_date)[0][0]:np.where(
            np.array(stock2['date']) == last_date)[0][0]])
    beta = cociente(symbol1, symbol2)
    return [
        np.corrcoef(stock1, beta * stock2)[0][1],
        np.corrcoef(stock1, stock2)[0][1]
    ]


def bollinger(symbol1, symbol2, flag=False):
    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2, flag)
    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df
    yport = np.sum(yport, 1)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(symbol1 + "-" + symbol2)
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')
    # axis_x = np.arange(len(imf))
    # plt.plot(axis_x, imf)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('IMF')
    # plt.show()

    # plt.savefig('USO_GLD.png')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std

    entryZscore = 0.7
    exitZscore = 0

    # CONDICIONES PARA POSICIONES EN LARGO
    longsEntry = z_score < -entryZscore
    longsExit = z_score > -exitZscore

    # CONDICIONES PARA POSICIONES EN CORTO
    shortsEntry = z_score > entryZscore
    shortsExit = z_score < exitZscore

    # SE CREAN ARREGLOS AUX PARA CALCULAR UNIDADES EN LARGO Y EN CORTO (BOLLINGER.M ARCHIVO MATLAB)
    numUnitsLong = np.ones(len(yport)) * float('nan') * yport
    numUnitsShort = np.ones(len(yport)) * float('nan') * yport

    numUnitsLong[0] = 0
    numUnitsLong[longsEntry] = 1
    numUnitsLong[longsExit] = 0
    numUnitsLong = numUnitsLong.fillna(method='pad')

    numUnitsShort[0] = 0
    numUnitsShort[shortsEntry] = -1
    numUnitsShort[shortsExit] = 0
    numUnitsShort = numUnitsShort.fillna(method='pad')

    numUnits = numUnitsLong + numUnitsShort

    # SE CALCULAN LAS POSICIONES
    CC = np.transpose(repmat(numUnits, 2, 1))
    BB = sm.add_constant(-hedgeratio, prepend=False)
    position = BB * CC * df2

    # SE CALCULA LAS UTILIDADES Y PERDIDAS DIARIAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # SE CALCULA EL APR Y LA RAZON DE SHARPE
    APR = np.prod(1 + ret)**(252 / len(ret)) - 1
    return APR


def bollinger_new(symbol1, symbol2, flag=False):
    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2, flag)
    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df1[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df1[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df
    yport = np.sum(yport, 1)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(symbol1 + "-" + symbol2)
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')
    # axis_x = np.arange(len(imf))
    # plt.plot(axis_x, imf)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('IMF')
    # plt.show()

    # plt.savefig('USO_GLD.png')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std

    entryZscore = 0.7
    exitZscore = 0

    # CONDICIONES PARA POSICIONES EN LARGO
    longsEntry = z_score < -entryZscore
    longsExit = z_score > -exitZscore

    # CONDICIONES PARA POSICIONES EN CORTO
    shortsEntry = z_score > entryZscore
    shortsExit = z_score < exitZscore

    # SE CREAN ARREGLOS AUX PARA CALCULAR UNIDADES EN LARGO Y EN CORTO (BOLLINGER.M ARCHIVO MATLAB)
    numUnitsLong = np.ones(len(yport)) * float('nan') * yport
    numUnitsShort = np.ones(len(yport)) * float('nan') * yport

    numUnitsLong[0] = 0
    numUnitsLong[longsEntry] = 1
    numUnitsLong[longsExit] = 0
    numUnitsLong = numUnitsLong.fillna(method='pad')

    numUnitsShort[0] = 0
    numUnitsShort[shortsEntry] = -1
    numUnitsShort[shortsExit] = 0
    numUnitsShort = numUnitsShort.fillna(method='pad')

    numUnits = numUnitsLong + numUnitsShort

    # SE CALCULAN LAS POSICIONES
    CC = np.transpose(repmat(numUnits, 2, 1))
    BB = sm.add_constant(-hedgeratio, prepend=False)
    position = BB * CC * df2

    # SE CALCULA LAS UTILIDADES Y PERDIDAS DIARIAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # SE CALCULA EL APR Y LA RAZON DE SHARPE
    APR = np.prod(1 + ret)**(252 / len(ret)) - 1
    return APR


def bollinger_n(symbol1, symbol2, flag=False):
    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2, flag)
    df = df2
    # df1 = df2
    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df1[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df1[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df
    yport = np.sum(yport, 1)

    # plt.figure()
    # yport.plot(x='timestamp', y=yport.values)
    # plt.title(symbol1 + "-" + symbol2)
    # plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
    # plt.ylabel('USO/GLD  \n')
    # axis_x = np.arange(len(imf))
    # plt.plot(axis_x, imf)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('IMF')
    # plt.show()

    # plt.savefig('USO_GLD.png')

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std

    entryZscore = 0.7
    exitZscore = 0

    # CONDICIONES PARA POSICIONES EN LARGO
    longsEntry = z_score < -entryZscore
    longsExit = z_score > -exitZscore

    # CONDICIONES PARA POSICIONES EN CORTO
    shortsEntry = z_score > entryZscore
    shortsExit = z_score < exitZscore

    # SE CREAN ARREGLOS AUX PARA CALCULAR UNIDADES EN LARGO Y EN CORTO (BOLLINGER.M ARCHIVO MATLAB)
    numUnitsLong = np.ones(len(yport)) * float('nan') * yport
    numUnitsShort = np.ones(len(yport)) * float('nan') * yport

    numUnitsLong[0] = 0
    numUnitsLong[longsEntry] = 1
    numUnitsLong[longsExit] = 0
    numUnitsLong = numUnitsLong.fillna(method='pad')

    numUnitsShort[0] = 0
    numUnitsShort[shortsEntry] = -1
    numUnitsShort[shortsExit] = 0
    numUnitsShort = numUnitsShort.fillna(method='pad')

    numUnits = numUnitsLong + numUnitsShort

    # SE CALCULAN LAS POSICIONES
    CC = np.transpose(repmat(numUnits, 2, 1))
    BB = sm.add_constant(-hedgeratio, prepend=False)
    position = BB * CC * df2

    # SE CALCULA LAS UTILIDADES Y PERDIDAS DIARIAS
    pnl = (position.shift(1) * df2.diff(1)) / df2.shift(1)
    pnl = pnl.fillna(value=0)
    pnl = np.sum(pnl, 1)

    # SE CALCULA EL VALOR EN EL MERCADO
    mrk_val = position.shift(1)
    mrk_val = mrk_val.apply(abs)
    mrk_val = mrk_val.fillna(value=0)
    mrk_val = np.sum(mrk_val, 1)

    # SE CALCULAN LOS RENDIMIENTOS
    ret = pnl / mrk_val
    ret = ret.fillna(value=0)

    # SE CALCULA EL APR Y LA RAZON DE SHARPE
    APR = np.prod(1 + ret)**(252 / len(ret)) - 1
    return APR


if __name__ == '__main__':
    # print(verificacion())
    # best_pvalues()
    # cociente()
    # print(short_range())
    # compute_ceemdan()
    # print(read_dictionary)
    # new_series_emd, series_1 = create_new_time_series_emd('TSLA', 'AAPL')
    # stock = create_new_time_series_emd('TSLA', 'AAPL')
    # axis_x = np.arange(len(stock))
    # plt.plot(axis_x, stock)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('TSLA - AAPL')
    # plt.show()
    # PATH = '/home/angel/Desktop/Datos/sp500-1day/'
    # symbols = pd.read_csv(PATH + "sp500_symbols.csv")['symbol'][:100]
    pvalues_x = pd.read_csv(
        '/home/angel/Desktop/Sectorizacion/best_pvalues.csv')['0']
    pvalues_y = pd.read_csv(
        '/home/angel/Desktop/Sectorizacion/best_pvalues.csv')['1']
    # std_coc = np.load('std_cocientes.npy', allow_pickle='TRUE').item()
    # APR = {}
    # APR = []
    # for k in range(len(pvalues_x)):
    #    prov = price_spread_n(pvalues_x[k], pvalues_y[k])
    #    prov.append(bollinger_n(pvalues_x[k], pvalues_y[k]))
    #    APR.append(prov)
    # APR = pd.DataFrame(APR)
    # pvalues_df6 = pd.read_csv(
    #    '/home/angel/Desktop/Sectorizacion/pvalues_df6.csv')
    # pvalues_df7 = pd.concat(
    #    [pvalues_df6, APR],
    #    axis=1,
    # )
    # np.save('APR.npy', APR)
    # pvalues_df = pd.DataFrame(pvalues_df)
    # pvalues_df7.to_csv('/home/angel/Desktop/Sectorizacion/pvalues_df7.csv')

    # series_1 = np.array(pd.read_csv(PATH + 'TSLA' + ".csv")['close'])
    # print(stock0)
    # axisx = np.arange(len(new_series_emd))
    # graph_of_imfs_ceemdan('TSLA', True)
    # hurst_exponent_adftest(new_series_emd)
    # hurst_exponent_adftest(new_series_ceemdan)
    # cadftest(series_1, new_series_emd)
    # cadftest(series_1, new_series_ceemdan)
    # print('###########################')
    # new_strategy('AAPL', 'AMZN')
    # print('###########################')
    # new_strategy('BAC', 'AMZN')
    # print('###########################')
    # new_strategy('INTC', 'ANTM')
    # print('###########################')
    # new_strategy('T', 'BKNG')
    # print('###########################')
    # new_strategy('PFE', 'HON')
    # cocientes = np.load('cocientes.npy', allow_pickle='TRUE').item()
    # best_pvalues_x = pd.read_csv(
    #    "/home/angel/Desktop/Sectorizacion/best_pvalues.csv")['0']
    # best_pvalues_y = pd.read_csv(
    #    "/home/angel/Desktop/Sectorizacion/best_pvalues.csv")['1']
    # best_APR = []
    # for k in range(len(best_pvalues_x)):
    #    apr = price_spread(best_pvalues_x[k], best_pvalues_y[k])
    #    if apr > 0.0:
    #        best_APR.append([(best_pvalues_x[k], best_pvalues_y[k]), apr])
    # print(cocientes[best_pvalues_x[1], best_pvalues_y[1]])
    # print(best_APR)
    # best_APR = pd.DataFrame(best_APR)
    # best_APR.to_csv('/home/angel/Desktop/Sectorizacion/best_APR.csv')
    # best_APR = pd.read_csv(
    #    '/home/angel/Desktop/Sectorizacion/best_APR.csv')['0']
    # best_APR_y = pd.read_csv(
    #    '/home/angel/Desktop/Sectorizacion/best_APR.csv')['1']
    k = 0
    # print(cocientes[best_pvalues_x[k], best_pvalues_y[k]])
    # print(best_APR[k][2])
    # print(price_spread(best_APR_x[k], best_APR_y[k], True))
    # new_strategy(best_pvalues_x[k], best_pvalues_y[k], True)
    # print(price_spread_new_corr(pvalues_x[k], pvalues_y[k]))
    df = read_ods('/home/angel/Desktop/Sectorizacion/pvalues_dfbest.ods')
    lista = ['amplitud_cociente', 'std_cociente_norma_euclidiana']
    lista0 = [
        'Spread_1', 'Spread_2', 'Spread_3', 'Log_1', 'Log_2', 'Log_3',
        'Bollinger_1', 'Bollinger_2', 'Bollinger_3', 'spread', 'log',
        'bollinger'
    ]
    # k = 8
    # prov_1 = []
    # for k in [0, 1, 2, 9]:
    #    print(k)
    #    prov1 = []
    #    for j in [2, 1]:
    #        prov1.append((len(df[lista0[k]][df[lista0[k]] > df[lista0[j]]])) /
    #                     len(df[lista0[k]]))
    #    prov_1.append(prov1)
    # prov_1 = pd.DataFrame(prov_1)
    # prov_1.to_csv('/home/angel/Desktop/Sectorizacion/prov_1.csv')
    mos = list(df['std_cociente_norma_euclidiana'])
    plt.title('std_cociente_norma_euclidiana')
    plt.hist(mos, bins=100)
    plt.grid(True)
    plt.show()
    plt.clf()
