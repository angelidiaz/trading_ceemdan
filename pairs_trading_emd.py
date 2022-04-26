#! ./venv/bin/python3.8
# -*- coding: utf-8 -*-

'''Pairs Trading Strategies using the CEEMDAN'''

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


def graph_of_imfs(series, imfs=[]):
    """
    Shows the graphs of IMFs, one by one, of the series and the graph of the series too.

    PARAMETER
    ---------
    series : numpy-array

    imfs : numpy-array
          The IMFs of the series

    RETURN
    ------
        None
        Show the graphs of IMFs
    """
    axis_x = np.arange(len(series))
    plt.plot(axis_x, series)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    for imf in imfs:
        axis_x = np.arange(len(imf))
        plt.plot(axis_x, imf)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('IMF')
        plt.show()


def hurst_exponent_adftest(series):
    """
    Prints the hurst exponent, variance-test ratio
    and the half life of a time series

    PARAMETER
    ---------
    series : numpy-array

    RETURN
    ------
        None
        print the hurst exponent
    """
    series = pd.DataFrame(series)
    X = series.Value
    Y = sts.adfuller(series, maxlag=1)

    # test ADF con sus valores criticos
    print('test estadístico ADF : %f' % Y[0])
    print('Valores criticos:')
    for key, value in Y[4].items():
        print('\t%s: %.3f' % (key, value))

    # Exponente de Hurst
    H, c, data = compute_Hc(series, kind='change', simplified=True)
    print('Exponente de Hurst:' "{:.4f}".format(H))

    # test de ratio de varianza
    vr = VarianceRatio(log(series))

    print(vr.summary().as_text())

    # Vida media
    series['Value_lagged'] = series['Value'].shift(1)
    series['delta'] = series['Value'] - series['Value_lagged']

    results = smf.ols('delta ~ Value_lagged', data=series).fit()
    lam = results.params['Value_lagged']

    halflife = -np.log(2) / lam
    print('la vida media es de %f dias' % halflife)


def cadftest(series1, series2, flag=False):
    """
    Return the CADF test of two series and print critical values
    if flag is True

    PARAMETER
    ---------
    series1 : numpy array

    series2 : numpy array

    RETURN
    ------
        p-values of the CADF test.
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


def amplitude_and_std_of_quotient(residue1, residue2):
    """
    Return the amplitude of the quotient residue1/residue2 and the Standard Deviation of the quotient
    residue1/residue2, (r2*residue1)/(r1*residue2) where r1 and r2 are the norm of residue1 and residue2, respectively,
    using the supreme norm, euclidean norm and L1-norm.

    PARAMETER
    ---------
    residue1 : numpy array

    residue2 : numpy array

    RETURN
    ------
        A list with the amplitude and the STDs.
    """
    quotient = residue1/residue2
    return [
        max(list(quotient)) - min(list(quotient)),
        np.std((residue1 / LA.norm(residue1)) /
               (residue2 / LA.norm(residue2))),
        np.std((residue1 / sum(list(residue1))) /
               (residue2 / sum(list(residue2)))),
        np.std((residue1 / max(list(residue1))) /
               (residue2 / max(list(residue2))))
    ]


def price_spread(df, df0, df1, flag=False):
    """
    Trading Pairs Strategy using Price Spreads. For more information on this strategy, read:
    Chan, E. (2013). Algorithmic trading: winning strategies and their rationale (Vol. 625). John Wiley
    & Sons, chapter 3.

    PARAMETER
    ---------
    df : pandas.DataFrame
        A DataFrame containing the original time series of two stocks.

    df0 : pandas.DataFrame
        A DataFrame containing two time series that are derivative from series of the DataFrame df. They will be used
        to compute the hedge ratio.

    df1 : pandas.DataFrame
        A DataFrame containing two time series that are derivative from series of the DataFrame df. They will be used
        to compute the spreads of the strategy.

    flag : boolean
        If flag == True, it shows the graph of spreads.

    RETURN
    ------
        APR and Sharpe ratio
    """
    symbol1 = list(df)[0]
    symbol2 = list(df)[1]

    lookback = 20

    ### ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS ###

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df0[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df0[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df1
    yport = np.sum(yport, 1)

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    # SE PROCEDE A CALCULAR EL INDICE DE SHARPE Y EL APR

    # SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df.diff(1)) / df.shift(1)
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
    APR = np.prod(1 + ret) ** (252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    if flag:
        plt.figure()
        yport.plot(x='timestamp', y=yport.values)
        plt.title('Price spread' + symbol1 + "-" + symbol2)
        plt.show()

    return APR, sharpe


def price_spread_log(df, df0, df1, flag=False):
    """
    Trading Pairs Strategy using Log Price Spreads. For more information on this strategy, read:
    Chan, E. (2013). Algorithmic trading: winning strategies and their rationale (Vol. 625). John Wiley
    & Sons, chapter 3.

    PARAMETER
    ---------
    df : pandas.DataFrame
        A DataFrame containing the original time series of two stocks.

    df0 : pandas.DataFrame
        A DataFrame containing two time series that are derivative from series of the DataFrame df. They will be used
        to compute the hedge ratio.

    df1 : pandas.DataFrame
        A DataFrame containing two time series that are derivative from series of the DataFrame df. They will be used
        to compute the spreads of the strategy.

    flag : boolean
        If flag == True, it shows the graph of spreads.

    RETURN
    ------
        APR and Sharpe ratio

    """

    symbol1 = list(df)[0]
    symbol2 = list(df)[1]

    lookback = 20

    ## ESTRATEGIA PARA EL SPREAD DE LOS LOG PRECIOS ##

    hedgeratio = np.ones(
        (len(df), 1)
    )  # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        res = sm.OLS(
            df0.apply(log)[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df0.apply(log)[symbol1][i - lookback + 1:i +
                                                                     1])).fit()
        hedgeratio[i] = res.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df1.apply(log)
    yport = np.sum(yport, 1)
    hedgeratio[0:20] = np.zeros((20, 1))
    yport[0:20] = np.zeros(20)
    s = (lookback, 2)
    df.values[0:20, :] = np.zeros(s)

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df.diff(1)) / df.shift(1)
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
    APR = np.prod(1 + ret) ** (252 / len(ret)) - 1
    # print('\nEl valor del APR es:' + str(APR))
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    if flag:
        plt.figure()
        yport.plot(x='timestamp', y=yport.values)
        plt.title('Log price spread' + symbol1 + "-" + symbol2)
        plt.show()

    return APR, sharpe


def bollinger_bands(df, df0, df1, flag=False):
    """
    Trading Pairs Strategy using Bollinger Bands. For more information on this strategy, read:
    Chan, E. (2013). Algorithmic trading: winning strategies and their rationale (Vol. 625). John Wiley
    & Sons, chapter 3.

    PARAMETER
    ---------
    df : pandas.DataFrame
        A DataFrame containing the original time series of two stocks.

    df0 : pandas.DataFrame
        A DataFrame containing two time series that are derivative from series of the DataFrame df. They will be used
        to compute the hedge ratio.

    df1 : pandas.DataFrame
        A DataFrame containing two time series that are derivative from series of the DataFrame df. They will be used
        to compute the spreads of the strategy.

    flag : boolean
        If flag == True, it shows the graph of spreads.

    RETURN
    ------
        APR and Sharpe ratio
    """

    symbol1 = list(df)[0]
    symbol2 = list(df)[1]

    lookback = 20

    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df0[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df0[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df1
    yport = np.sum(yport, 1)

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

    # SE CREAN ARREGLOS AUX PARA CALCULAR UNIDADES EN LARGO Y EN CORTO
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
    position = BB * CC * df

    # SE CALCULA LAS UTILIDADES Y PERDIDAS DIARIAS
    pnl = (position.shift(1) * df.diff(1)) / df.shift(1)
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
    APR = np.prod(1 + ret) ** (252 / len(ret)) - 1
    sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret)

    if flag:
        plt.figure()
        yport.plot(x='timestamp', y=yport.values)
        plt.title('Bollinguer bandas' + symbol1 + "-" + symbol2)
        plt.show()

    return APR, sharpe


def new_strategies(df, df_residues):
    """
    4 trading strategies using the CEEMDAN are applied to the Traditional Pairs Trading strategies, described on this
    reference: Chan, E. (2013). Algorithmic trading: winning strategies and their rationale (Vol. 625). John Wiley
    & Sons, chapter 3.
    The strategies are:

    Strategy 1: We apply the traditional strategy to time series S1 and S2 in the DataFrame df.

    Strategy 2: We apply the traditional strategy to time series S1 and S2 in the DataFrame df. To compute the
    hedge ratios, we use the residues r1 and r2 in the DataFrame df_residues, which correspond to the gotten
    residues applying the CEEMDAN to series S1 and S2, respectively.

    Strategy 3: We apply the traditional strategy to time series S1 and S2 in the DataFrame df. To compute the
    hedge ratios and the spreads, we make a new time series S3 = q*S2, where q = r1/r2 and where r1 and r2 in
    the DataFrame df_residues, which correspond to the gotten residues applying the CEEMDAN to series S1 and S2,
    respectively. We use the time series S1 and S3 to calculate the hedge ratios and the spreads.

    Strategy 4: We apply the traditional strategy to time series S1 and S2 in the DataFrame df. To compute the
    hedge ratios, we use the residues r1 and r2 in the DataFrame df_residues, which correspond to the gotten
    residues applying the CEEMDAN to series S1 and S2 respectively. To compute the spreads, we make a new time
    series S3 = q*S2, where q = r1/r2; we use the time series S1 and S3 to calculate the spreads.

    PARAMETER
    ---------
    df : pandas.DataFrame
        a DataFrame containing two time series.

    df_residues: pandas.DataFrame
        a DataFrame containing the gotten residues applying CEEMDAN to the time series in df.

    RETURN
    ------
        a DataFrame containing the APR and ratio sharpe of each strategy.
    """

    symbol1 = list(df)[0]
    symbol2 = list(df)[1]

    residue1 = np.array(df_residues[symbol1])
    residue2 = np.array(df_residues[symbol2])

    series1 = np.array(df[symbol1])
    series2 = np.array(df[symbol2])

    quotient = residue1 / residue2

    df0 = pd.DataFrame()
    df0[symbol1] = series1
    df0[symbol2] = quotient * series2

    strategies = [(df, df, df), (df, df_residues, df),
                  (df, df0, df0), (df, df_residues, df0)]

    results = pd.DataFrame()
    cont = 1

    for strategy in strategies:
        results[f"strategy {cont}"] = [price_spread(strategy[0], strategy[1], strategy[2]),
                                       price_spread_log(strategy[0], strategy[1], strategy[2]),
                                       bollinger_bands(strategy[0], strategy[1], strategy[2])]
        cont += 1
    results.index = ['Price Spread', 'Log Price Spread', 'Bollinger Bands']

    return results


if __name__ == "__main__":
    series1 = pd.read_csv("AAPL.csv")['close']
    series1_imfs = pd.read_csv("AAPL_imf.csv")
    residue1 = series1_imfs[list(series1_imfs)[len(list(series1_imfs)) - 1]]
    series2 = pd.read_csv("TSLA.csv")['close']
    series2_imfs = pd.read_csv("AAPL_imf.csv")
    residue2 = series2_imfs[list(series2_imfs)[len(list(series2_imfs)) - 1]]

    df = pd.DataFrame()
    df_residues = pd.DataFrame()
    df['AAPL'] = series1
    df['TSLA'] = series2
    df_residues['AAPL'] = residue1
    df_residues['TSLA'] = residue2

    results = new_strategies(df, df_residues)

    print(results)

