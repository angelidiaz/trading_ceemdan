#! ./venv/bin/python3.8
# -*- coding: utf-8 -*-
"""
Pairs Trading Strategies using the CEEMDAN
"""

import copy
import pandas as pd
import numpy as np
# from pandas import read_csv
# from numpy import cumsum, polyfit, sqrt, std, subtract, var, seterr
from numpy import log
# from numpy.random import randn
# import matplotlib.pyplot as plt
# from PyEMD import EMD
# from PyEMD import CEEMDAN
import statsmodels.tsa.stattools as sts
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as smf
# import decimal
# import quandl
from numpy import linalg as la
from arch.unitroot import VarianceRatio
from hurst import compute_Hc
# import statsmodels.tsa.vector_ar.vecm as vm
import datetime
# import http.client
# from alpha_vantage.timeseries import TimeSeries
# from pprint import pprint
import matplotlib.pyplot as plt
# import statsmodels.formula.api as sm
# from numpy.matlib import repmat
# import math
# from numpy.linalg import inv, eig, cholesky as chol
# import csv
# from statsmodels.regression.linear_model import OLS
# import statsmodels.tsa.tsatools as tsat
# from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
# from pandas_ods_reader import read_ods
# from series import time_series


# Program constants
LOOK_BACK = 20


def graph_of_imfs(series, imfs=None):
    """
    Shows the graphs of IMFs, one by one, of the series and the graph of the
    series too.

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
    if imfs is None:
        return
    for imf in imfs:
        axis_x = np.arange(len(imf))
        plt.plot(axis_x, imf)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('IMF')
        plt.show()


def hurst_exponent_adf_test(series, flag=False):
    """
    Computes media reversion parameters of 'series': the hurst exponent,
    the adf test, the variance-test ratio, the linear regression parameters and
    the half life.

    Parameters
    ---------
    series : numpy-array
        containg the series to test.

    Return
    ------
        dict
        With the following keys:
        adf statistic: Augmented Dickey Fuller Test results.
        hurst_exponent: The Hurst exponent.
        linear regression: Linear regression results.
        half life: Half life reversion time.

    Examples
    --------
    >>> aux = time_series.read_ts_from_ibdb('AAPL', '1 day', None, None,
    ...                                     last=1000, imf=True)
    >>> hurst_exponent_adf_test(np.array(aux['adj_close']))
    """
    result = dict()
    series = pd.DataFrame(series, columns=['Value'])
    adf_result = tuple(sts.adfuller(series, maxlag=1))

    # ADF test and critical values.

    result['adf statistic'] = adf_result[0]

    # Exponente de Hurst
    result['hurst exponent'], c, data = \
        compute_Hc(series, kind='change')


    # test de ratio de varianza
    vr = VarianceRatio(series)
    result['variance rate'] = vr.summary()

    # Vida media
    series['Value_lagged'] = series['Value'].shift(1)
    series['delta'] = series['Value'] - series['Value_lagged']

    result['linear regression'] = \
        smf.ols('delta ~ Value_lagged', data=series).fit()
    lam = result['linear regression'].params['Value_lagged']

    result['half life'] = -np.log(2) / lam

    if flag:
        print('ADF statistic : %f' % adf_result[0])
        print('Critical values:')
        for key, value in adf_result[4].items():
            print('\t%s: %.3f' % (key, value))
        print('Exponente de Hurst:' "{:.4f}".format(result['hurst exponent']))
        print(vr.summary().as_text())
        print('la vida media es de %f dias' % result['half life'])

    return result


def cadf_test(series1, series2, flag=False):
    """
    Returns the CADF test of two series and print critical values when flag is
    True

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
    cointegration_t, pvalue, crit_value = ts.coint(series1, series2)
    if flag:
        print('t-statistic = %f' % cointegration_t)
        print('pvalue = %f' % pvalue)
        print('CADV critical values')
        print(crit_value)
    return cointegration_t


def amplitude_and_std_of_quotient(residue1, residue2):
    """
    Returns the amplitude of the quotient residue1/residue2 and the Standard
    Deviation of the quotient residue1/residue2, (r2*residue1)/(r1*residue2)
    where r1 and r2 are the norm of residue1 and residue2, respectively,
    using the supreme norm, euclidean norm and L1-norm.

    PARAMETER
    ---------
    residue1 : numpy array

    residue2 : numpy array

    RETURN
    ------
        A list with the amplitude and the STDs.
    """
    quotient = residue1 / residue2
    return [
        max(list(quotient)) - min(list(quotient)),
        np.std(
            (residue1 / la.norm(residue1)) / (residue2 / la.norm(residue2))),
        np.std((residue1 / sum(list(residue1))) /
               (residue2 / sum(list(residue2)))),
        np.std((residue1 / max(list(residue1))) /
               (residue2 / max(list(residue2))))
    ]

def kalman_filter(df):
    """
    Implementation of the Kalman Filter.

    PARAMETER
    ---------
    df : pandas.DataFrame
        A DataFrame containing time series of two stocks.

    RETURN
    ------
        dict
        Containing:
        df: A copy of df.
        hedge: Predicted hedge ratio.
        mean: Predicted mean value
        variance: Predicted variance value
        error: The measurement prediction error
    """
    # SE CREA EL ARREGLO x AUMENTADO PARA LOS POSIBLES VALORES DE LA REGRESION y
    x = sm.add_constant(df[list(df)[0]], prepend=False)
    y = df[list(df)[1]]
    size = (len(df), 1)
    delta = 0.0001

    # MEDIDAS DE PREDICCION
    yhat = np.ones(size) * float('nan')

    # MEDIDAS DE PREDICCION DEL ERROR
    e = np.ones(size) * float('nan')

    # MEDIDAS DE PREDICCION DE LA VARIANZA DEL ERROR
    Q = np.ones(size) * float('nan')

    # SE INICIALIZA R,P Y BETA
    R = np.zeros((2, 2))  # R = R(t+1|t)
    P = np.zeros((2, 2))  # P = R(t|t)
    beta = np.zeros((2, len(df))) * float('nan')  # beta(t|t-1) = beta(t|t)
    Vw = (delta / (1 - delta)) * np.identity(2)
    Ve = 0.001

    # SE INICIALIZA BETA[:,0] CON VALORES 0
    beta[:, 0] = 0

    # DADOS VALORES INICIALES DE BETA Y R (Y P)
    for i in range(len(df)):
        if i > 0:
            beta[:, i] = beta[:, i - 1]  # ESTADO DE PREDICCION
            R = P + Vw  # ESTADO DE PREDICCION COVARIANZA
        yhat[i] = np.matmul(x.values[i, :], beta[:, i])  # MEDIDA DE PREDICCION
        Q[i] = np.matmul(np.matmul(x.values[i, :], R),
                         np.transpose(x.values[i, :])) + Ve  # MEDIDA DE PREDICCION VARIANZA
        e[i] = y[i] - yhat[i]  # MEDIDA DE PREDICCION DEL ERROR
        K = np.matmul(R, np.transpose(x.values[i, :])) / Q[i]  # GANANCIA KALMAN
        beta[:, i] = beta[:, i] + K * e[i]  # ACTUALIZACION ESTADO
        K_aux = K.reshape(2, 1)
        x_aux = x.values[i, :].reshape(1, 2)
        P = R - np.matmul(np.matmul(K_aux, x_aux), R)  # ACTUALIZACION ESTADO COVARIANZA
    result = {}
    result['error'] = e
    result['hedge'] = beta[0]
    result['mean'] = beta[1]
    result['variance'] = Q
    result['df'] = df
    return result

def build_spread(df, df_hedge, df_spread, spread_type, using_kf=False):
    """
    Computes the spread of two time series under three criteria: price spread, log
    price spread and ratios.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the original time series of two stocks.
    df_hedge : pandas.DataFrame
        with two time series. Used to compute the hedge ratio if the criteria are
        price and log price spread. If the criterion is ratios, it is used
        to compute the ratio (in this case, the ratio is the spread).
    df_spread : pandas.DataFrame
        with two time series. Used to compute the spread if the criteria are
        price and log price spread. If the criterion is ratios, it is not used.
    spread_type : str
        Type of spread: 'price', 'log' or 'ratio'.

    Returns
    -------
        dict
        Containing:
        df: A copy of df.
        df_hedge: A copy of df_hedge if spread_type == 'price' or 'ratio', otherwise it
            contains log(df_hedge)
        df_spread: A copy of df_spread if spread_type == 'price' or 'ratio', otherwise it
            contains log(df_spread)
        symbol: An array contain the two symbols names in such a way that
            symbol[0] is the dependent variable.
        hedge: The computed hedging to yield the spread.
        ext_hedge: A data frame with columns -hedge, -1.
        (In case the criterion is ratios, the dictionary does not contain the keys
            hedge and ext_hedge)
        y_port: The spread time series in a pandas dataframe.
        reversion: The dictionary returned by hurst_exponent_adf_test.
        look_back: The half media reversion time life.
    """
    # Results dictionary creation.
    spread = dict()
    spread['spread_type'] = spread_type

    # Adjust the time series to the type of spread
    if spread_type == 'price' or spread_type == 'ratio':
        spread['df'] = copy.deepcopy(df)
        spread['df_hedge'] = copy.deepcopy(df_hedge)
        spread['df_spread'] = copy.deepcopy(df_spread)
    elif spread_type == 'log':
        spread['df'] = copy.deepcopy(df)
        spread['df_hedge'] = df_hedge.apply(log)
        spread['df_spread'] = df_spread.apply(log)
    else:
        raise ValueError("Unknown spread %s" % spread_type)

    # Symbol names
    spread['symbol'] = [list(df)[0], list(df)[1]]

    # Evaluate the cadf test and order the symbols in the more relevant way.
    stat01 = cadf_test(spread['df_hedge'][spread['symbol'][0]],
                       spread['df_hedge'][spread['symbol'][1]])
    stat10 = cadf_test(spread['df_hedge'][spread['symbol'][1]],
                       spread['df_hedge'][spread['symbol'][0]])
    # Put the dependent and independent variable in such a way cadf the test be
    # more negative.
    # TODO Uncomment the next five lines
    # if stat10 < stat01:
    #     spread['symbol'].reverse()
    #     spread['df'] = spread['df'][spread['symbol']]
    #     spread['df_hedge'] = spread['df_hedge'][spread['symbol']]
    #     spread['df_spread'] = spread['df_spread'][spread['symbol']]

    # Compute dynamic hedge ratios
    if spread_type == 'price' or spread_type == 'log':
        spread['hedge'] = np.ones((len(spread['df_hedge']), 1))
        for day in range(LOOK_BACK, len(df)):
            regression_result = sm.OLS(
                spread['df_hedge'][spread['symbol'][1]][day - LOOK_BACK + 1:
                                                        day + 1], sm.add_constant(
                    spread['df_hedge'][spread['symbol'][0]][day - LOOK_BACK + 1:day
                                                            + 1])).fit()
            spread['hedge'][day] = regression_result.params[1]

        # Compute spreads
        spread['ext_hedge'] = sm.add_constant(-spread['hedge'], prepend=False)

        spread['y_port'] = spread['ext_hedge'] * spread['df_spread']
        spread['y_port'] = np.sum(spread['y_port'], 1)
        if spread_type == 'log':
            spread['hedge'][0:LOOK_BACK] = np.zeros((LOOK_BACK, 1))
            spread['y_port'][0:LOOK_BACK] = np.zeros(LOOK_BACK)

    else:
        spread['y_port'] = spread['df_hedge'][spread['symbol'][0]] / \
                          spread['df_hedge'][spread['symbol'][1]]

        spread['y_port'][0:LOOK_BACK] = np.zeros(LOOK_BACK)

    # Ángel
    # TODO Uncomment the next five lines
    #if spread_type == 'log':
    #    spread['ext_hedge'] /= np.tile(
    #        np.sum((lambda x: abs(x))
    #               (spread['ext_hedge']), 1).reshape(
    #            [len(spread['ext_hedge']), 1]), (1, 2))
    # Ángel

    spread['reversion'] = \
        hurst_exponent_adf_test(spread['y_port'][LOOK_BACK:])
    # TODO uncomment the next line.
    # spread['look_back'] = round(spread['reversion']['half life'])
    # TODO Eliminate the next line
    spread['look_back'] = 20

    return spread


# TODO This program requires the incorporation of methods similar to
#  build_spread to:
#  1. Build spreads from Johansen test.
#  2.                    reversible time series alone
#  3.                    the quotient of two time series
#  See the comments on the parameter spread in the media_reversion method.


def media_reversion(spread, reversion_type):
    """
    Performs two types of media reversion investment methods: linear and
    Bollinger bands.

    Parameters
    ----------
    spread: dict
        The spread to work with. It could be obtained from the dictionary
        returned from the build_spread method (to working with just two
        symbols). Or from a similar method, (not yet implemented TODO) based on
        the Johansen test to include more than two symbols. Or directly from a
        price reversible series TODO. The esencial information that
        media_reversion requires are the following keys inside spread.
        df: The pandas dataframe containing the original price time series used
            to build the spread.
        symbol: A list containing the names of the assets used to construct the
            spread. pd.columns should be equal to this list.
        y_port: The spread time series in a pandas dataframe.
        look_back: The half media reversion time life.
        spread_type: Type of the spread, 'linear, 'log' or 'ratios'.
        ext_hedge: The hedging factors used to construct the spread (just if the
            spread_type is 'linear' or 'log').
    reversion_type: str
        If the investment method is 'linear' or 'bollinger' (Bollinger bands).

    Returns
    -------
        dict
        With the following keys:
        reversion_tipe: The reversion_type parameter.
        moving_mean: The moving mean time series for a look back equals
            to spread['look_back']
        moving_std: Similar to moving_mean for the standard deviation.
        z_score: Normalized spread.
        num_units: -z_score
        entry_score: Entry score for the Bollinger bands method.
            TODO now it is a constant inside the code but the best value
             should be computed from backtesting.
        exit_score: Exit score for the Bollinger bands method.
            TODO the same as for entry_score.
        position: A pandas dataframe containing the positions in cash for each
            asset.
        pnl: The profits and losses pandas dataframe.
        mkt_val: Market value of the portfolio when it was bough or sold.
        ret: The returns pandas dataframe.
        apr: Annual percentage rate of the strategy.
        sharpe: Sharpe ratio of the strategy
        acc_ret: The accumulated returs pandas data frame.
    """
    result = dict()
    result['reversion_type'] = reversion_type

    # Strategy evaluation preliminaries.
    result['moving_mean'] = spread['y_port'].rolling(spread['look_back']).mean()
    result['moving_std'] = spread['y_port'].rolling(spread['look_back']).std()
    result['z_score'] = (spread['y_port'] - result['moving_mean']) / \
        result['moving_std']



    # num_units computation
    if reversion_type == 'linear':
        result['num_units'] = \
            pd.DataFrame(result['z_score'] * -1, columns=['num_units'])
    elif reversion_type == 'bollinger':
        result['entry_zscore'] = 0.7
        result['exit_zscore'] = 0
        longs_entry = result['z_score'] < -result['entry_zscore']
        longs_exit = result['z_score'] > -result['exit_zscore']
        shorts_entry = result['z_score'] > result['entry_zscore']
        shorts_exit = result['z_score'] < result['exit_zscore']

        num_units_long = np.ones(len(spread['y_port'])) * \
            float('nan') * spread['y_port']
        num_units_short = np.ones(len(spread['y_port'])) * \
            float('nan') * spread['y_port']
        num_units_long[0] = 0
        num_units_long[longs_entry] = 1
        num_units_long[longs_exit] = 0
        num_units_long = num_units_long.fillna(method='pad')
        num_units_short[0] = 0
        num_units_short[shorts_entry] = -1
        num_units_short[shorts_exit] = 0
        num_units_short = num_units_short.fillna(method='pad')
        result['num_units'] = (num_units_long +
                               num_units_short)
        result['num_units'] = result['num_units'].values.reshape((len(spread['df']), 1))
    else:
        raise ValueError("Unknown reversion %s" % reversion_type)

    # Investment units in dollars.
    if spread['spread_type'] == 'price':
        result['position'] = \
            spread['ext_hedge'] * np.tile(result['num_units'],
                                          (1, 2)) * spread['df']
    elif spread['spread_type'] == 'log':
        # Ángel
        df_aux = copy.deepcopy(spread['df'])
        df_aux.values[:,:] = np.ones((len(df_aux), 2))

        #result['position'] = \
        #    pd.DataFrame(spread['ext_hedge'] *
        #                 np.tile(result['num_units'], (1, 2)),
        #                 columns=spread['symbol'])

        result['position'] = spread['ext_hedge'] * \
                             np.tile(result['num_units'],
                                 (1, 2)) * df_aux
        # Ángel

    elif spread['spread_type'] == 'ratio':
        df_aux = copy.deepcopy(spread['df'])
        df_aux.values[:, :] = np.ones((len(df_aux), 2))
        df_aux[spread['symbol'][1]] = -np.ones(len(df_aux))

        result['position'] = np.tile(result['num_units'],
                                     (1, 2)) * df_aux


    # Profit and losses.

    result['pnl'] = (result['position'].shift(1) * spread['df'].diff(1)
                     ) / spread['df'].shift(1)

    result['pnl'] = result['pnl'].fillna(value=0.0)
    result['pnl'] = np.sum(result['pnl'], 1)

    # Market value.
    result['mrk_val'] = result['position'].shift(1)
    result['mrk_val'] = result['mrk_val'].apply(abs)
    result['mrk_val'] = result['mrk_val'].fillna(value=0)
    result['mrk_val'] = np.sum(result['mrk_val'], 1)

    # Returns.
    result['ret'] = result['pnl'] / result['mrk_val']
    result['ret'] = result['ret'].fillna(value=0)

    # Sharpe rate and annual percentage rate.
    result['apr'] = np.prod(1 + result['ret'])**(252 / len(result['ret'])) - 1
    # print('\nEl valor del APR es: ' + str(APR))
    result['sharpe'] = (np.sqrt(252) * np.mean(result['ret'])
                        ) / np.std(result['ret'])

    # Accumulated losses and profits
    result['ret_acc'] = pd.Series(np.ones((len(result['ret']))))
    for i in range(len(result['ret'])):
        result['ret_acc'][i] = np.prod(1 + result['ret'][0:i+1]) ** \
            (252 / (i + 1)) - 1.0

    return result


def plot_result(spread, result, ind_strategy):
    """
    Plots the results from the pairs trading strategy

    Parameters
    ----------
    spread: dict
        Containing the spread information. See the documentation about the
        dictionary returned by the build_spread method.
    result: dict
        Containg the results of the strategy. See the documentation about the
        dictionary returned by the media reversion method.
    ind_strategy: int
        Strategy identifier for the pair spread.symbol
    """
    plt.figure()
    spread['y_port'].plot(x='timestamp', y=spread['y_port'].values)
    if result['reversion_type'] == 'bollinger':
        result['moving_mean'].plot(x='timestamp',
                                   y=result['moving_mean'].values)
        banda_sup = result['moving_mean'] + result['entry_zscore'] * \
            result['moving_std']
        banda_inf = result['moving_mean'] - result['entry_zscore'] * \
            result['moving_std']
        banda_sup.plot(x='timestamp', y=banda_sup.values)
        banda_inf.plot(x='timestamp', y=banda_inf.values)
    plt.title(f"SPREAD {spread['symbol'][0]}-{spread['symbol'][1]} "
              f"{spread['spread_type']}-{result['reversion_type']} "
              f"strategy {ind_strategy}")
    plt.annotate("APR: {0:.4f}".format(result['apr']),
                 xy=(0.0, -0.13),
                 xycoords='axes fraction')
    plt.annotate("SHARPE RATIO: {0:0.4f}".format(result['sharpe']),
                 xy=(0.2, -0.13),
                 xycoords='axes fraction')
    plt.show()
    plt.figure()
    result['ret_acc'].plot(x='timestamp', y=result['ret_acc'].values)
    plt.title('Accumulative returs '
              f"{spread['symbol'][0]}-{spread['symbol'][1]} "
              f"{spread['spread_type']}-{result['reversion_type']} "
              f"strategy {ind_strategy}")
    plt.annotate("APR: {0:.4f}".format(result['apr']),
                 xy=(0.0, -0.13),
                 xycoords='axes fraction')
    plt.annotate("SHARPE RATIO: {0:0.4f}".format(result['sharpe']),
                 xy=(0.2, -0.13),
                 xycoords='axes fraction')
    plt.show()


def new_strategies(df, df_residues):
    """
    We extend the pairs trading strategy describe in Chan, E. (2013).
    Algorithmic trading: Winning strategies and their rationale (Vol. 625).
    John Wiley & Sons, Chapter 3. We use the CEEMDAN. Four strategies were
    developed.

    Strategy 1: Original strategy over two time series S1 and S2 stored in df.

    Strategy 2: Original strategy over two time series S1 and S2 in df. To
    compute the hedge ratios, we use the Ceemdan residues r1 and r2 stored into
    df_residues.

    Strategy 3: Original strategy over two time series S1 and S2 in df. To
    compute the hedge ratios and spreads we use S1 and S3 = (r1/r2)*S2,
    where r1 and r2 are the Ceemdan residues of S1 and S2.

    Strategy 4: Original strategy over two time series S1 and S2 in df. To
    compute the hedge ratios we use r1 and r2, and to evaluate the
    spread we use S1 and S3 as defined in the previous paragraph.

    PARAMETER
    ---------
    df : pandas.DataFrame
        a DataFrame containing two time series.

    df_residues: pandas.DataFrame
        a DataFrame the Ceemdan residues of the time series in df.

    RETURN
    ------
        A dataframe containing the APR and ratio sharpe of each strategy.
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

    strategies = [(df, df, df), (df, df_residues, df), (df, df0, df0),
                  (df, df_residues, df0)]

    result = pd.DataFrame()

    for ind, strategy in enumerate(strategies):
        # price_spread_log(strategy[0], strategy[1], strategy[2], flag=True)
        for spread_type in ['price', 'log']:
            spread = build_spread(strategy[0], strategy[1], strategy[2],
                                  spread_type)
            for reversion_type in ['linear', 'bollinger']:
                result = media_reversion(spread, reversion_type)
                plot_result(spread, result, ind)

    return result


def close_and_imfs_residual(symbol,
                            bar_size,
                            start_bar,
                            end_bar,
                            last=0):
    """
    Gets the imf residual of a time series.

    PARAMETERS
    ----------
    symbol : str
        Name of the stock
    bar_size : str
        Size of the bar
    start_bar : datetime.pyi
        Left end of the time interval to be read. It could be None.
    end_bar : datetime.pyi
        Right end of the time interval to be read. It could be None.
    last : int
        Last records number to read.

    RETURNS
    -------
    A Pandas data frame containing the series.

    Examples
    --------
    >>> time_series.read_ts_from_ibdb('AAPL', '1 day', None,
    ...                               datetime.datetime(2022, 12, 4), last=1000)

    """
    series = time_series.read_ts_from_ibdb(symbol, bar_size, start_bar,
                                           end_bar, last=last, imf=True)
    return np.array(series['adj_close']), np.array(series[series.columns[-1]])


def pairs_trading(symbol1, symbol2):
    """
    Applies the pairs trading on the sp500 symbols.

    Parameters
    ----------
    symbol1 : str
        First element in the pair.
    symbol2 : str
        Second element in the pair

    Returns
    -------
        dict
        Containing the relevant information about the process.
    """
    date = min(time_series.last_record_bar(symbol1, '1 day', True),
               time_series.last_record_bar(symbol2, '1 day', True))
    df = pd.DataFrame()
    df_residues = pd.DataFrame()
    df[symbol1], df_residues[symbol1] = close_and_imfs_residual(symbol1,
                                                                '1 day',
                                                                None,
                                                                date,
                                                                last=1000)

    df[symbol2], df_residues[symbol2] = close_and_imfs_residual(symbol2,
                                                                '1 day',
                                                                None,
                                                                date,
                                                                last=1000)
    result = new_strategies(df, df_residues)
    print(result)
    return result


if __name__ == "__main__":
    #df = pd.read_csv('EWA_EWC.csv', index_col=0)
    # df = pd.read_csv('GLD_USO.csv', index_col=0)
    # print(kalman_filter(df))
    df = pd.read_csv('GLD_USO.csv',index_col = 0)
    spread = build_spread(df, df, df, 'price')
    results = media_reversion(spread, 'bollinger')
    print(results['apr'], results['sharpe'])
