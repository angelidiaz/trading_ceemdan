'''Pruebas de estrategias de pares utilizando
la EMD'''

import pandas as pd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from PyEMD import EMD
from PyEMD import CEEMDAN
import statsmodels.tsa.stattools as ts
import datetime
import http.client
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import statsmodels.formula.api as sm
from numpy.matlib import repmat
import math
import csv
from statsmodels.regression.linear_model import OLS
import statsmodels.tsa.tsatools as tsat
import statsmodels.api as sm
from pandas_ods_reader import read_ods
from numpy import zeros, ones, flipud, log


def fistdate_lastdate():
    """
    Return the minimal first date and the maximal last
    of all 100 first series of S&P500

    PARAMETER
    ---------
    None

    RETURN
    ------
        A 2-tuple with both dates.
    """
    path = '/home/angel/Desktop/Datos/sp500-1day/'
    symbols = pd.read_csv(path + "sp500_symbols.csv")
    symbols = symbols['symbol'][:100]
    first_dates = []
    last_dates = []
    for symbol in symbols:
        if symbol not in ['BF B', 'OGN', 'VTRS', 'CARR', 'OTIS']:
            dates = np.array(pd.read_csv(path + symbol + ".csv")["date"])
            first_dates.append(min(dates))
            last_dates.append(max(dates))
    return max(first_dates), min(last_dates)


def cociente(symbol1, symbol2):
    """
    Return the quotient of the residue of series of symbol1
    divided by the residue of series of symbol2

    PARAMETER
    ---------
    symbol1 : string
        a symbol of S&P500

    symbol2 : string
        another symbol os S&P500

    RETURN
    ------
        Numpy-array
    """
    series1 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol1 +
                          '.csv')
    series2 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' + symbol2 +
                          '.csv')
    residuo1 = np.array(series1[str(len(series1.columns) - 2)])
    residuo2 = np.array(series2[str(len(series2.columns) - 2)])
    return residuo1 / residuo2


def new_strategy(symbol1, symbol2):
    """
    Return 3 DataFrames:
    1. df: df[symbol1] is the series asociated to the symbol1
           df[symbol2] is the series asociated to the symbol2
           times the quotients of theirs residues.
    2. df1 has two residues of the 2 stocks asociated to both symbols
    3. df2 has two time series of the stocks asociated to both symbols

    PARAMETER
    ---------
    symbol1 : string
        a symbol of S&P500

    symbol2 : string
        another symbol os S&P500

    RETURN
    ------
        DataFrame
    """
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
    imfs_of_stock1 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' +
                                 symbol1 + '.csv')
    imfs_of_stock2 = pd.read_csv('/home/angel/Desktop/Datos/CEEMDAN/' +
                                 symbol2 + '.csv')
    residuo1 = np.array(imfs_of_stock1[str(len(imfs_of_stock1.columns) - 2)])
    residuo2 = np.array(imfs_of_stock2[str(len(imfs_of_stock2.columns) - 2)])
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


def price_spread(symbol1, symbol2, opc):
    """
    Return 2 APRs.[Spanish version] Se aplican las estrategias de Trading Pairs
    usando los price spread y log price spread a las series de tiempo S1 y S2
    correspondientes a los simbolos symbol1 y symbol2 del S&P500. opc
    representa las siguientes opciones:

    opc = 1: La estrategia se aplica a S1 y S2 sin ninguna modificación
    de la estrategia original.

    opc = 2: Se aplica la estrategia al par (S1, S2) y al calcular dinamicamen-
    los spreads se utiliza unicamente los residuos.

    opc = 3: Se calcula el cociente de los residuos de S1 y S2 (llamado r)
    y aplicamos la estragia a S1 y S2. Al calcular los spreads y las
    posiciones utilizamos el par (S1, r*S2).

    opc = 4: Se aplica la estrategia a S1 y S2. Para calcular los spreads
    se utilizan unicamente los residuos y para calcular las posiciones
    se utiliza el par (S1, r*S2).

    PARAMETER
    ---------
    symbol1 : string
        a symbol of S&P500

    symbol2 : string
        another symbol os S&P500

    RETURN
    ------
        a 2-tuple
    """

    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2)
    if opc == 1:
        df_0 = df_1 = df_2 = df2
    if opc == 2:
        df_0 = df1
        df_1 = df_2 = df2
    if opc == 3:
        df_0 = df_1 = df
        df_2 = df2
    if opc == 4:
        df_0 = df1
        df_1 = df
        df_2 = df2

    ### ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS ###

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df_0[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df_0[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df_1
    yport = np.sum(yport, 1)

    moving_mean = yport.rolling(lookback).mean()
    moving_std = yport.rolling(lookback).std()
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

    # SE PROCEDE A CALCULAR EL INDICE DE SHARPE Y EL APR

    # SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
    position = sm.add_constant(-hedgeratio, prepend=False) * repmat(
        numunits, 1, 2) * df_2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df_2.diff(1)) / df_2.shift(1)
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

    ## ESTRATEGIA PARA EL SPREAD DE LOS LOG PRECIOS ##

    hedgeratio = np.ones(
        (len(df), 1)
    )  # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        res = sm.OLS(
            df_0.apply(log)[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df_0.apply(log)[symbol1][i - lookback + 1:i +
                                                     1])).fit()
        hedgeratio[i] = res.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df_1.apply(log)
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
        numunits, 1, 2) * df_2

    # SE CALCULAN LAS UTILIDADES Y PERDIDAS
    pnl = (position.shift(1) * df_2.diff(1)) / df_2.shift(1)
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


def bollinger_bands(symbol1, symbol2, opc):
    """
    Return 2 APRs.[Spanish version] Se aplican las estrategias bandas
    de bollinger a las series de tiempo S1 y S2 correspondientes a los simbolos
    symbol1 y symbol2 del S&P500. opc representa las siguientes opciones:

    opc = 1: La estrategia se aplica a S1 y S2 sin ninguna modificación
    de la estrategia original.

    opc = 2: Se aplica la estrategia al par (S1, S2) y al calcular dinamicamen-
    los spreads se utiliza unicamente los residuos.

    opc = 3: Se calcula el cociente de los residuos de S1 y S2 (llamado r)
    y aplicamos la estragia a S1 y S2. Al calcular los spreads y las
    posiciones utilizamos el par (S1, r*S2).

    opc = 4: Se aplica la estrategia a S1 y S2. Para calcular los spreads
    se utilizan unicamente los residuos y para calcular las posiciones
    se utiliza el par (S1, r*S2).


    PARAMETER
    ---------
    symbol1 : string
        a symbol of S&P500

    symbol2 : string
        another symbol of S&P500

    RETURN
    ------
        a float
    """

    lookback = 20
    df, df1, df2 = new_strategy(symbol1, symbol2)
    if opc == 1:
        df_0 = df_1 = df_2 = df2
    if opc == 2:
        df_0 = df1
        df_1 = df_2 = df2
    if opc == 3:
        df_0 = df_1 = df
        df_2 = df2
    if opc == 4:
        df_0 = df1
        df_1 = df
        df_2 = df2

    # ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

    # print('######################################################')
    # print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

    # CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
    for i in range(lookback, len(df)):
        resultado_regresion = sm.OLS(
            df_0[symbol2][i - lookback + 1:i + 1],
            sm.add_constant(df_0[symbol1][i - lookback + 1:i + 1])).fit()
        hedgeratio[i] = resultado_regresion.params[1]

    # SE CALCULAN LOS SPREADS
    AA = sm.add_constant(-hedgeratio, prepend=False)
    yport = AA * df_1
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
    position = BB * CC * df_2

    # SE CALCULA LAS UTILIDADES Y PERDIDAS DIARIAS
    pnl = (position.shift(1) * df_2.diff(1)) / df_2.shift(1)
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
