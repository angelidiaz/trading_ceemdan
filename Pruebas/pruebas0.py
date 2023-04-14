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

def fistdate_lastdate(symbols):
    symbols = symbols[:100]
    first_dates = []
    last_dates = []
    for symbol in symbols:
        # print(new_nodes[node1],node1)
        if symbol not in ['BF B', 'OGN', 'VTRS', 'CARR', 'OTIS']:
            dates = np.array(pd.read_csv(PATH1 + symbol + ".csv")["date"])
            first_dates.append(min(dates))
            last_dates.append(max(dates))
    return max(first_dates), min(last_dates)


def correlaciones():
    symbols_example = ["AAPL", "AMZN", "FB", "GOOGL", "TSLA"]
    for symbol1 in symbols_example:
        for symbol2 in symbols_example:
            print(symbol1, symbol2)
            stock1 = pd.read_csv(PATH1 + symbol1 + ".csv")
            stock2 = pd.read_csv(PATH1 + symbol2 + ".csv")
            stock1 = np.array(stock1['close'][np.where(
                np.array(stock1['date']) == FIRST_DATE)[0][0]:np.where(
                np.array(stock1['date']) == LAST_DATE)[0][0]])
            stock2 = np.array(stock2['close'][np.where(
                np.array(stock2['date']) == FIRST_DATE)[0][0]:np.where(
                np.array(stock2['date']) == LAST_DATE)[0][0]])
            regresion = np.corrcoef(stock1, stock2)
            print(regresion)


def new_price_spread(symbol1, symbol2, flag=False):
    # print(symbol1 + '-' + symbol2)
    series1 = pd.read_csv(PATH1 + symbol1 + ".csv")
    series2 = pd.read_csv(PATH1 + symbol2 + ".csv")
    series1 = np.array(series1['close'][np.where(
        np.array(series1['date']) == FIRST_DATE)[0][0]:np.where(
        np.array(series1['date']) == LAST_DATE)[0][0]])
    series2 = np.array(series2['close'][np.where(
        np.array(series2['date']) == FIRST_DATE)[0][0]:np.where(
        np.array(series2['date']) == LAST_DATE)[0][0]])
    imf_1 = pd.read_csv(PATH2 + symbol1 + '.csv')
    imf_2 = pd.read_csv(PATH2 + symbol2 + '.csv')
    residue1 = np.array(imf_1[str(len(imf_1.columns) - 2)])
    residue2 = np.array(imf_2[str(len(imf_2.columns) - 2)])
    quotient = residue1 / residue2

    df = pd.DataFrame()
    df[symbol1] = series1
    df[symbol2] = series2

    df1 = pd.DataFrame()
    df1[symbol1] = series1
    df[symbol2] = quotient * series2

    lookback = 20

    ### ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS ###

    hedgeratio = np.ones((len(df), 1))
    # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

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






if __name__ == '__main__':
    PATH1 = '/home/angel/Documentos/Datos/sp500-1day/'
    PATH2 = '/home/angel/Documentos/Datos/CEEMDAN/'
    symbols = pd.read_csv(PATH1 + "sp500_symbols.csv")['symbol'][:100]
    FIRST_DATE, LAST_DATE = fistdate_lastdate(symbols)

    new_price_spread("TSLA", "AAPL", flag=True)


