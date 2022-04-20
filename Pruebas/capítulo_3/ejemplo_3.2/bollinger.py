import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import zeros, ones, flipud, log
from numpy.linalg import inv, eig, cholesky as chol
import csv
from statsmodels.regression.linear_model import OLS
import statsmodels.tsa.tsatools as tsat
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from numpy.matlib import repmat

# SE LEEN LOS DATOS DEL ARCHIVO GLD_USO.CSV
df = pd.read_csv('GLD_USO.csv',index_col = 0)

# SE FIJA EL LOOKBACK PARA CALCULAR LOS CAMBIOS DINAMICAMENTE
lookback = 20

hedgeratio = np.ones((len(df),1)) # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

# CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
for i in range(lookback,len(df)):
	resultado_regresion = sm.OLS(df['USO'][i-lookback+1:i+1],sm.add_constant(df['GLD'][i-lookback+1:i+1])).fit()
	hedgeratio[i] = resultado_regresion.params[1]

# SE CALCULAN LOS SPREADS
AA = sm.add_constant(-hedgeratio, prepend = False)
yport = AA*df
yport = np.sum(yport,1)

hedgeratio[0:20] = np.zeros((20,1))
yport[0:20] = np.zeros(20)
s = (lookback,2)
df.values[0:20,:] = np.zeros(s)

##############################################################################
#----ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS USANDO BANDAS DE BOLLINGER-----#
##############################################################################
print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS USANDO BANDAS DE BOLLINGER:')

entryZscore = 0.7
exitZscore = 0

# SE CALCULA LA MEDIA Y DESVIACION ESTANDAR MOVIL
moving_mean = yport.rolling(lookback).mean()
moving_std = yport.rolling(lookback).std()
z_score = (yport - moving_mean) / moving_std
m=moving_mean
d=moving_std
m=m.fillna(value = 0)
d=d.fillna(value = 0)
d1=m-d
d2=m+d


plt.figure()
yport.plot(x = 'timestamp', y = yport.values)
m.plot(x = 'timestamp', y=m.values)
d1.plot(x = 'timestamp', y=d1.values)
d2.plot(x = 'timestamp', y=d2.values)
plt.title(''' Spread = USO-hedgeratio*GLD \n''') 
plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
plt.ylabel('USO/GLD  \n')
plt.savefig('USO_GLD.png')


# CONDICIONES PARA POSICIONES EN LARGO
longsEntry = z_score < -entryZscore
longsExit = z_score > -exitZscore

# CONDICIONES PARA POSICIONES EN CORTO
shortsEntry = z_score > entryZscore
shortsExit = z_score < exitZscore

# SE CREAN ARREGLOS AUX PARA CALCULAR UNIDADES EN LARGO Y EN CORTO (BOLLINGER.M ARCHIVO MATLAB)
numUnitsLong = np.ones(len(yport))*float('nan')*yport
numUnitsShort = np.ones(len(yport))*float('nan')*yport

numUnitsLong[0] = 0
numUnitsLong[longsEntry] = 1 
numUnitsLong[longsExit] = 0
numUnitsLong = numUnitsLong.fillna(method = 'pad')

numUnitsShort[0] = 0
numUnitsShort[shortsEntry] = -1 
numUnitsShort[shortsExit] = 0
numUnitsShort = numUnitsShort.fillna(method = 'pad')

numUnits = numUnitsLong + numUnitsShort

# SE CALCULAN LAS POSICIONES
CC = np.transpose(repmat(numUnits,2,1))
BB = sm.add_constant(-hedgeratio, prepend = False)
position = BB*CC*df

# SE CALCULA LAS UTILIDADES Y PERDIDAS DIARIAS
pnl = (position.shift(1)*df.diff(1))/df.shift(1)
pnl = pnl.fillna(value = 0)
pnl = np.sum(pnl,1)

# SE CALCULA EL VALOR EN EL MERCADO
mrk_val = position.shift(1)
mrk_val = mrk_val.apply(abs)
mrk_val = mrk_val.fillna(value = 0)
mrk_val = np.sum(mrk_val,1)

# SE CALCULAN LOS RENDIMIENTOS
ret = pnl/mrk_val
ret = ret.fillna(value = 0)

# SE CALCULA EL APR Y LA RAZON DE SHARPE
APR = np.prod(1+ret)**(252/len(ret))-1
print('\nEl valor del APR es:'+ str(APR))
sharpe = (np.sqrt(252)*np.mean(ret))/np.std(ret)
print('La razon de Sharpe es de: '+ str(sharpe))

# GRAFICA
rtn = np.cumprod(ret+1)-1
plt.figure()
rtn.plot(x='timestamp', y=rtn.values)
plt.title(''' Rendimiento acumulado de la estrategia de \nBandas de Bollinger sobre GLD-USO ''') 
plt.xlabel('\n 26 de Abril de 2006 a 9 de Abril de 2012')
plt.ylabel('Rendimiento Acumulado \n')
plt.savefig('Banda_Bollinger_rend_acum.png')
plt.show()
