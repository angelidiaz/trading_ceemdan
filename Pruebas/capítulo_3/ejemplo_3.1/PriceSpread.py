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

df = pd.read_csv('GLD_USO.csv',index_col = 0)
lookback = 20

#print(df)

#ESTRATEGIA PARA EL SPREAD DE LOS PRECIOS

print('######################################################')
print('RESULTADOS DE LA ESTRATEGIA PARA EL SPREAD DE PRECIOS:')

hedgeratio = np.ones((len(df),1)) # ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE
 

# CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
for i in range(lookback,len(df)):
	resultado_regresion = sm.OLS(df['USO'][i-lookback+1:i+1],sm.add_constant(df['GLD'][i-lookback+1:i+1])).fit()
	hedgeratio[i] = resultado_regresion.params[1]

# SE CALCULAN LOS SPREADS
AA = sm.add_constant(-hedgeratio, prepend = False)
yport = AA*df

yport = np.sum(yport,1)


plt.figure()
yport.plot(x = 'timestamp', y = yport.values)
plt.title(''' Spread = USO-hedgeratio*GLD \n''') 
plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
plt.ylabel('USO/GLD  \n')
plt.savefig('USO_GLD.png')

# OBSERVANDO QUE LA GRAFICA ANTERIOR LUCE ESTACIONARIA MOTIVA A APLICAR UNA ESTRATEGIA LINEAL DE REVERSION  A GLD Y USO
moving_mean = yport.rolling(lookback).mean()
moving_std = yport.rolling(lookback).std()
z_score = (yport - moving_mean) / moving_std
numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

# SE PROCEDE A CALCULAR EL INDICE DE SHARPE Y EL APR



# SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
position = sm.add_constant(-hedgeratio, prepend = False)*repmat(numunits,1,2)*df

# SE CALCULAN LAS UTILIDADES Y PERDIDAS
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

# RAZON DE SHARPE Y APR
APR = np.prod(1+ret)**(252/len(ret))-1
print('\nEl valor del APR es:'+ str(APR))
sharpe = (np.sqrt(252)*np.mean(ret))/np.std(ret)

print('La razon de Sharpe es de: '+ str(sharpe))

print('\nVer grafica en USO_GLD.png')
print('Nota: la grafica luce bastante estacionaria ideal para aplicar la estrategia')



#ESTRATEGIA PARA EL SPREAD DE LOS LOGARITMOS DE LOS PRECIOS

print('###########################################################################')
print('\nRESULTADOS DE LA ESTRATEGIA PARA EL SPREAD EL LOGARITMO DE LOS DE PRECIOS')


hedgeratio = np.ones((len(df),1))# ARREGLO AUX CON UNOS PARA CALCULAR PESOS DINAMICAMENTE LA RAZON DE HEDGE

# CALCULO DE LA RAZON DE HEDGE DINAMICAMENTE
for i in range(lookback,len(df)):
	res = sm.OLS(df.apply(log)['USO'][i-lookback+1:i+1],sm.add_constant(df.apply(log)['GLD'][i-lookback+1:i+1])).fit()
	hedgeratio[i] = res.params[1]

# SE CALCULAN LOS SPREADS
AA = sm.add_constant(-hedgeratio, prepend = False)
yport = AA*df.apply(log)
yport = np.sum(yport,1)
hedgeratio[0:20] = np.zeros((20,1))
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(hedgeratio)
yport[0:20] = np.zeros(20)
s = (lookback,2)
df.values[0:20,:] = np.zeros(s)
print('***************************************************************************')
print(yport)
plt.figure()
yport.plot(x = 'timestamp', y = yport.values)
plt.title(''' Spread = USO-hedgeratio*GLD (usando logaritmo) \n''') 
plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
plt.ylabel('USO/GLD  \n')
plt.savefig('USO_GLD_log.png')

# OBSERVANDO QUE LA GRAFICA ANTERIOR LUCE ESTACIONARIA MOTIVA A APLICAR UNA ESTRATEGIA LINEAL DE REVERSION  A GLD Y USO
moving_mean = yport.rolling(lookback).mean()
moving_std = yport.rolling(lookback).std()
z_score = (yport - moving_mean) / moving_std
numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

# SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
df1 = pd.read_csv('GLD_USO.csv',index_col = 0)

s = (1500,2)
df1.values[:,:] = np.ones(s)

position = sm.add_constant(-hedgeratio, prepend = False)*repmat(numunits,1,2)*df1

# SE CALCULAN LAS UTILIDADES Y PERDIDAS DIARIAS
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

print('\nVer grafica en USO_GLD_log.png')
print('Nota: la grafica luce bastante estacionaria ideal para aplicar la estrategia')



#ESTRATEGIA PARA LA RAZON USO/GLD

print('###################################################')
print('\nRESULTADOS DE LA ESTRATEGIA PARA LA RAZON USO/GLD')

# SE CALCULA LA RAZON USO/GLD
ratio = df['USO']/df['GLD']
ratio[0:lookback] = np.zeros(20)

# SE GRAFICA LA RAZON USO/GLD
plt.figure()
ratio.plot(x = 'timestamp', y = ratio.values)
plt.title(''' Razon = USO/GLD \n''') 
plt.xlabel('\n 24 de Mayo de 2006 a 9 de Abril de 2012')
plt.ylabel('USO/GLD  \n')
plt.savefig('USO_GLD_razon.png')
plt.show()
# OBSERVANDO QUE LA GRAFICA *NO* LUCE TAN ESTACIONARIA, PERO AUN ASI SE APLICA
# LA ESTRATEGIA LINEAL DE REVERSION  A USO/GLD

moving_mean = ratio.rolling(lookback).mean()
moving_std = ratio.rolling(lookback).std()
z_score = (ratio - moving_mean) / moving_std
numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

# SE CALCULA EL NUMERO DE UNIDADES INVERTIDAS EN DOLARES
CC = np.tile(numunits, (1,2))
print(CC)
print(df1)
df1['USO'] = -np.ones(len(df))
print(df1)
position = -CC*df1

# SE CALCULAN LAS UTILIDADES Y PERDIDAS DIARIAS
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

print('\nVer grafica en USO_GLD_razon.png')
print('Nota: la grafica NO luce estacionaria, aun asi se aplica la estrategia\n')
