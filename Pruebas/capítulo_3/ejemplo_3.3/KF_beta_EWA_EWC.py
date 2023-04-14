import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from numpy.matlib import repmat

# SE DEFINE LA MULTIPLICACION AB DONDE A ES 2X1 Y B ES 1X2
def mulmat(A,B):
	S = np.zeros((2,2))
	S[0,0] = A[0]*B[0]
	S[0,1] = A[0]*B[1]
	S[1,0] = A[1]*B[0]
	S[1,1] = A[1]*B[1]
	return S

# SE LEEN LOS DATOS DEL ARCHIVO EWA_EWC.CSV
df = pd.read_csv('EWA_EWC.csv',index_col = 0)

# SE CREA EL ARREGLO EWA AUMENTADO PARA LOS POSIBLES VALORES DE LA REGRESION EWC
x = sm.add_constant(df['EWA'], prepend = False)
delta = 0.0001
s=(len(df),1)

# MEDIDAS DE PREDICCION
yhat = np.ones(s)*float('nan')
# MEDIDAS DE PREDICCION DEL ERROR
e = np.ones(s)*float('nan')

# MEDIDAS DE PREDICCION DE LA VARIANZA DEL ERROR
Q = np.ones(s)*float('nan')
# pARA ACLARAR, SE DENOTA R(T|T) POR P(T). SE INICIALIZA R,P Y BETA
R = np.zeros((2,2)) # R(t+1|t)
P = np.zeros((2,2)) # R(t|t)
r = (2,len(df))
beta = np.zeros(r)*float('nan') # beta(t+1|t)
beta_t = np.zeros(r)*float('nan') # beta(t|t)
Vw = (delta/ (1 -delta))*np.identity(2)
Ve = 0.001

# SE INICIALIZA BETA[:,0] CON VALORES 0
beta[:,0] = 0
# DADOS VALORES INICIALES DE BETA Y R (Y P) 
for i in range(len(df)):
	if i > 0:
		beta[:,i] = beta[:,i-1]# ESTADO DE PREDICCION
		R = P+Vw# ESTADO DE PREDICCION COVARIANZA
	yhat[i] = np.matmul(x.values[i,:],beta[:,i])# MEDIDA DE PREDICCION
	Q[i] = np.matmul(np.matmul(x.values[i,:],R),np.transpose(x.values[i,:]))+Ve# MEDIDA DE PREDICCION VARIANZA
	e[i] = df['EWC'][i]-yhat[i]# MEDIDA DE PREDICCION DEL ERROR
	K = np.matmul(R,np.transpose(x.values[i,:]))/Q[i] #GANANCIA KALMAN
	beta[:,i] = beta[:,i]+K*e[i]#ACTUALIZACION ESTADO
	K_aux = K.reshape(2,1)
	x_aux = x.values[i,:].reshape(1,2)
	P = R-np.matmul(np.matmul(K_aux,x_aux),R)#ACTUALIZACION ESTADO COVARIANZA

print(beta)
# GRAFICA ESTIMACION MEDIANTE EL FILTRO DE KALMAN DE LA PENDIENTE ENTRE EWC Y EWA
plt.figure()
plt.plot(beta[0,:])
plt.title(''' Estimacion de la pendiente entre EWC y EWA con el filtro de Kalman \n''') 
plt.xlabel('\n 26 de Abril de 2006 a 9 de Abril de 2012')
plt.ylabel('Pendiente = ß(0,t) \n')
plt.savefig('Pendiente.png')

# GRAFICA ESTIMACION MEDIANTE EL FILTRO DE KALMAN DE LA INTERSECCION ENTRE EWC Y EWA
plt.figure()
plt.plot(beta[1,:])
plt.title(''' Estimacion de la intercepcion entre EWC y EWA con el filtro de Kalman \n''') 
plt.xlabel('\n 26 de Abril de 2006 a 9 de Abril de 2012')
plt.ylabel('Intercepcion = ß(1,t) \n')
plt.savefig('Intercepcion.png')

# GRAFICA DE MEDIDAS DE PREDICCION DEL ERROR Y DESVIACION ESTANDAR DEL ERROR
plt.figure()
plt.plot(e[2:],label = "e(t)")
plt.plot(np.sqrt(Q[2,:]), label = "sqrt(Q(t))")
plt.legend()
plt.title(''' Medidas de prediccion error y desv. estandar error \n''') 
plt.xlabel('\n 26 de Abril de 2006 a 9 de Abril de 2012')
plt.ylabel('Intercepcion = ß(1,t) \n')
plt.savefig('Pred_error.png')

# CALCULAR POSICIONES EN LARGO
longsEntry = e < -np.sqrt(Q)
longsExit = e > -np.sqrt(Q)

# CALCULAR POSICIONES EN CORTO
shortsEntry = e > np.sqrt(Q)
shortsExit = e < np.sqrt(Q)

# SE CREAN ARREGLOS AUX PARA CALCULAR UNIDADES EN CORTO Y LARGO
numUnitsLong = np.ones(len(df))*float('nan')*np.sum(df,1)
numUnitsShort = np.ones(len(df))*float('nan')*np.sum(df,1)

numUnitsLong[0] = 0
numUnitsLong[longsEntry[:,0]] = 1 
numUnitsLong[longsExit[:,0]] = 0
numUnitsLong = numUnitsLong.fillna(method = 'pad')

numUnitsShort[0] = 0
numUnitsShort[shortsEntry[:,0]] = -1 
numUnitsShort[shortsExit[:,0]] = 0
numUnitsShort = numUnitsShort.fillna(method = 'pad')

numUnits = numUnitsLong + numUnitsShort

# SE CALCULAN LAS POSICIONES
CC = np.transpose(repmat(numUnits,2,1))
BB = sm.add_constant(-np.transpose(beta[0,:]), prepend = False)
position = BB*CC*df

# SE CALCULAN LAS UTILIDADES Y PERDIDAS DIARIAMENTE
pnl = (position.shift(1)*df.diff(1))/df.shift(1)
pnl = pnl.fillna(value = 0)
pnl = np.sum(pnl,1)

# SE CALCULA EL VALOR DEL MERCADO
mrk_val = position.shift(1)
mrk_val = mrk_val.apply(abs)
mrk_val = mrk_val.fillna(value = 0)
mrk_val = np.sum(mrk_val,1)

# SE CALCULA LOS RENDIMIENTOS
ret = pnl/mrk_val
ret = ret.fillna(value = 0)

# Comentamos un poco de la interfaz del programa
print('RESULTADOS FILTRO DE KALMAN COMO REGRESION LINEAL DINAMICA: ')

# SE CALCULA EL APR Y LA RAZON DE SHARPE
APR = np.prod(1+ret)**(252/len(ret))-1
print('\nEl valor del APR es:'+ str(APR))
sharpe = (np.sqrt(252)*np.mean(ret))/np.std(ret)
print('La razon de Sharpe es de: '+ str(sharpe))


# GRAFICA RENDIMIENTO ACUMULADO DE LA ESTRATEGIA DEL FILTRO DE KALMAN SOBRE EWA-EWC
rtn = np.cumprod(ret+1)-1
plt.figure()
rtn.plot(x='timestamp', y=rtn.values)
plt.title(''' Rendimiento acumulado de la estrategia con \nFiltro de Kalman sobre EWC-EWA ''') 
plt.xlabel('\n 26 de Abril de 2006 a 9 de Abril de 2012')
plt.ylabel('Rendimiento Acumulado \n')
plt.savefig('Rendimiento_Acum.png')
plt.show()
