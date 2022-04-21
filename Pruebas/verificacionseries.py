import pandas as pd
import numpy as np
import yfinance as yf# Biblioteca para descargar series de tiempo de Yahoo Finance


path='/home/angel/Desktop/Sectorizacion/sp500-1day/'
simbolos= pd.read_csv(path+"sp500_symbols.csv")# Lee el archivo donde se encuentran los simbolos y sus pesos.

print(simbolos)

A=list(simbolos['symbol'])# Creamos una lista donde se encuentran solo los simbolos

def restaseries(k): #Esta funcion resta la serie de tiempo proporcionada con la descargada de Yahoo  
	df1=pd.read_csv(path+str(A[k])+".csv")#Lee la serie del k-esimo simbolo
	data = yf.download(A[k], start=df1['date'][0], end=df1['date'][len(df1['date'])-1])#descarga la serie de Yahoo Finance del k-esimo simbolo
	'''Algunas series de tiempo descargadas de Yahoo Finance muestran algunas inconsistencias, por ejemplo
	en lugar de valores tienen unicamente None, y con eso no es posible hacer una resta, por lo que
	escribi una excepcion con ValuError'''
	try:
		r=np.array(df1['date'][:-1])[np.absolute((np.array(df1['close'][:-1])-np.array(data['Adj Close'])))>0.01]
		if len(r)==len(df1['date'][:-1]):
			return (np.array([A[k],'all']))#si en todas las fechas del activo los precios difieren de mas de $0.01, devuelve all. 
		else:
			return(np.array([A[k],r]))#devuelve las fechas, del activo en cuestion, donde los precios difieren por mas de $0.01.
	except ValueError:
		return(A[k]+'*')#Si no se pudo realizar la operacion devuelve el simbolo del activo con un *

def restaseries1(k): #Esta funcion resta la serie de tiempo proporcionada con la descargada de Yahoo  
	df1=pd.read_csv(path+str(A[k])+".csv")#Lee la serie del k-esimo simbolo
	data = yf.download(A[k], start=df1['date'][0], end=df1['date'][len(df1['date'])-1])#descarga la serie de Yahoo Finance del k-esimo simbolo
	'''Algunas series de tiempo descargadas de Yahoo Finance muestran algunas inconsistencias, por ejemplo
	en lugar de valores tienen unicamente None, y con eso no es posible hacer una resta, por lo que
	escribi una excepcion con ValuError'''
	try:
		r=len(np.array(df1['date'][:-1])[np.absolute((np.array(df1['close'][:-1])-np.array(data['Adj Close'])))>0.01])
		if r>0:
			return ([A[i],r])
	except ValueError:
		return(A[k]+'*')#Si no se pudo realizar la operacion devuelve el simbolo del activo con un *

#fechas_precios_diferentes=[]

#for i in range(len(A)):
#	print (A[i],i)
#	if A[i]!='AMCR' and A[i]!='FANG' and A[i]!='FLIR':#Estas series se excluyen poque son las que estan vacias.
#		fechas_precios_diferentes.append(restaseries(i))#Guarda en una lista el simbolo del activo con las fechas.



#df_fechas= pd.DataFrame(np.array(fechas_precios_diferentes))
#df_fechas.to_csv('fechas_precios_diferentes.csv', header=False, index=False)#Guarda la lista formada en un documento .csv

simbolos_fechas_precios_diferentes=[]

for i in range(len(A)):
	print (A[i],i)
	if A[i]!='AMCR' and A[i]!='FANG' and A[i]!='FLIR':#Estas series se excluyen poque son las que estan vacias.
		h=restaseries1(i)
		if h!=None:
			simbolos_fechas_precios_diferentes.append(h)#Guarda en una lista el simbolo del activo con las fechas.



df_fechas= pd.DataFrame(np.array(simbolos_fechas_precios_diferentes))
df_fechas.to_csv('/home/angel/Desktop/Sectorizacion/resultados/'+'simbolos fechas_precios_diferentes.csv', header=False, index=False)#Guarda la lista formada en un documento .csv
