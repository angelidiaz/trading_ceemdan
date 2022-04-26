Contenido del repositorio:

1. pairs_trading_emd.py: Es el documento principal, en el cual se encuentra la aplicación de la ceemdan a las estrategias por pares, descritas en la 
   siguiente referencia: Chan, E. (2013). Algorithmic trading: winning strategies and their rationale (Vol. 625). John Wiley & Sons, chapter 3. Se han 
   creado tres nuevas versiones de las estrategias descritas en la referencia, utilizando la ceemdan.
   
2. AAPL.csv, AAPL_imf.csv, TSLA.csv, TSLA_imf.csv: Son series de tiempo utilizadas, y su descompocición en IMFs, y se utilizan para debuggear el programa
   principal pairs_trading_emd.py.

3. trading-env: Es el ambiente virtual que contiene todas las bibliotecas necesarias para la ejecución de los programas.

4. Pruebas: Esta carpeta contiene los siguientes archivos:
   
   1. capitulo_3: Esta carpeta contiene las estrategias descritas en esta referencia: Chan, E. (2013). Algorithmic trading: winning strategies and 
      their rationale (Vol. 625). John Wiley & Sons, chapter 3.
   
   2. emdpruebas.py: Es el programa con el que se inició el proyecto, en el cual se hacen pruebas de nuestra propuesta de la aplicación de la ceemdan a las 
      estrategias de pares. pairs_trading_emd.py es una versión más limpia y sintética de emdpruebas.py.
      
   3. pairs_trading_emd_copy.py: Es una versión antigua del programa pairs_trading_emd.py.
   
   4. estacionariedad_hilbert.py: Es la programación de un nuevo test de estacionariedad, utilizando la transformada de Hilbert.
   
   5. verificacionseries.py: Es el programa que revisa con cuidado las series de tiempo del S&P 500.  
