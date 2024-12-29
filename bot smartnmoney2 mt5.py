import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SmartMoneyBot:
    def __init__(self, capital_inicial=1000, riesgo_porcentaje=0.02, max_riesgo=0.1):
        self.capital = capital_inicial
        self.riesgo_base = riesgo_porcentaje
        self.max_riesgo = max_riesgo
        self.operaciones_abiertas = 0
        self.max_operaciones = 2
        self.modelo_ia = None
        self.conectar_mt5()

    def conectar_mt5(self):
        try:
            if not mt5.initialize():
                raise Exception("No se pudo conectar a MetaTrader 5")
            print("Conexión exitosa a MetaTrader 5")
        except Exception as e:
            print(f"Error al conectar a MetaTrader 5: {e}")

    def obtener_datos_historial(self, symbol, timeframe, cantidad_barras):
        print(f"Obteniendo datos históricos para {symbol} con timeframe {timeframe}")
        datos = mt5.copy_rates_from_pos(symbol, timeframe, 0, cantidad_barras)
        if datos is None or len(datos) == 0:
            print(f"No se pudo obtener datos históricos para {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(datos)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def calcular_indicadores(self, df):
        # Media Móvil Simple (SMA)
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # Índice de Fuerza Relativa (RSI)
        df['RSI'] = self.calcular_rsi(df['close'], period=14)

        # MACD
        df['MACD'], df['Signal_Line'] = self.calcular_macd(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Bandas de Bollinger
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calcular_bollinger_bands(df['close'], window=20)

        # Fibonacci (retracement levels)
        high = df['high'].max()
        low = df['low'].min()
        df['Fibonacci_0.236'] = high - (high - low) * 0.236
        df['Fibonacci_0.382'] = high - (high - low) * 0.382
        df['Fibonacci_0.618'] = high - (high - low) * 0.618

        # Order Block (simplificado como un cambio significativo en el precio)
        df['Order_Block'] = df['close'] - df['open']
        
        # Break of Structure (BOS): Identificar posibles rupturas de estructura en el gráfico
        df['BOS'] = df['close'] > df['SMA_50']

        return df

    def calcular_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calcular_macd(self, series, fastperiod=12, slowperiod=26, signalperiod=9):
        macd = series.ewm(span=fastperiod, min_periods=fastperiod).mean() - series.ewm(span=slowperiod, min_periods=slowperiod).mean()
        signal_line = macd.ewm(span=signalperiod, min_periods=signalperiod).mean()
        return macd, signal_line

    def calcular_bollinger_bands(self, series, window=20):
        middle_band = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = middle_band + (rolling_std * 2)
        lower_band = middle_band - (rolling_std * 2)
        return upper_band, middle_band, lower_band

    def preparar_datos(self, df):
        df = self.calcular_indicadores(df)

        # Normalización de los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        datos_normalizados = scaler.fit_transform(df[['close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower', 'Fibonacci_0.236', 'Fibonacci_0.382', 'Fibonacci_0.618', 'Order_Block', 'BOS']].values)
        
        # Creación de las secuencias de características y etiquetas
        X = []
        y = []
        for i in range(60, len(datos_normalizados)):
            X.append(datos_normalizados[i-60:i, :])
            y.append(1 if datos_normalizados[i, 0] > datos_normalizados[i-1, 0] else 0)
        
        X, y = np.array(X), np.array(y)
        
        # Redimensionar para ser usado con XGBoost
        return X, y

    def entrenar_modelo(self, X, y):
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Usar XGBoost para entrenar el modelo
        modelo = xgb.XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200, use_label_encoder=False)
        modelo.fit(X_train, y_train)

        # Evaluación del modelo
        y_pred = modelo.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {precision:.4f}")

        self.modelo_ia = modelo

    def predecir(self, df):
        # Preparar los datos para la predicción
        X, _ = self.preparar_datos(df)

        # Realizar la predicción
        if self.modelo_ia is None:
            print("El modelo de IA no está entrenado.")
            return None

        prediccion = self.modelo_ia.predict(X[-1].reshape(1, -1))  # Usar la última secuencia de datos
        return 'BULLISH' if prediccion == 1 else 'BEARISH'

    def gestionar_riesgo(self, capital, stop_loss, take_profit):
        riesgo = capital * self.riesgo_base
        # Calcular el tamaño de la posición según el riesgo y el stop loss
        tamanio_posicion = riesgo / (stop_loss * 0.0001)
        tamanio_posicion = min(tamanio_posicion, self.max_riesgo * capital)
        
        # Verificar que el tamaño de la posición sea válido
        if tamanio_posicion <= 0:
            print("Tamaño de la posición no válido.")
            return 0
        
        return tamanio_posicion

    def abrir_operacion(self, symbol, decision, stop_loss, take_profit):
        if self.operaciones_abiertas >= self.max_operaciones:
            print("Número máximo de operaciones abiertas alcanzado.")
            return False

        # Gestión del riesgo
        tamanio_posicion = self.gestionar_riesgo(self.capital, stop_loss, take_profit)
        if tamanio_posicion == 0:
            print("Operación no abierta debido a tamaño de posición inválido.")
            return False
        
        # Abrir operación según la predicción de la IA
        if decision == 'BULLISH':
            order_type = mt5.ORDER_TYPE_BUY
        else:
            order_type = mt5.ORDER_TYPE_SELL

        # Obtener precios
        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        sl_price = price - stop_loss if order_type == mt5.ORDER_TYPE_BUY else price + stop_loss
        tp_price = price + take_profit if order_type == mt5.ORDER_TYPE_BUY else price - take_profit

        # Validar precios de stop-loss y take-profit
        if sl_price <= 0 or tp_price <= 0:
            print("Precios de Stop-Loss o Take-Profit inválidos.")
            return False

        # Enviar orden
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": tamanio_posicion,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "SmartMoneyBot",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Error al abrir operación: {result.comment}")
            return False

        self.operaciones_abiertas += 1
        print(f"Operación abierta: {symbol}, tipo: {order_type}, volumen: {tamanio_posicion}")
        return True

# Crear una instancia del bot de dinero inteligente
bot = SmartMoneyBot()

# Obtener datos históricos para EURUSD en timeframe de 5 minutos
df = bot.obtener_datos_historial("EURUSD", mt5.TIMEFRAME_M5, 1000)

# Entrenar el modelo con los datos
if not df.empty:
    X, y = bot.preparar_datos(df)
    bot.entrenar_modelo(X, y)

    # Tomar decisiones y abrir operaciones
    decision = bot.predecir(df)
    if decision:
        print(f"Predicción de IA: {decision}")
        bot.abrir_operacion("EURUSD", decision, stop_loss=50, take_profit=100)
