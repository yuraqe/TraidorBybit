
import asyncio
import io
import os
import joblib
import threading
import pandas as pd
import functools
import logging
from datetime import datetime, timezone

from verification import BUCKET_NAME, KEYS_GCS_PATH, MODEL_PATH, BYBIT_SECRET_KEY, BYBIT_API_KEY, MODEL_PATH2, \
    BYBIT_API_KEY_sub, BYBIT_SECRET_KEY_sub
from google.cloud import storage
from pybit.unified_trading import WebSocket
from collections import deque
from finta import TA
from pybit.unified_trading import HTTP as UnifiedHTTP


class TradeMachine:

    def __init__(self, symbols, model, trade_client, model2, second_client):
        self.sub_model = model2
        self.symbol = symbols
        self.trades_buffer = []
        self.candles_1sec = []
        self.candle_2 = []
        self.candle_queue_1s = asyncio.Queue()
        self.candle_queue_2 = asyncio.Queue()
        self.candles_gcs = []
        self.candles_1min = deque(maxlen=20)
        self.candles_30sec = deque(maxlen=20)
        self.candle_queue_30s = asyncio.Queue()
        self.candle_queue_1min = asyncio.Queue()
        self._buffer_lock = threading.Lock()
        self.current_price = None
        self.last_modified = os.path.getmtime(MODEL_PATH)
        self.sub_last_modified = os.path.getmtime(MODEL_PATH2)
        self.modelXGB = model
        self.trade_client = trade_client
        self.sub_client = second_client
        self.predict_result = None
        self.sub_predict_result = None

        logging.basicConfig(level=logging.INFO)
        # Запуск WebSocket (работает в отдельном потоке)
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear"
        )
        self.ws.trade_stream(symbol=self.symbol, callback=self.handle_trade)


    async def get_available_balance(self, moneta, account=None):
        if account == '1':
            trade_client = self.trade_client
        else:
            trade_client = self.sub_client

        loop = asyncio.get_running_loop()
        try:
            wallet_info = await loop.run_in_executor(None, functools.partial(
                trade_client.get_wallet_balance,
                accountType="UNIFIED",  # или другой тип, если ты используешь classic/contract
                coin=moneta
            ))
            balance = float(wallet_info["result"]["list"][0]["totalAvailableBalance"])
            return balance
        except Exception as e:
            logging.error(f"Ошибка при получении баланса: {e}")
            return None


    async def place_sell_order(self, qty, account):
        if account == '1':
            trade_client, present_predict, nap = self.trade_client, self.predict_result, 90
        else:
            trade_client, present_predict, nap = self.sub_client, self.sub_predict_result, 50

        limit_price = round(self.current_price * (1 + present_predict / 100), 2)
        order = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(limit_price),
            "timeInForce": "GTC",
            "isPostOnly": True,
            "reduceOnly": True,
        }
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, functools.partial(trade_client.place_order, **order))
            order_id = response['result']['orderId']

            await asyncio.sleep(nap)

            await loop.run_in_executor(None, functools.partial(
                trade_client.cancel_order,
                category="linear",
                symbol=self.symbol,
                orderId=order_id
            ))
        except Exception as e:
            logging.warning(f"Не удалось отменить ордер (возможно, уже исполнен): {e}")



    async def place_buy_order(self, qty, account):
        if account == '1':
            trade_client, present_predict, nap = self.trade_client, self.predict_result, 90
        else:
            trade_client, present_predict, nap = self.sub_client, self.sub_predict_result, 50

        limit_price = round(self.current_price * (1 + present_predict / 100), 2)

        order = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Buy",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(limit_price),
            "timeInForce": "GTC",
            "isPostOnly": True
        }

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, functools.partial(trade_client.place_order, **order))
            order_id = response["result"]["orderId"]

            await asyncio.sleep(nap)

            await loop.run_in_executor(None, functools.partial(
                trade_client.cancel_order,
                category="linear",
                symbol=self.symbol,
                orderId=order_id
            ))
        except Exception as e:
            logging.warning(f"Не удалось отменить ордер (возможно, уже исполнен): {e}")


    async def execute_trade_logic(self, account=None):
        if account == '1':
            present_predict =self.predict_result
        else:
            present_predict = self.sub_predict_result
        quantity20 = max(round(20 / self.current_price, 2), 0.01)

        if present_predict <= -0.06:
            usd = await self.get_available_balance('USDT', account)
            if usd > 20:
                await self.place_buy_order(quantity20, account)

        elif present_predict >= 0.06:
            eth = await self.get_available_balance('ETH', account)
            if eth > quantity20:
                await self.place_sell_order(quantity20, account)


    async def save_to_gcs(self):
        client = storage.Client.from_service_account_json(KEYS_GCS_PATH)
        bucket = client.bucket(BUCKET_NAME)

        while True:
            if self.candles_gcs:
                current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                file_path = f"datasets/ETH-USDT_{current_date}.pkl"
                blob = bucket.blob(file_path)

                try:
                    filebuffer = io.BytesIO()
                    await asyncio.to_thread(blob.download_to_file, filebuffer)
                    filebuffer.seek(0)
                    existing_data = pd.read_pickle(filebuffer)
                except Exception:
                    existing_data = pd.DataFrame()

                new_data = pd.DataFrame(self.candles_gcs)
                self.candles_gcs.clear()

                updated_data = pd.concat([existing_data, new_data], ignore_index=True)

                filebuffer = io.BytesIO()
                updated_data.to_pickle(filebuffer)
                filebuffer.seek(0)

                await asyncio.to_thread(blob.upload_from_file, filebuffer, content_type="application/octet-stream")
            #logging.info("save_to_gcs спит ")
            await asyncio.sleep(3600)


    async def check_model_update(self):
        while True:
            current_modified = os.path.getmtime(MODEL_PATH)
            sub_current_modified = os.path.getmtime(MODEL_PATH2)
            if current_modified > self.last_modified:
                self.modelXGB = await asyncio.to_thread(functools.partial(joblib.load, MODEL_PATH))
                self.last_modified = current_modified
            if sub_current_modified > self.sub_last_modified:
                self.sub_model = await asyncio.to_thread(functools.partial(joblib.load, MODEL_PATH2))
                self.sub_last_modified = sub_current_modified
            #logging.info("check_model_update спит ")
            await asyncio.sleep(21600)


    async def preprocess_predict(self):
        while True:
            new_candle = await self.candle_queue_30s.get()
            self.candles_30sec.append(new_candle)
            ln = len(self.candles_30sec)
            if ln < 16:
                logging.info(f"30sec свечей {ln}")
                continue
            df = pd.DataFrame(self.candles_30sec)

            df['close'] = df['close'].shift(-1)

            df['SMA14'] = TA.SMA(df, 14).shift(1)
            df['RSI14'] = TA.RSI(df).shift(1)
            df['OBV14'] = TA.OBV(df).shift(1)
            df['ATR14'] = TA.ATR(df, period=14).shift(1)
            macd_df = TA.MACD(df).shift(1)
            df['MACD'] = macd_df['MACD']
            df['MACD_SIGNAL'] = macd_df['SIGNAL']
            df.dropna(inplace=True)

            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            }, inplace=True)

            y_predictors = df.drop('close', axis=1).iloc[[-1]]
            y_prediction = self.modelXGB.predict(y_predictors)[0]

            curr_price = self.current_price
            price_change = y_prediction - curr_price
            percent_change = (price_change / curr_price) * 100
            self.predict_result = percent_change

            asyncio.create_task(
                self.execute_trade_logic(account='1')
            )


    async def preprocess_1min(self):
        while True:
            new_candle = await self.candle_queue_1min.get()
            self.candles_1min.append(new_candle)
            ln = len(self.candles_1min)
            if ln < 16:
                logging.info(f"1min свечей {ln}")
                continue
            df = pd.DataFrame(self.candles_1min)

            df['close'] = df['close'].shift(-1)

            df['SMA14'] = TA.SMA(df, 14).shift(1)
            df['RSI14'] = TA.RSI(df).shift(1)
            df['OBV14'] = TA.OBV(df).shift(1)
            df['ATR14'] = TA.ATR(df, period=14).shift(1)
            macd_df = TA.MACD(df).shift(1)
            df['MACD'] = macd_df['MACD']
            df['MACD_SIGNAL'] = macd_df['SIGNAL']
            df.dropna(inplace=True)


            y_predictors = df.drop('close', axis=1).iloc[[-1]]
            y_prediction = self.sub_model.predict(y_predictors)[0]

            curr_price = self.current_price
            price_change = y_prediction - curr_price
            percent_change = (price_change / curr_price) * 100
            self.sub_predict_result = percent_change

            asyncio.create_task(
                self.execute_trade_logic(account='2')
            )


    async def forming_1min_candles(self):
        while True:
            new_candle = await self.candle_queue_2.get()
            self.candle_2.append(new_candle)

            if len(self.candle_2) < 60:
                continue
            df = pd.DataFrame(self.candle_2)
            self.candle_2.clear()
            new_candle_1min = {
                'open': df['open'].iloc[0],
                'high': df['high'].max(),
                'low': df['low'].min(),
                'close': df['close'].iloc[-1],
                'volume': df['volume'].sum(),
            }
            await self.candle_queue_1min.put(new_candle_1min)


    async def forming_30sec_candles(self):
        while True:
            new_candle = await self.candle_queue_1s.get()
            self.candles_1sec.append(new_candle)

            if len(self.candles_1sec) < 30:
                continue
            df = pd.DataFrame(self.candles_1sec)
            self.candles_1sec.clear()
            new_candle_30s = {
                'open': df['open'].iloc[0],
                'high': df['high'].max(),
                'low': df['low'].min(),
                'close': df['close'].iloc[-1],
                'volume': df['volume'].sum(),
            }
            await self.candle_queue_30s.put(new_candle_30s)


    async def forming_candles(self):
        """Формирование свечей каждую секунду"""
        while True:
            await asyncio.sleep(1)

            with self._buffer_lock:
                if not self.trades_buffer:
                    continue
                trades = self.trades_buffer.copy()
                self.trades_buffer.clear()

            trades.sort(key=lambda x: x['timestamp'])

            open_price = trades[0]['price']
            close_price = trades[-1]['price']
            self.current_price = close_price

            high_price = low_price = open_price
            volume = 0
            for trade in trades:
                price = trade['price']
                volume += trade['size']
                if price > high_price:
                    high_price = price
                if price < low_price:
                    low_price = price

            candle = {
                'timestamp': datetime.now(),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            }
            self.current_price = close_price
            await self.candle_queue_1s.put(candle)
            await self.candle_queue_2.put(candle)
            self.candles_gcs.append(candle)



    def handle_trade(self, message):
        if "data" in message:
            with self._buffer_lock:
                for trade in message["data"]:
                    self.trades_buffer.append({
                        "timestamp": int(trade["T"]) / 1000,
                        "price": float(trade["p"]),
                        "size": float(trade["v"]),
                        "side": trade["S"]
                    })

    async def run(self):
        """Запуск формирования свечей (WebSocket работает в фоне)"""
        await asyncio.gather(
            self.forming_candles(),
            self.save_to_gcs(),
            self.forming_30sec_candles(),
            self.preprocess_predict(),
            self.preprocess_1min(),
            self.check_model_update(),
            self.forming_1min_candles()
        )


client1 = UnifiedHTTP(
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_SECRET_KEY,
    testnet=False
)
desired_leverage = "10"
positions = client1.get_positions(category="linear", symbol="ETHUSDT")
current_leverage = positions["result"]["list"][0]["leverage"]
if current_leverage != desired_leverage:
    client1.set_leverage(
        category="linear",
        symbol="ETHUSDT",
        buyLeverage=desired_leverage,
        sellLeverage=desired_leverage
    )
    print(f"Leverage set to {desired_leverage}x")

client2 = UnifiedHTTP(
    api_key=BYBIT_API_KEY_sub,
    api_secret=BYBIT_SECRET_KEY_sub,
    testnet=False
)
desired_leverage = "10"
positions = client2.get_positions(category="linear", symbol="ETHUSDT")
current_leverage = positions["result"]["list"][0]["leverage"]
if current_leverage != desired_leverage:
    client2.set_leverage(
        category="linear",
        symbol="ETHUSDT",
        buyLeverage=desired_leverage,
        sellLeverage=desired_leverage
    )
    print(f"Leverage set to {desired_leverage}x")
print("Client1 class:", type(client1))
print("Client1 module:", client1.__class__.__module__)
modelXGB = joblib.load(MODEL_PATH)
second_model2 = joblib.load(MODEL_PATH2)

# ETHUSDT   BTCUSDT
if __name__ == '__main__':
    symbol = "ETHUSDT"
    collector = TradeMachine(symbols=symbol, model=modelXGB, trade_client=client1, model2=second_model2, second_client=client2)
    asyncio.run(collector.run())
