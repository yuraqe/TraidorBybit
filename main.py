
import asyncio
import io
import os
import joblib
import threading
import pandas as pd
import functools
import logging

from datetime import datetime, timezone
from verification import BUCKET_NAME, KEYS_GCS_PATH, MODEL_PATH, BYBIT_SECRET_KEY, BYBIT_API_KEY
from google.cloud import storage
from pybit.unified_trading import WebSocket
from collections import deque
from finta import TA
from pybit.unified_trading import HTTP


class TradeMachine:
    FEE_BUY = 0.00055    # 0.055 от маркет покупки
    FEE_SELL = 0.0002    # 0.02 от лимтной продажи

    def __init__(self, symbols, model, trade_client):
        self.symbol = symbols
        self.trades_buffer = []
        self.candles_1sec = []
        self.candle_queue_1s = asyncio.Queue()
        self.candles_gcs = []
        self.candles_30sec = deque(maxlen=15)
        self.candle_queue_30s = asyncio.Queue()
        self._buffer_lock = threading.Lock()
        self.current_price = None
        self.last_modified = os.path.getmtime(MODEL_PATH)
        self.modelXGB = model
        self.trade_client = trade_client
        self.usd_bank = 100
        self.eth_contract = 0
        self.predict_result = None
        self.STOP_LOSS_PERCENT = 0.001  # 0.1% убытка

        logging.basicConfig(level=logging.INFO)
        # Запуск WebSocket (работает в отдельном потоке)
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear"
        )
        self.ws.trade_stream(symbol=self.symbol, callback=self.handle_trade)

    async def cleanup_old_orders(self):
        while True:
            try:
                orders = self.trade_client.get_open_orders(category="linear", symbol="ETHUSDT")
                active_orders = orders["result"]["list"]

                if len(active_orders) <= 1:
                    logging.info("Нет старых ордеров для отмены")
                    await asyncio.sleep(160)
                    continue

                active_orders.sort(key=lambda o: int(o["createdTime"]), reverse=True)
                orders_to_cancel = active_orders[1:]

                for order in orders_to_cancel:
                    self.trade_client.cancel_order(
                        category="linear",
                        symbol="ETHUSDT",
                        orderId=order["orderId"]
                    )
                    logging.info(f"Старый ордер отменён: {order['orderId']}")

            except Exception as e:
                logging.error(f"Ошибка при очистке старых ордеров: {e}")
            await asyncio.sleep(160)


    async def place_sell_order(self, qty, limit_price):
        if not self.eth_contract < qty:
            order = {
                "category": "linear",
                "symbol": self.symbol,
                "side": "Sell",
                "orderType": "Limit",
                "qty": str(qty),
                "price": str(limit_price),
                "reduceOnly": True,  # чтобы не открыть шорт
                "timeInForce": "GoodTillCancel"
            }
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, functools.partial(self.trade_client.place_order, **order))
                #logging.info(f"Продажа ETH лимитным ордером размещена: {qty} по цене {limit_price}")
            except Exception as e:
                logging.error(f"Ошибка при размещении ордера на продажу: {e}")

            self.usd_bank += 2 * (1 - self.FEE_SELL)
            self.eth_contract -= qty
            logging.info(f"Продал ETH: {qty:.4f}")
        else:
            logging.warning(f"Недостаточно ETH для продажи: есть {self.eth_contract}, требуется {qty}")


    async def place_buy_order(self, qty):
        order = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GoodTillCancel"
        }

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, functools.partial(self.trade_client.place_order, **order))
            #logging.info(f"Ордер на покупку размещён: {order}")
        except Exception as e:
            logging.error(f"Ошибка при размещении ордера на покупку: {e}")


        self.usd_bank -= 2 * (1 + self.FEE_BUY)
        self.eth_contract += qty
        logging.info(f"Купил ETH: {qty:.4f}")

        # --- STOP-LOSS ---
        stop_loss_price = round(self.current_price * (1 - self.STOP_LOSS_PERCENT), 2)
        stop_order = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "triggerPrice": str(stop_loss_price),
            "triggerDirection": 2,
            "reduceOnly": True,
            "closeOnTrigger": True,
            "timeInForce": "GoodTillCancel"
        }

        try:
            await loop.run_in_executor(None, functools.partial(self.trade_client.place_order, **stop_order))
            #logging.info(f"Стоп-лосс размещён: {stop_order}")
        except Exception as e:
            logging.error(f"Ошибка при размещении стоп-лосса: {e}")

        # --- TAKE-PROFIT ---
        take_profit_price = round(self.current_price * (1 + self.predict_result / 100), 2)

        tp_order = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(take_profit_price),
            "reduceOnly": True,
            "timeInForce": "PostOnly"
        }

        try:
            await loop.run_in_executor(None, functools.partial(self.trade_client.place_order, **tp_order))
            #logging.info(f"Тейк-профит размещён: {tp_order}")
        except Exception as e:
            logging.error(f"Ошибка при размещении тейк-профита: {e}")


    async def execute_trade_logic(self):
        quantity25 = max(round(25 / self.current_price, 2), 0.01)
        quantity20 = max(round(20 / self.current_price, 2), 0.01)

        if self.predict_result >= 0.15:
            await self.place_buy_order(quantity25)
        elif self.predict_result >= 0.09:
            await self.place_buy_order(quantity20)

        elif self.predict_result <= -0.15 and self.eth_contract > quantity25:
            limit_price = round(self.current_price * (1 - abs(self.predict_result) / 100), 2)
            await self.place_sell_order(quantity25, limit_price)
        elif self.predict_result <= -0.09 and self.eth_contract > quantity20:
            limit_price = round(self.current_price * (1 - abs(self.predict_result) / 100), 2)
            await self.place_sell_order(quantity20, limit_price)

        else:
            logging.info("Сигнал недостаточно сильный для входа в сделку.")


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
            if current_modified > self.last_modified:
                self.modelXGB = await asyncio.to_thread(functools.partial(joblib.load, MODEL_PATH))
                self.last_modified = current_modified
            #logging.info("check_model_update спит ")
            await asyncio.sleep(3600)


    async def preprocess_predict(self):
        while True:
            new_candle = await self.candle_queue_30s.get()
            self.candles_30sec.append(new_candle)
            ln = len(self.candles_30sec)
            if ln < 15:
                #logging.info(f"30sec свечей {ln}")
                continue
            df = pd.DataFrame(self.candles_30sec)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            }, inplace=True)
            df['close'] = df['close'].shift(-1)
            df = df.dropna(subset=['close'])

            df['SMA14'] = TA.SMA(df, 14).shift(1)
            df['RSI14'] = TA.RSI(df).shift(1)
            df['OBV14'] = TA.OBV(df).shift(1)
            df['ATR14'] = TA.ATR(df, period=14).shift(1)
            macd_df = TA.MACD(df).shift(1)
            df['MACD'] = macd_df['MACD']
            df['MACD_SIGNAL'] = macd_df['SIGNAL']

            y_predictors = df.drop(['close'], axis=1).iloc[-1].to_frame().T
            y_prediction = self.modelXGB.predict(y_predictors)

            curr_price = self.current_price
            predicted_prices = sum([float(i) for i in y_prediction])
            price_change = predicted_prices - curr_price
            percent_change = (price_change / curr_price) * 100
            self.predict_result = percent_change

            asyncio.create_task(
                self.execute_trade_logic()
            )


    async def forming_30sec_candles(self):
        while True:
            new_candle = await self.candle_queue_1s.get()
            self.candles_1sec.append(new_candle)
            if len(self.candles_1sec) < 30:
                continue
            df = pd.DataFrame(list(self.candles_1sec))
            self.candles_1sec.clear()

            df.set_index('timestamp', inplace=True)

            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

            resampled = df.resample('30s').agg(agg_dict)

            new_candle = resampled.iloc[0].to_dict()
            await self.candle_queue_30s.put(new_candle)




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
            self.check_model_update(),
            self.cleanup_old_orders()
        )


client = HTTP(
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_SECRET_KEY,
    testnet=False
)
desired_leverage = "1"
positions = client.get_positions(category="linear", symbol="ETHUSDT")
current_leverage = positions["result"]["list"][0]["leverage"]
if current_leverage != desired_leverage:
    client.set_leverage(
        category="linear",
        symbol="ETHUSDT",
        buyLeverage=desired_leverage,
        sellLeverage=desired_leverage
    )
    print(f"Leverage set to {desired_leverage}x")

modelXGB = joblib.load(MODEL_PATH)

# ETHUSDT   BTCUSDT
if __name__ == '__main__':
    symbol = "ETHUSDT"
    collector = TradeMachine(symbols=symbol, model=modelXGB, trade_client=client)
    asyncio.run(collector.run())
