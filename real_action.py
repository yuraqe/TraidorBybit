
import asyncio
import threading
import pandas as pd
import logging
import numpy as np

from datetime import datetime, timezone, timedelta
from stable_baselines3 import PPO
from xgboost.testing.data import joblib

from verification import BYBIT_SECRET_KEY, BYBIT_API_KEY
from pybit.unified_trading import WebSocket
from collections import deque
from finta import TA
from pybit.unified_trading import HTTP as UnifiedHTTP


class ActionTest:

    def __init__(self, symbols, model, trade_client, xgb_model):
        self.symbol = symbols
        self.trades_buffer = []
        self.candles_1min = deque(maxlen=69)
        self.candle_queue_1min = asyncio.Queue()
        self._buffer_lock = threading.Lock()
        self.current_price = None
        self.modelPRO = model
        self.model_XGB = xgb_model
        self.trade_client = trade_client
        self.predict_result = None
        self.candles_skip = 0
        self.order_activated = False

        logging.basicConfig(level=logging.INFO)
        # Запуск WebSocket (работает в отдельном потоке)
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear"
        )
        self.ws.trade_stream(symbol=self.symbol, callback=self.handle_trade)

    async def margin_quit(self):
        pass


    async def check_orders(self):
        pass


    async def place_order(self, act_type, entry_price, exit_price, leverage, stop_loss):
        pass


    async def preprocess_1min_candle(self):
        while True:
            new_candle = await self.candle_queue_1min.get()
            self.candles_1min.append(new_candle)
            ln = len(self.candles_1min)
            if ln < 66:
                logging.info(f"1min свечей {ln}")
                continue
            if self.order_activated:
                await self.check_orders()
                if self.order_activated:
                    continue
            df = pd.DataFrame(self.candles_1min)

            df['original_close'] = df['close']
            df['close'] = df['close'].shift(-1)

            df['SMA14'] = TA.SMA(df, 14).shift(1)
            df['RSI14'] = TA.RSI(df).shift(1)
            df['OBV14'] = TA.OBV(df).shift(1)
            df['ATR14'] = TA.ATR(df, period=14).shift(1)
            macd_df = TA.MACD(df).shift(1)
            df['MACD'] = macd_df['MACD']
            df['MACD_SIGNAL'] = macd_df['SIGNAL']

            df.dropna(inplace=True)

            features = ['open', 'high', 'low', 'volume', 'SMA14', 'RSI14', 'OBV14', 'ATR14', 'MACD', 'MACD_SIGNAL']
            X_live_np = df[features].to_numpy(dtype=np.float32)
            pred1 = model_xgb.estimators_[0].predict(X_live_np)
            pred2 = model_xgb.estimators_[1].predict(X_live_np)
            pred5 = model_xgb.estimators_[2].predict(X_live_np)
            pred10 = model_xgb.estimators_[3].predict(X_live_np)
            logr10 = model_xgb.estimators_[4].predict(X_live_np)
            current_price_np = df['original_close'].to_numpy(dtype=np.float32)
            obs = np.stack([
                (current_price_np + pred1).round(5),
                (current_price_np + pred2).round(5),
                (current_price_np + pred5).round(5),
                (current_price_np + pred10).round(5),
                logr10.round(5),
                current_price_np,
                X_live_np['high'],
                X_live_np['low'],
                X_live_np['volume'],], axis=1)

            obs = obs[-50:].reshape(1, 50, 9)
            action, _ = self.modelPRO.predict(obs)
            logging.info(f"action: 0:{action[0]}, 1:{action[1]}, 2:{action[2]}, 3:{action[3]}, 4:{action[4]}")

            act_type = int(np.clip(np.round(action[0]), 0, 2))
            if act_type != 2:
                current_price = self.current_price
                entry_offset = np.clip(action[1], 0.001, 0.01)
                exit_offset = np.clip(action[2], 0.0041, 0.3)

                if act_type == 0:  # Лонг
                    entry_price = round(current_price * (1 - entry_offset), 4)
                    exit_price = round(entry_price * (1 + exit_offset), 4)
                else:  # Шорт
                    entry_price = round(current_price * (1 + entry_offset), 4)
                    exit_price = round(entry_price * (1 - exit_offset), 4)

                raw_leverage = np.clip(action[3], 0.1, 1)
                leverage = int((raw_leverage - 0.1) / 0.9 * 99 + 1)
                stop_loss = int(action[4] * 10)
                self.candles_skip = stop_loss # каждый свеча будет -1

                await self.place_order(act_type, entry_price, exit_price, leverage, stop_loss)

    async def forming_candles(self):
        """Формирование свечей каждую секунду"""
        while True:
            await self.sleep_until_next_minutes()

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
                'timestamp': datetime.now(timezone.utc),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            }
            logging.info(f"свеча готова {candle}")
            await self.candle_queue_1min.put(candle)

    async def sleep_until_next_minutes(self):
        now = datetime.now(timezone.utc)
        next_tick = (now.replace(microsecond=0) + timedelta(minutes=1))
        await asyncio.sleep((next_tick - now).total_seconds())


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
            self.preprocess_1min_candle(),
        )


client1 = UnifiedHTTP(
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_SECRET_KEY,
    testnet=False
)



print("Client1 class:", type(client1))
print("Client1 module:", client1.__class__.__module__)
model_pro = PPO.load("test_pro_ethmodel")
model_xgb = joblib.load('path')

# ETHUSDT   BTCUSDT
if __name__ == '__main__':
    symbol = "HIGHUSDT"
    collector = ActionTest(symbols=symbol, model=model_pro, trade_client=client1, xgb_model=model_xgb)
    asyncio.run(collector.run())
