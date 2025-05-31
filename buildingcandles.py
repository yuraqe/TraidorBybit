import logging
import threading
import asyncio
import io
import pandas as pd

from pybit.unified_trading import WebSocket
from datetime import datetime, timezone, timedelta
from google.cloud import storage
from verification import BUCKET_NAME, KEYS_GCS_PATH



class CollectCandles:

    def __init__(self, symbols):
        self.candles_gcs = []
        self.symbol = symbols
        logging.basicConfig(level=logging.INFO)
        self.current_price = None

        # Запуск WebSocket (работает в отдельном потоке)
        self._buffer_lock = threading.Lock()
        self.trades_buffer = []
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear"
        )
        self.ws.trade_stream(symbol=self.symbol, callback=self.handle_trade)


    async def sleep_until_next_hour(self):
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        await asyncio.sleep((next_hour - now).total_seconds())


    async def sleep_until_next_second(self):
        now = datetime.now(timezone.utc)
        next_tick = (now.replace(microsecond=0) + timedelta(seconds=1))
        await asyncio.sleep((next_tick - now).total_seconds())


    async def save_to_gcs(self):
        client = storage.Client.from_service_account_json(KEYS_GCS_PATH)
        bucket = client.bucket(BUCKET_NAME)

        while True:
            if self.candles_gcs:
                current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                file_path = f"datasets/HIGH-USDT_{current_date}.pkl"
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
            await self.sleep_until_next_hour()


    async def forming_candles(self):
        """Формирование свечей каждую секунду"""
        while True:
            await self.sleep_until_next_second()

            with self._buffer_lock:
                trades = self.trades_buffer.copy()
                self.trades_buffer.clear()
            if trades:
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
            else:
                price = self.current_price or 0.0
                open_price = close_price = high_price = low_price = price
                volume = 0.0

            candle = {
                'timestamp': datetime.now(timezone.utc),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            }
            self.candles_gcs.append(candle)


    async def run(self):
        """Запуск формирования свечей (WebSocket работает в фоне)"""
        await asyncio.gather(
            self.forming_candles(),
            self.save_to_gcs())

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


if __name__ == '__main__':
    symbol = "HIGHUSDT"
    collector = CollectCandles(symbols=symbol)
    asyncio.run(collector.run())