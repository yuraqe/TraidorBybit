import io
import pandas as pd
import joblib
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from verification import MODEL_PATH, KEYS_GCS_PATH, BUCKET_NAME, MODEL_PATH2
from finta import TA
import numpy as np
from pybit.unified_trading import HTTP
import logging


def online_learning():
    try:
        client = storage.Client.from_service_account_json(KEYS_GCS_PATH)
        bucket = client.bucket(BUCKET_NAME)
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_path = f"datasets/ETH-USDT_{current_date}.pkl"
        blob = bucket.blob(file_path)

        filebuffer = io.BytesIO()
        blob.download_to_file(filebuffer)
        filebuffer.seek(0)

        df = pd.read_pickle(filebuffer)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        last_hour = datetime.now(timezone.utc) - timedelta(hours=6)
        df = df[df['timestamp'] >= last_hour]
        df = df.sort_values(by='timestamp')
        df.set_index('timestamp', inplace=True)

        orce = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df = df.resample('30s').agg(orce)

        df['close'] = df['close'].shift(-1)
        df = df.dropna(subset=['сlose'])

        df['SMA14'] = TA.SMA(df, 14).shift(1)
        df['RSI14'] = TA.RSI(df).shift(1)
        df['OBV14'] = TA.OBV(df).shift(1)
        df['ATR14'] = TA.ATR(df, period=14).shift(1)
        macd_df = TA.MACD(df).shift(1)
        df['MACD'] = macd_df['MACD']
        df['MACD_SIGNAL'] = macd_df['SIGNAL']

        df.fillna(0, inplace=True)
        df = df.reset_index(drop=True)

        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        X = df.drop(['Close'], axis=1)
        y = df['Close']

        model = joblib.load(MODEL_PATH)
        model.fit(X, y, xgb_model=model.get_booster())

        joblib.dump(model, MODEL_PATH)
    except Exception as e:
        logging.error(f"Ошибка при работе с ордером на продажу:{e}")


def sub_online_learn(ses):
    try:
        current_date = datetime.now(timezone.utc)
        end_date = current_date - timedelta(hours=6)
        current_ms = np.int64(current_date.timestamp() * 1000)
        end_ms = np.int64(end_date.timestamp() * 1000)
        a = ses.get_kline(
            category="linear",
            symbol="ETHUSDT",
            interval="1",
            start=end_ms,
            end=current_ms,
            limit=360   # 60 * 6
        )
        df = pd.DataFrame(a['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        df = df.drop(['timestamp', 'turnover'], axis=1)
        df = df.astype({
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float,
        })
        df['close'] = df['close'].shift(-1)

        df['SMA14'] = TA.SMA(df, 14).shift(1)
        df['RSI14'] = TA.RSI(df).shift(1)
        df['OBV14'] = TA.OBV(df).shift(1)
        df['ATR14'] = TA.ATR(df, period=14).shift(1)
        macd_df = TA.MACD(df).shift(1)
        df['MACD'] = macd_df['MACD']
        df['MACD_SIGNAL'] = macd_df['SIGNAL']
        df = df.dropna()

        X = df.drop('close', axis=1)
        y = df['close']
        model = joblib.load(MODEL_PATH2)
        model.fit(X, y, xgb_model=model.get_booster())
        joblib.dump(model, MODEL_PATH2)
    except Exception as e:
        logging.error(f"Ошибка при работе с ордером на продажу:{e}")

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    session = HTTP(testnet=False)
    online_learning()
    sub_online_learn(ses=session)