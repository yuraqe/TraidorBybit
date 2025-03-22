import time
import io
import pandas as pd
import joblib
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from verification import MODEL_PATH, KEYS_GCS_PATH, BUCKET_NAME
from finta import TA

def online_learning():
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

    last_hour = datetime.now(timezone.utc) - timedelta(hours=1)
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


if __name__ == '__main__':
    online_learning()