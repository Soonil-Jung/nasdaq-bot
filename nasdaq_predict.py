# 파일명: nasdaq_predict.py
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 텔레그램 전송 함수
def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("❌ 텔레그램 설정 오류: Secrets를 확인하세요.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"전송 실패: {e}")

def main():
    print(">>> [Nasdaq-100 LSTM] 데이터 수집 및 분석 시작...")
    
    # 1. 나스닥 100 선물 티커
    ticker = 'NQ=F' 
    
    try:
        # 데이터 수집
        df = yf.download(ticker, start="2018-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]
        df.columns = ['Close']
        df = df.ffill().dropna()
    except Exception as e:
        send_telegram_message(f"⚠️ 나스닥 데이터 수집 에러: {e}")
        return

    last_price = float(df['Close'].iloc[-1])
    last_date = df.index[-1].strftime('%Y-%m-%d')

    # 2. 전처리 (0~1 정규화)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

    # Lookback: 60일 (과거 60일 패턴을 보고 내일 예측)
    time_step = 60
    X_all, y_all = [], []
    for i in range(len(scaled_data) - time_step):
        X_all.append(scaled_data[i:(i + time_step), 0])
        y_all.append(scaled_data[i + time_step, 0])

    X_all = np.array(X_all).reshape(-1, time_step, 1)
    y_all = np.array(y_all)

    # 3. 모델 학습 (LSTM)
    # Github Actions 시간 제한을 고려해 Epoch 15회 설정
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_all, y_all, epochs=15, batch_size=32, verbose=0)

    # 4. 내일 가격 예측
    last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred_scaled = model.predict(last_60_days)
    pred_price = float(scaler.inverse_transform(pred_scaled)[0][0])

    # 5. 결과 리포트 작성
    diff = pred_price - last_price
    pct = (diff / last_price) * 100
    
    emoji = "🤔"
    action = "HOLD (관망)"
    
    # 나스닥은 변동성이 크므로 기준을 0.4%로 설정 (금은 0.3%)
    threshold = 0.4 
    
    if pct > threshold:
        emoji = "🚀"
        action = "*STRONG BUY (매수)*"
    elif pct < -threshold:
        emoji = "📉"
        action = "*STRONG SELL (매도)*"

    msg = f"{emoji} [Nasdaq-100 Futures AI]\n"
    msg += f"📅 기준: {last_date}\n\n"
    msg += f"💰 현재가: {last_price:,.2f}\n"
    msg += f"🔮 예측가: {pred_price:,.2f}\n"
    msg += f"📊 변동폭: {diff:+,.2f} ({pct:+.2f}%)\n\n"
    msg += f"📢 포지션: {action}"

    print(msg)
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
