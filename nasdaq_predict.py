import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ---------------------------------------------------------
# 텔레그램 전송
# ---------------------------------------------------------
def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ 텔레그램 설정 오류")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"전송 실패: {e}")

# ---------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------
def main():
    print(">>> [Nasdaq Best Model (3x)] 데이터 수집 및 분석 시작...")
    ticker = 'NQ=F'
    
    # 1. 데이터 수집
    try:
        df = yf.download(ticker, start="2018-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df = df['Close']
        else: df = df[['Close']]
        df.columns = ['Close']
        df = df.ffill().dropna()
    except Exception as e:
        send_telegram_message(f"⚠️ 데이터 에러: {e}")
        return

    # 2. 지표 계산
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()

    last_price = float(df['Close'].iloc[-1])
    last_ma60 = float(df['MA60'].iloc[-1])
    last_date = df.index[-1].strftime('%Y-%m-%d')

    # 3. 전처리 & 데이터셋
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df['Return'].values.reshape(-1, 1))

    time_step = 60
    X_all, y_all = [], []
    for i in range(len(scaled_data) - time_step):
        X_all.append(scaled_data[i:(i + time_step), 0])
        y_all.append(scaled_data[i + time_step, 0])

    X_all = np.array(X_all).reshape(-1, time_step, 1)
    y_all = np.array(y_all)

    # 4. 모델 학습 (Daily Retraining)
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_all, y_all, epochs=15, batch_size=32, verbose=0)

    # 5. 예측
    last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred_scaled = model.predict(last_60_days)
    pred_return_log = float(scaler.inverse_transform(pred_scaled)[0][0])
    pred_pct = (np.exp(pred_return_log) - 1) * 100
    
    # 6. [최적화된 전략] Hybrid Mode (3x Leverage)
    # Grid Search 결과 1위: 3배 레버리지 / -3% 손절
    
    # 임계값 (AI 성향 반영)
    buy_threshold = 0.000   
    sell_threshold = -0.05  
    
    emoji = "🤔"
    action = "HOLD (관망)"
    comment = "방향성 탐색 중."
    leverage_guide = "1x (기본)"

    # --- 포지션 결정 로직 ---
    if pred_pct > buy_threshold:
        emoji = "🔥"
        action = "*STRONG BUY (3x 진입/홀딩)*"
        comment = "AI 상승 확신. 3배 레버리지 적극 활용 구간."
        leverage_guide = "3x (TQQQ / 선물 3배수)"
        
    elif pred_pct < sell_threshold:
        # 하락 예측 시
        if last_price > last_ma60:
            emoji = "🛡️"
            action = "*WEAK HOLD (1x 버티기)*"
            comment = "AI 하락 예측이나 대세 상승장(MA60 위). 3배는 팔고 1배로 방어."
            leverage_guide = "1x (QLD 줄이거나 QQQ로 이동)"
        else:
            emoji = "⚠️"
            action = "*CASH (전량 매도)*"
            comment = "📉 대세 하락장 + AI 하락 예측. 뒤도 돌아보지 말고 현금화."
            leverage_guide = "0x (현금 100%)"
    else:
        # 애매할 때
        if last_price > last_ma60:
            emoji = "👀"
            action = "*HOLD (1x 유지)*"
            comment = "상승 추세 유지 중. 무리하지 말고 시장 흐름 편승."
            leverage_guide = "1x (기본)"
        else:
            emoji = "☁️"
            action = "*WAIT (관망)*"
            comment = "하락 추세 중 반등 미약. 현금 보유 추천."
            leverage_guide = "0x (현금)"

    # 7. 메시지 전송
    msg = f"{emoji} [Nasdaq AI Strategy: 3x Hybrid]\n"
    msg += f"📅 기준: {last_date}\n\n"
    msg += f"💰 현재가: {last_price:,.2f}\n"
    msg += f"📏 추세선: {last_ma60:,.2f}\n"
    msg += f"🔮 AI 예측: {pred_pct:+.3f}%\n\n"
    msg += f"📢 시그널: {action}\n"
    msg += f"🎰 추천 레버리지: {leverage_guide}\n"
    msg += f"💡 코멘트: {comment}\n"
    msg += f"----------------------------\n"
    msg += f"🚨 필수 안전장치: 자산 대비 -3% 손절\n"
    msg += f"(3배 레버리지 기준 지수 -1% 하락 시 매도)"

    print(msg)
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
