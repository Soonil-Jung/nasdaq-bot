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
    print(">>> [Nasdaq AI Pro] 데이터 수집 및 분석 시작...")
    ticker = 'NQ=F'
    
    # 1. 데이터 수집 (나스닥 + VIX)
    try:
        # 나스닥 선물
        df = yf.download(ticker, start="2018-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df = df['Close']
        else: df = df[['Close']]
        df.columns = ['Close']
        
        # 공포지수 (VIX)
        vix = yf.download('^VIX', start="2018-01-01", progress=False)['Close']
        if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
        
        # 데이터 병합
        df['VIX'] = vix.reindex(df.index).ffill()
        df = df.ffill().dropna()
        
    except Exception as e:
        send_telegram_message(f"⚠️ 데이터 에러: {e}")
        return

    # 2. 지표 계산
    # A. 추세선 (MA60)
    df['MA60'] = df['Close'].rolling(window=60).mean()
    # B. RSI (14일)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # C. 학습용 수익률
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df = df.dropna()

    last_price = float(df['Close'].iloc[-1])
    last_ma60 = float(df['MA60'].iloc[-1])
    last_vix = float(df['VIX'].iloc[-1])
    last_rsi = float(df['RSI'].iloc[-1])
    last_date = df.index[-1].strftime('%Y-%m-%d')

    # 3. 전처리 & AI 학습
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df['Return'].values.reshape(-1, 1))

    time_step = 60
    X_all, y_all = [], []
    for i in range(len(scaled_data) - time_step):
        X_all.append(scaled_data[i:(i + time_step), 0])
        y_all.append(scaled_data[i + time_step, 0])

    X_all = np.array(X_all).reshape(-1, time_step, 1)
    y_all = np.array(y_all)

    # 모델 학습
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_all, y_all, epochs=15, batch_size=32, verbose=0)

    # 예측
    last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred_scaled = model.predict(last_60_days)
    pred_return_log = float(scaler.inverse_transform(pred_scaled)[0][0])
    pred_pct = (np.exp(pred_return_log) - 1) * 100
    
    # 4. [Pro 전략] 필터링 적용 포지션 결정
    # 기본값
    action = "HOLD (관망)"
    emoji = "🤔"
    comment = "방향성 탐색 중."
    leverage_guide = "1x (기본)"
    
    # 기준값
    buy_thresh = 0.000
    sell_thresh = -0.05
    
    # --- 로직 분기 ---
    
    # [특수 상황 1] 공포지수 폭발 (VIX > 28) -> 무조건 1배 방어
    if last_vix > 28:
        emoji = "🌪️"
        action = "*WEAK HOLD (VIX 경보)*"
        comment = f"공포지수 급등({last_vix:.1f}). 예측 무시하고 1배수로 리스크 관리."
        leverage_guide = "1x (3배수 금지)"
        
    # [특수 상황 2] RSI 과열 (RSI > 75) -> 신규 매수 금지
    elif last_rsi > 75 and pred_pct > 0:
        emoji = "🔥"
        action = "*HOLD (과매수 구간)*"
        comment = f"상승세이나 RSI 과열({last_rsi:.1f}). 신규 진입 자제, 보유 물량만 홀딩."
        leverage_guide = "1x ~ 2x (보유)"
        
    # [일반 상황] AI 예측 따름
    elif pred_pct > buy_thresh:
        emoji = "🚀"
        action = "*STRONG BUY (3x 진입)*"
        comment = "AI 상승 확신 + 지표 안정적. 3배 레버리지 적극 활용."
        leverage_guide = "3x (TQQQ / 선물)"
        
    elif pred_pct < sell_thresh:
        if last_price > last_ma60:
            emoji = "🛡️"
            action = "*WEAK HOLD (1x 버티기)*"
            comment = "AI 하락 예측이나 대세 상승장. 3배 -> 1배로 축소."
            leverage_guide = "1x (안전 자산)"
        else:
            emoji = "⚠️"
            action = "*CASH (전량 매도)*"
            comment = "📉 대세 하락장 + AI 하락 예측. 즉시 현금화."
            leverage_guide = "0x (현금 100%)"
    else:
        # 애매함
        if last_price > last_ma60:
            emoji = "👀"
            action = "*HOLD (1x 유지)*"
            comment = "추세 양호. 무리하지 말고 시장 흐름 편승."
            leverage_guide = "1x (기본)"
        else:
            emoji = "☁️"
            action = "*WAIT (관망)*"
            comment = "하락 추세 지속. 진입 보류."
            leverage_guide = "0x (현금)"

    # 5. 메시지 전송
    msg = f"{emoji} [Nasdaq AI Pro: 3x Hybrid]\n"
    msg += f"📅 기준: {last_date}\n\n"
    msg += f"💰 현재가: {last_price:,.2f}\n"
    msg += f"📊 VIX: {last_vix:.2f} | RSI: {last_rsi:.1f}\n"
    msg += f"🔮 AI 예측: {pred_pct:+.3f}%\n\n"
    msg += f"📢 시그널: {action}\n"
    msg += f"🎰 추천 레버리지: {leverage_guide}\n"
    msg += f"💡 코멘트: {comment}\n"
    msg += f"----------------------------\n"
    msg += f"🚨 필수: 자산 대비 -3% 손절 설정"

    print(msg)
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
