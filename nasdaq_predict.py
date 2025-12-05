import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ---------------------------------------------------------
# 텔레그램 전송 함수
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------
def main():
    print(">>> [Nasdaq Final Algo] 데이터 수집 및 분석 시작...")
    ticker = 'NQ=F' # 나스닥 100 선물
    
    # 1. 데이터 수집 (2018 ~ 현재)
    try:
        df = yf.download(ticker, start="2018-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df = df['Close']
        else: df = df[['Close']]
        df.columns = ['Close']
        df = df.ffill().dropna()
    except Exception as e:
        send_telegram_message(f"⚠️ 데이터 수집 에러: {e}")
        return

    # 2. 핵심 지표 계산
    # A. 대세 추세선 (60일 이동평균선)
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # B. 로그 수익률 (가격 격차 해소용 학습 목표)
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()

    last_price = float(df['Close'].iloc[-1])
    last_ma60 = float(df['MA60'].iloc[-1])
    last_date = df.index[-1].strftime('%Y-%m-%d')

    # 3. 데이터 전처리 (AI 학습용)
    # 수익률은 -0.05 ~ +0.05 사이의 작은 값이므로 스케일링 필수 (-1 ~ 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df['Return'].values.reshape(-1, 1))

    # Lookback: 과거 60일 패턴을 보고 내일 등락률 예측
    time_step = 60
    X_all, y_all = [], []
    for i in range(len(scaled_data) - time_step):
        X_all.append(scaled_data[i:(i + time_step), 0])
        y_all.append(scaled_data[i + time_step, 0])

    X_all = np.array(X_all).reshape(-1, time_step, 1)
    y_all = np.array(y_all)

    # 4. 모델 학습 (Daily Retraining)
    # 매일 최신 데이터로 뇌를 갈아끼워 '최신 트렌드' 반영
    print(">>> AI 모델 학습 중 (LSTM)...")
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1)) # 출력: 내일의 예상 등락률
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 데이터가 적으므로(약 1800개) Epoch 15회면 충분
    model.fit(X_all, y_all, epochs=15, batch_size=32, verbose=0)

    # 5. 내일 예측 수행
    last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred_scaled = model.predict(last_60_days)
    pred_return_log = float(scaler.inverse_transform(pred_scaled)[0][0])
    
    # 로그 수익률 -> 퍼센트 변환
    pred_pct = (np.exp(pred_return_log) - 1) * 100
    
    # 6. [최종 알고리즘] 포지션 결정 로직
    # 백테스트에서 검증된 'Signal Hold' + 'Sniper' 전략
    
    action = ""
    emoji = ""
    comment = ""
    
    # Case A: AI가 상승 확신 (0% 초과)
    if pred_pct > 0.000:
        emoji = "🔥"
        action = "*LONG (매수/홀딩)*"
        comment = "AI 상승 예측. 기존 매수자는 '홀딩', 신규는 '진입' 가능."

    # Case B: AI가 하락 확신 (-0.2% 미만)
    elif pred_pct < -0.2:
        # 하지만 대세 상승장(60일선 위)이라면? -> 숏 금지, 그냥 버티기(1배)
        if last_price > last_ma60:
            emoji = "🛡️"
            action = "*WEAK HOLD (관망)*"
            comment = "단기 조정 예상되나 대세 상승장이므로 '버티기' 추천."
        # 대세 하락장(60일선 아래)이라면? -> 전량 매도(현금화)
        else:
            emoji = "⚠️"
            action = "*CASH (전량 매도)*"
            comment = "하락장 진입 + AI 하락 예측. 현금화 후 대피."
            
    # Case C: 애매한 구간
    else:
        emoji = "👀"
        action = "*WAIT (관망)*"
        comment = "뚜렷한 방향성 없음. 기존 포지션 유지."

    # 7. 메시지 전송
    msg = f"{emoji} [Nasdaq AI Signal]\n"
    msg += f"📅 기준: {last_date}\n\n"
    msg += f"💰 현재가: {last_price:,.2f}\n"
    msg += f"📏 추세선(60일): {last_ma60:,.2f}\n"
    msg += f"🔮 AI 예측: {pred_pct:+.2f}%\n\n"
    msg += f"📢 시그널: {action}\n"
    msg += f"💡 코멘트: {comment}\n"
    msg += f"----------------------------\n"
    msg += f"※ 필수: 진입 시 손절 -1%(-5%) 설정"

    print(msg)
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
