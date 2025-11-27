import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from ta.trend import IchimokuIndicator
from ta.momentum import RSIIndicator
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from GoogleNews import GoogleNews
from datetime import datetime, timedelta

# ======================================================
# ▼▼▼ 사용자 설정 정보 ▼▼▼
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(Trend Ver) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except:
            pass
        self.keywords = ['Jerome Powell', 'Donald Trump', 'Fed Rate', 'Recession', 'Nasdaq']

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try: requests.post(url, data=data)
        except: pass

    def get_news_sentiment(self):
        try:
            googlenews = GoogleNews(lang='en', period='1d')
            total_score = 0
            count = 0
            for keyword in self.keywords:
                googlenews.clear()
                googlenews.search(keyword)
                results = googlenews.results(sort=True)
                if not results: continue
                for item in results[:2]:
                    try:
                        res = self.nlp(item['title'][:512])[0]
                        score = res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0
                        total_score += score
                        count += 1
                    except: continue
            return total_score / count if count > 0 else 0
        except: return 0

    def get_market_data(self):
        try:
            # VIX(공포지수) 추가 다운로드
            df = yf.download('NQ=F', period='5d', interval='1h', progress=False)
            df_vol = yf.download('QQQ', period='5d', interval='1h', progress=False)
            vix = yf.download('^VIX', period='5d', interval='1h', progress=False) # 공포지수

            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
            if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

            if df.empty: return pd.DataFrame()

            df = df[['High', 'Low', 'Close']].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df_vol.index = pd.to_datetime(df_vol.index).tz_localize(None)
            vix.index = pd.to_datetime(vix.index).tz_localize(None)

            df['Volume'] = df_vol['Volume'].reindex(df.index).fillna(0)
            # VIX 데이터 병합 (결측치는 앞뒤 값으로 채움)
            df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill().fillna(20.0)
            
            return df.dropna(subset=['Close'])
        except: return pd.DataFrame()

    def analyze_danger(self):
        df = self.get_market_data()
        if df.empty: return

        # 1. 기술적 지표 계산
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        
        # 2. 변화량(Trend) 계산 [사용자 요청 반영]
        current_close = df['Close'].iloc[-1]
        
        # VIX 추세 (현재값 - 5시간 평균)
        current_vix = df['VIX'].iloc[-1]
        vix_ma5 = df['VIX'].rolling(window=5).mean().iloc[-1]
        vix_trend = current_vix - vix_ma5 
        
        # RSI 추세 (현재값 - 직전값)
        rsi_series = RSIIndicator(close=df['Close'], window=14).rsi()
        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        rsi_trend = current_rsi - prev_rsi

        news_score = self.get_news_sentiment()
        
        # 3. 위험 점수 산출
        danger_score = 0
        reasons = []
        
        # [A] 구름대 이탈
        if current_close < span_a: 
            danger_score += 30
            reasons.append("☁️ 구름대 이탈")
        
        # [B] 거래량 폭증
        avg_vol = df['Vol_MA20'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else df['Volume'].iloc[-1] / avg_vol
        if vol_ratio > 1.5:
            danger_score += 20
            reasons.append(f"📢 거래량 급증 ({vol_ratio:.1f}배)")
            
        # [C] 공포지수 급상승 (Trend) ★
        # VIX가 평균보다 높고, 계속 상승 중이라면?
        if vix_trend > 0.5: # 공포가 확산 중
            danger_score += 20
            reasons.append(f"😱 공포지수 상승세 (VIX: {current_vix:.1f}, 추세: ↗)")
            
        # [D] 매수심리 급랭 (Trend) ★
        # RSI가 하락 추세라면?
        if rsi_trend < -3:
            danger_score += 15
            reasons.append(f"📉 매수심리 위축 (RSI변화: {rsi_trend:.1f})")

        # [E] 뉴스 악재
        if news_score < -0.2: 
            danger_score += 15
            reasons.append(f"📰 뉴스 악재 ({news_score:.2f})")

        # 메시지 전송
        status = '🔴 위험 (매도)' if danger_score >= 70 else '🟡 주의 (관망)' if danger_score >= 40 else '🟢 안정 (매수)'
        
        now_kst = datetime.now() + timedelta(hours=9)
        msg = f"🔔 [AI 시장 감시 - GitHub]\n시간: {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"상태: {status} (점수: {danger_score})\n"
        
        if reasons: msg += "\n".join(["- " + r for r in reasons])
        else: msg += "- 특이사항 없음"
        
        msg += f"\n\n📊 VIX지수: {current_vix:.2f} ({'상승중 ↗' if vix_trend>0 else '하락중 ↘'})"
        msg += f"\n📈 나스닥: {current_close:.2f}"
        
        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
