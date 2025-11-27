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
# [보안 수정] GitHub Secrets에서 토큰을 가져옵니다.
# (직접 입력하지 마세요!)
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(GitHub Action) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            print("✅ AI 모델 로딩 완료")
        except Exception as e:
            print(f"⚠️ AI 모델 로딩 실패: {e}")
            
        self.keywords = ['Jerome Powell', 'Donald Trump', 'Fed Rate', 'Recession', 'Nasdaq']

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ 오류: GitHub Secrets에 토큰이 설정되지 않았습니다.")
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
            print("✅ 텔레그램 전송 성공")
        except Exception as e:
            print(f"전송 실패: {e}")

    def get_news_sentiment(self):
        print("📰 뉴스 데이터 분석 중...")
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
        except:
            return 0

    def get_market_data(self):
        print("📈 시장 데이터 수집 중...")
        try:
            df = yf.download('NQ=F', period='5d', interval='1h', progress=False)
            df_vol = yf.download('QQQ', period='5d', interval='1h', progress=False)
            tnx = yf.download('^TNX', period='5d', interval='1h', progress=False)

            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
            if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = tnx.columns.get_level_values(0)

            if df.empty: return pd.DataFrame()

            df = df[['High', 'Low', 'Close']].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df_vol.index = pd.to_datetime(df_vol.index).tz_localize(None)
            tnx.index = pd.to_datetime(tnx.index).tz_localize(None)

            df['Volume'] = df_vol['Volume'].reindex(df.index).fillna(0)
            tnx_series = tnx['Close'].reindex(df.index)
            df['US_10Y'] = tnx_series.ffill().bfill().fillna(4.0)
            
            return df.dropna(subset=['Close'])
        except:
            return pd.DataFrame()

    def analyze_danger(self):
        df = self.get_market_data()
        if df.empty: 
            print("❌ 데이터 없음")
            return

        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        
        current_close = df['Close'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Vol_MA20'].iloc[-1]
        
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        rsi = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        news_score = self.get_news_sentiment()
        
        danger_score = 0
        reasons = []
        
        if current_close < span_a: 
            danger_score += 30
            reasons.append("☁️ 구름대 이탈")
        
        vol_ratio = 0
        if avg_vol > 0: vol_ratio = current_vol / avg_vol
        
        if vol_ratio > 1.5:
            danger_score += 20
            reasons.append(f"📢 거래량 급증 ({vol_ratio:.1f}배)")
        
        if news_score < -0.2: 
            danger_score += 25
            reasons.append(f"📰 뉴스 악재 ({news_score:.2f})")
        
        if rsi < 30: 
            danger_score += 15
            reasons.append(f"📉 RSI 과매도 ({rsi:.1f})")
            
        if (current_close < span_a) and (vol_ratio > 1.5):
            danger_score += 10
            reasons.append("💥 대량 거래 하락")

        status = '🔴 위험 (매도 고려)' if danger_score >= 70 else '🟡 주의 (관망)' if danger_score >= 40 else '🟢 안정 (매수 유효)'
        
        # 한국 시간 계산
        now_utc = datetime.now()
        now_kst = now_utc + timedelta(hours=9)
        now_time = now_kst.strftime("%Y-%m-%d %H:%M")
        
        msg = f"🔔 [AI 시장 감시 - GitHub]\n시간: {now_time} (KST)\n"
        msg += f"상태: {status} (점수: {danger_score})\n"
        
        if reasons: msg += "\n".join(["- " + r for r in reasons])
        else: msg += "- 특이사항 없음"
        
        msg += f"\n\n📊 거래량(QQQ): {int(vol_ratio*100)}%"
        msg += f"\n📈 현재가(NQ): {current_close:.2f}"
        
        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
