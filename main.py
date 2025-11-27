import os
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from ta.trend import IchimokuIndicator
from ta.momentum import RSIIndicator
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from GoogleNews import GoogleNews
from datetime import datetime

# --- 환경 변수에서 비밀번호 가져오기 (보안) ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(Cloud Ver) 가동...")
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.keywords = ['Jerome Powell', 'Donald Trump', 'Fed Rate', 'Recession', 'Nasdaq']

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ 토큰이 없습니다. GitHub Secrets 설정을 확인하세요.")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=data)

    def get_news_sentiment(self):
        googlenews = GoogleNews(lang='en', period='1d')
        total_score = 0
        count = 0
        for keyword in self.keywords:
            googlenews.clear()
            googlenews.search(keyword)
            for item in googlenews.results(sort=True)[:2]:
                try:
                    res = self.nlp(item['title'][:512])[0]
                    score = res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0
                    total_score += score
                    count += 1
                except: continue
        return total_score / count if count > 0 else 0

    def get_market_data(self):
        df = yf.download('NQ=F', period='6mo', progress=False)
        tnx = yf.download('^TNX', period='6mo', progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = tnx.columns.get_level_values(0)
        df = df[['High', 'Low', 'Close']].copy()
        df['US_10Y'] = tnx['Close']
        return df.fillna(method='ffill')

    def analyze_danger(self):
        try:
            df = self.get_market_data()
            current_close = df['Close'].iloc[-1]
            
            # 지표 계산
            ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
            span_a = ichimoku.ichimoku_a().iloc[-1]
            rsi = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
            
            news_score = self.get_news_sentiment()
            
            danger_score = 0
            reasons = []
            
            if current_close < span_a: danger_score += 40; reasons.append("☁️ 구름대 하단 이탈")
            if news_score < -0.2: danger_score += 30; reasons.append(f"📰 뉴스 심리 악화({news_score:.2f})")
            if rsi < 35: danger_score += 15; reasons.append(f"📉 RSI 과매도({rsi:.1f})")
            
            status = '🔴 위험' if danger_score >= 70 else '🟡 주의' if danger_score >= 40 else '🟢 안정'
            
            msg = f"🔔 [시장 알림]\n상태: {status} (점수: {danger_score})\n"
            if reasons: msg += "\n".join(["- " + r for r in reasons])
            else: msg += "- 특이사항 없음"
            
            self.send_telegram(msg)
            print("✅ 분석 완료 및 전송 성공")
            
        except Exception as e:
            self.send_telegram(f"❌ 봇 에러 발생: {str(e)}")

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
