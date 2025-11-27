import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from ta.trend import IchimokuIndicator
from ta.momentum import RSIIndicator
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from GoogleNews import GoogleNews
from datetime import datetime

# --- 환경 변수 ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(Cloud Ver) 가동... (Hourly Check)")
        # FinBERT 모델 로딩
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.keywords = ['Jerome Powell', 'Donald Trump', 'Fed Rate', 'Recession', 'Nasdaq']

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ 토큰 오류: GitHub Secrets 설정을 확인하세요. (콘솔 출력으로 대체)")
            print(message)
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"텔레그램 전송 실패: {e}")

    def get_news_sentiment(self):
        # GoogleNews 라이브러리가 종종 불안정하므로 예외처리 강화
        try:
            googlenews = GoogleNews(lang='en', period='1d')
            total_score = 0
            count = 0
            for keyword in self.keywords:
                googlenews.clear()
                googlenews.search(keyword)
                # 검색 결과가 없거나 에러날 경우를 대비
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
        except Exception as e:
            print(f"뉴스 수집 중 에러: {e}")
            return 0

    def get_market_data(self):
        # ★ 핵심 개선: interval='1h' (1시간 봉) 사용
        # 기간은 최근 1달(1mo)이면 지표 계산에 충분함
        print("데이터 수집 중 (1시간 봉 기준)...")
        
        # 1. 나스닥 선물 (가격 분석용)
        df = yf.download('NQ=F', period='1mo', interval='1h', progress=False)
        
        # 2. QQQ (거래량 분석용 - 선물의 거래량 데이터 오류 방지)
        df_vol = yf.download('QQQ', period='1mo', interval='1h', progress=False)
        
        # 3. 금리
        tnx = yf.download('^TNX', period='1mo', interval='1h', progress=False)

        # MultiIndex 컬럼 처리
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
        if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = tnx.columns.get_level_values(0)

        # 데이터 병합
        # 시간축이 다를 수 있으므로 인덱스 기준으로 맞춤 (마지막 행이 중요)
        df = df[['High', 'Low', 'Close']].copy()
        
        # 거래량은 QQQ 데이터를 사용 (신뢰도 향상)
        # 인덱스(시간)를 맞춰서 가져오기 위해 reindex 사용
        df['Volume'] = df_vol['Volume'].reindex(df.index).fillna(0)
        df['US_10Y'] = tnx['Close'].reindex(df.index).fillna(method='ffill')
        
        return df.dropna()

    def analyze_danger(self):
        try:
            df = self.get_market_data()
            
            if df.empty:
                print("❌ 데이터 수집 실패 (Empty DataFrame)")
                return

            # --- 지표 계산 (1시간 봉 기준) ---
            
            # 1. 거래량 이평선 (최근 20시간 평균)
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # 최신 데이터 (마지막 캔들)
            current_close = df['Close'].iloc[-1]
            current_vol = df['Volume'].iloc[-1]
            avg_vol = df['Vol_MA20'].iloc[-1]
            
            # 2. 일목균형표 (9, 26, 52 시간)
            ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
            span_a = ichimoku.ichimoku_a().iloc[-1]
            
            # 3. RSI (14시간)
            rsi = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
            
            # 4. 뉴스
            news_score = self.get_news_sentiment()
            
            # --- 위험 점수 계산 ---
            danger_score = 0
            reasons = []
            
            # [1] 구름대 이탈 (1시간 봉 기준 추세 이탈은 단기 위험 신호)
            if current_close < span_a: 
                danger_score += 30
                reasons.append("☁️ 구름대 하단 이탈 (단기 추세 하락)")
            
            # [2] 거래량 폭증
            # 0으로 나누기 방지
            vol_ratio = 0
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
            
            if vol_ratio > 1.5:
                danger_score += 20
                reasons.append(f"📢 거래량 급증 (직전평균 대비 {vol_ratio:.1f}배)")
            
            # [3] 뉴스 심리
            if news_score < -0.2: 
                danger_score += 25
                reasons.append(f"📰 뉴스 심리 악화 ({news_score:.2f})")
            
            # [4] RSI 과매도
            if rsi < 30: # 1시간 봉에서는 30 이하가 더 확실한 과매도
                danger_score += 15
                reasons.append(f"📉 RSI 과매도 ({rsi:.1f})")
            
            # [5] 복합 위험 (거래량 실린 하락)
            if (current_close < span_a) and (vol_ratio > 1.5):
                danger_score += 10
                reasons.append("💥 [위험] 대량 거래 동반 하락")

            # --- 결과 전송 ---
            status = '🔴 위험 (현금화)' if danger_score >= 70 else '🟡 주의 (관망)' if danger_score >= 40 else '🟢 안정 (매수)'
            
            msg = f"🔔 [시장 알림 - 1시간봉 기준]\n상태: {status} (점수: {danger_score})\n"
            if reasons: 
                msg += "\n".join(["- " + r for r in reasons])
            else: 
                msg += "- 특이사항 없음"
            
            msg += f"\n\n📊 거래량(QQQ): 평소 대비 {int(vol_ratio*100)}%"
            msg += f"\n📈 현재가(NQ): {current_close:.2f}"
            
            self.send_telegram(msg)
            print("✅ 분석 완료")

        except Exception as e:
            print(f"Main logic Error: {e}")
            self.send_telegram(f"❌ 봇 에러: {str(e)}")

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
