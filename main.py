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

# ======================================================
# ▼▼▼ 여기에 정보를 직접 입력하세요 (따옴표 필수!) ▼▼▼
TELEGRAM_TOKEN = "7961108822:AAG1gMSmtDuJ5F7P29szagNri6OvDzZeQGg" 
TELEGRAM_CHAT_ID = "6376538116"
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템 가동 중... (잠시만 기다려주세요)")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            print("✅ AI 모델 로딩 완료")
        except Exception as e:
            print(f"⚠️ AI 모델 로딩 실패 (인터넷 연결 확인): {e}")
            
        self.keywords = ['Jerome Powell', 'Donald Trump', 'Fed Rate', 'Recession', 'Nasdaq']

    def send_telegram(self, message):
        if "여기에" in TELEGRAM_TOKEN or "여기에" in TELEGRAM_CHAT_ID:
            print("\n[!!!!] 경고: 토큰과 ID를 입력하지 않으셨습니다!")
            print(f"--- 전송 예정이었던 메시지 ---\n{message}\n-----------------------------")
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("✅ 텔레그램 메시지 전송 성공!")
            else:
                print(f"❌ 전송 실패 (에러코드 {response.status_code}): {response.text}")
        except Exception as e:
            print(f"텔레그램 접속 실패: {e}")

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
            print("⚠️ 뉴스 수집 건너뜀 (일시적 오류)")
            return 0

    def get_market_data(self):
        print("📈 시장 데이터 수집 중 (1시간 봉 기준)...")
        try:
            # period='5d' (5일치), interval='1h' (1시간봉)
            df = yf.download('NQ=F', period='5d', interval='1h', progress=False)
            df_vol = yf.download('QQQ', period='5d', interval='1h', progress=False)
            tnx = yf.download('^TNX', period='5d', interval='1h', progress=False)

            # MultiIndex 컬럼 평탄화
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
            if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = tnx.columns.get_level_values(0)

            if df.empty:
                print("❌ 오류: NQ=F 데이터가 없습니다.")
                return pd.DataFrame()

            # 필요한 컬럼만 추출
            df = df[['High', 'Low', 'Close']].copy()
            
            # 시간대 통일 (UTC 제거)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df_vol.index = pd.to_datetime(df_vol.index).tz_localize(None)
            tnx.index = pd.to_datetime(tnx.index).tz_localize(None)

            # 데이터 병합 (중요: 결측치 방어 로직 추가)
            # 1. 거래량: 없으면 0으로 채움
            df['Volume'] = df_vol['Volume'].reindex(df.index).fillna(0)
            
            # 2. 금리: 앞뒤 값으로 채움 (ffill + bfill)
            # 금리 데이터가 아예 없으면 4.0(기본값)으로 채워 에러 방지
            tnx_series = tnx['Close'].reindex(df.index)
            df['US_10Y'] = tnx_series.ffill().bfill().fillna(4.0)
            
            # 3. 그래도 비어있는 '가격(Close)' 데이터만 삭제 (금리 때문에 삭제되는 일 방지)
            final_df = df.dropna(subset=['Close'])
            
            print(f"✅ 데이터 준비 완료 ({len(final_df)}개 캔들)")
            return final_df

        except Exception as e:
            print(f"❌ 데이터 다운로드 중 치명적 오류: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def analyze_danger(self):
        df = self.get_market_data()
        if df.empty: 
            print("❌ 분석할 데이터가 없어 종료합니다.")
            return

        print("🧮 위험도 계산 중...")
        # 지표 계산
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        
        current_close = df['Close'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Vol_MA20'].iloc[-1]
        
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        rsi = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        news_score = self.get_news_sentiment()
        
        # 위험 점수 계산
        danger_score = 0
        reasons = []
        
        # [1] 구름대 하단 이탈
        if current_close < span_a: 
            danger_score += 30
            reasons.append("☁️ 구름대 하단 이탈 (하락추세)")
        
        # [2] 거래량 폭증
        vol_ratio = 0
        if avg_vol > 0: vol_ratio = current_vol / avg_vol
        
        if vol_ratio > 1.5:
            danger_score += 20
            reasons.append(f"📢 거래량 급증 ({vol_ratio:.1f}배)")
        
        # [3] 뉴스 악재
        if news_score < -0.2: 
            danger_score += 25
            reasons.append(f"📰 뉴스 악재 발생 ({news_score:.2f})")
        
        # [4] RSI 과매도
        if rsi < 30: 
            danger_score += 15
            reasons.append(f"📉 RSI 과매도 ({rsi:.1f})")
            
        # [5] 복합 위험 (거래량 실린 하락)
        if (current_close < span_a) and (vol_ratio > 1.5):
            danger_score += 10
            reasons.append("💥 [위험] 대량 거래 동반 하락")

        # 메시지 작성
        status = '🔴 위험 (매도 고려)' if danger_score >= 70 else '🟡 주의 (관망)' if danger_score >= 40 else '🟢 안정 (매수 유효)'
        
        msg = f"🔔 [AI 시장 감시 - 실시간]\n상태: {status} (점수: {danger_score})\n"
        if reasons: msg += "\n".join(["- " + r for r in reasons])
        else: msg += "- 특이사항 없음"
        
        msg += f"\n\n📊 거래량(QQQ): 평소 대비 {int(vol_ratio*100)}%"
        msg += f"\n📈 현재가(NQ): {current_close:.2f}"
        
        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
