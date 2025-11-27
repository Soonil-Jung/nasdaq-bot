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
        print("🤖 AI 시스템(Dashboard Ver) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except: pass
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
            # 주요 지표 6종 수집
            tickers = ['NQ=F', 'QQQ', '^VIX', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX']
            data = yf.download(tickers, period='5d', interval='1h', progress=False)

            if isinstance(data.columns, pd.MultiIndex): 
                df = pd.DataFrame()
                df['Close'] = data['Close']['NQ=F']
                df['High'] = data['High']['NQ=F']
                df['Low'] = data['Low']['NQ=F']
                df['Volume'] = data['Volume']['QQQ']
                df['VIX'] = data['Close']['^VIX']
                df['DXY'] = data['Close']['DX-Y.NYB']
                df['SOXX'] = data['Close']['SOXX']
                df['HYG'] = data['Close']['HYG']
                df['TNX'] = data['Close']['^TNX']
            else: return pd.DataFrame()

            if df.empty: return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.ffill().bfill()
            return df.dropna()
        except: return pd.DataFrame()

    def analyze_danger(self):
        df = self.get_market_data()
        if df.empty: return

        # --- 1. 데이터 및 지표 계산 ---
        # A. 기술적 지표
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        current_close = df['Close'].iloc[-1]
        
        # B. 매크로 및 섹터 데이터
        current_vix = df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1]
        
        current_dxy = df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[-24]) / df['DXY'].iloc[-24] * 100 # 24시간 전 대비 변화율
        
        current_tnx = df['TNX'].iloc[-1]
        
        # 반도체 상대 강도
        nq_ret = df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 

        # 하이일드 채권 (자금 이탈)
        hyg_ma20 = df['HYG'].rolling(window=20).mean().iloc[-1]
        current_hyg = df['HYG'].iloc[-1]

        news_score = self.get_news_sentiment()

        # --- 2. 위험 점수 산출 ---
        danger_score = 0
        reasons = []

        # [A] 구름대
        cloud_status = "구름대 위 (안정)"
        if current_close < span_a:
            danger_score += 25
            reasons.append("☁️ 구름대 이탈")
            cloud_status = "구름대 하단 이탈 ☁️"
        elif current_close < span_b: # 구름대 안
            cloud_status = "구름대 진입 (혼조)"

        # [B] 거래량
        avg_vol = df['Vol_MA20'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else df['Volume'].iloc[-1] / avg_vol
        vol_status = f"평소의 {int(vol_ratio*100)}%"
        if vol_ratio > 1.5:
            danger_score += 15
            reasons.append(f"📢 거래량 급증 ({vol_ratio:.1f}배)")
            vol_status += " (폭증) 🚨"

        # [C] 달러
        dxy_status = f"{current_dxy:.2f} ({dxy_chg:+.2f}%)"
        if dxy_chg > 0.3:
            danger_score += 15
            reasons.append(f"💵 달러 급등 (+{dxy_chg:.2f}%)")
            dxy_status += " 🔺"

        # [D] 반도체
        semi_status = "양호"
        if semi_weakness > 0.005:
            danger_score += 15
            reasons.append("📉 반도체 상대적 약세")
            semi_status = "나스닥 대비 약세 ⚠️"

        # [E] 스마트머니
        hyg_status = "유입 중"
        if current_hyg < hyg_ma20:
            danger_score += 15
            reasons.append("💸 스마트머니(HYG) 이탈")
            hyg_status = "자금 이탈 감지 ⚠️"

        # [F] 공포지수
        vix_status = f"{current_vix:.2f}"
        if vix_trend > 0.5:
            danger_score += 15
            reasons.append("😱 공포지수 확산")
            vix_status += " (확산 중 ↗)"
        else:
            vix_status += " (안정 ↘)"

        # [G] 뉴스
        news_status = f"{news_score:.2f}"
        if news_score < -0.2:
            danger_score += 10
            reasons.append("📰 뉴스 심리 악화")
            news_status += " (악재 우세) ⚠️"
        elif news_score > 0.2:
            news_status += " (호재 우세) 😊"
        else:
            news_status += " (중립) 😐"

        # RSI 상태 텍스트
        rsi_status = f"{rsi_val:.1f}"
        if rsi_val < 30: rsi_status += " (과매도) 📉"
        elif rsi_val > 70: rsi_status += " (과매수) 📈"
        else: rsi_status += " (중립)"

        # 점수 보정
        danger_score = min(danger_score, 100)

        # --- 3. 메시지 작성 (Dashboard Style) ---
        status_emoji = '🔴 위험 (매도)' if danger_score >= 60 else '🟡 주의 (관망)' if danger_score >= 35 else '🟢 안정 (매수)'
        
        now_kst = datetime.now() + timedelta(hours=9)
        
        msg = f"🔔 [AI 시장 정밀 분석 리포트]\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n\n"
        msg += f"🚦 종합 진단: {status_emoji}\n"
        msg += f"🔥 위험 점수: {danger_score}점 / 100점\n\n"
        
        msg += "───────────────\n"
        msg += "1️⃣ 매크로 & 수급 (Market Health)\n"
        msg += f"💵 달러 인덱스 : {dxy_status}\n"
        msg += f"🏦 국채금리(10Y): {current_tnx:.2f}%\n"
        msg += f"💸 하이일드(HYG): {hyg_status}\n"
        msg += f"📉 반도체(SOXX) : {semi_status}\n\n"
        
        msg += "2️⃣ 기술적 분석 (Technical)\n"
        msg += f"📈 나스닥 선물 : {current_close:,.2f}\n"
        msg += f"📊 거래량 강도 : {vol_status}\n"
        msg += f"☁️ 일목균형표 : {cloud_status}\n"
        msg += f"📉 RSI (14)   : {rsi_status}\n\n"
        
        msg += "3️⃣ 심리 지표 (Sentiment)\n"
        msg += f"😱 공포지수(VIX): {vix_status}\n"
        msg += f"📰 뉴스 투심   : {news_status}\n"
        msg += "───────────────\n\n"
        
        msg += "📋 [상세 위험 요인 분석]\n"
        if reasons:
            msg += "\n".join(["- " + r for r in reasons])
        else:
            msg += "- 특이사항 없음 (모든 지표 안정적)"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
