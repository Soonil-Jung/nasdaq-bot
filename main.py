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
        print("🤖 AI 시스템(Full-Report Ver) 가동 중...")
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
            tickers = ['NQ=F', 'QQQ', '^VIX', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', 'BTC-USD', '^IRX']
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
                df['IRX'] = data['Close']['^IRX']
                df['BTC'] = data['Close']['BTC-USD']
            else: return pd.DataFrame()

            if df.empty: return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.ffill().bfill()
            return df.dropna()
        except: return pd.DataFrame()

    def analyze_danger(self):
        df = self.get_market_data()
        if df.empty: return

        # --- 1. 지표 계산 ---
        # 기술적 지표
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        current_close = df['Close'].iloc[-1]
        
        # 매크로
        current_vix = df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1]
        
        current_dxy = df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[-24]) / df['DXY'].iloc[-24] * 100 
        
        current_tnx = df['TNX'].iloc[-1]
        current_irx = df['IRX'].iloc[-1]
        yield_spread = current_tnx - current_irx
        irx_chg = (current_irx - df['IRX'].iloc[-24]) / df['IRX'].iloc[-24] * 100
        
        # 자산 데이터
        current_btc = df['BTC'].iloc[-1]
        btc_chg = (current_btc - df['BTC'].iloc[-24]) / df['BTC'].iloc[-24] * 100
        
        nq_ret = df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 

        hyg_high = df['HYG'].max()
        current_hyg = df['HYG'].iloc[-1]
        hyg_drawdown = (current_hyg - hyg_high) / hyg_high * 100

        news_score = self.get_news_sentiment()

        # --- 2. 위험 점수 산출 ---
        danger_score = 0
        reasons = []

        # [A] 구름대
        cloud_str = "구름대 위 (안정)"
        if current_close < span_a:
            danger_score += 25
            reasons.append("☁️ 구름대 이탈")
            cloud_str = "하단 이탈 🚨"
        elif current_close < span_b: 
            cloud_str = "구름대 내부 (혼조)"

        # [B] 거래량
        avg_vol = df['Vol_MA20'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else df['Volume'].iloc[-1] / avg_vol
        if vol_ratio > 1.5:
            danger_score += 15
            reasons.append(f"📢 거래량 급증 ({vol_ratio:.1f}배)")

        # [C] 달러
        if dxy_chg > 0.3:
            danger_score += 15
            reasons.append(f"💵 달러 급등 (+{dxy_chg:.2f}%)")

        # [D] 금리
        if irx_chg > 2.0:
            danger_score += 15
            reasons.append(f"🏦 단기금리 급등 (+{irx_chg:.1f}%)")
        
        # [E] 비트코인
        if btc_chg < -3.0: 
            danger_score += 15
            reasons.append(f"📉 비트코인 급락 ({btc_chg:.2f}%)")

        # [F] 반도체
        if semi_weakness > 0.005:
            danger_score += 10
            reasons.append(f"📉 반도체 약세 (괴리: {semi_weakness*100:.1f}%)")

        # [G] 하이일드
        if hyg_drawdown < -0.3:
            danger_score += 15
            reasons.append(f"💸 스마트머니 이탈 ({hyg_drawdown:.2f}%)")

        # [H] 공포지수
        if vix_trend > 0.5:
            danger_score += 10
            reasons.append(f"😱 공포지수 확산 (+{vix_trend:.1f})")

        # [I] 뉴스
        if news_score < -0.2:
            danger_score += 10
            reasons.append(f"📰 뉴스 심리 악화 ({news_score:.2f})")

        danger_score = min(danger_score, 100)

        # --- 3. 메시지 작성 (Full Report Style) ---
        status_emoji = '🔴 위험 (매도)' if danger_score >= 60 else '🟡 주의 (관망)' if danger_score >= 35 else '🟢 안정 (매수)'
        
        now_kst = datetime.now() + timedelta(hours=9)
        
        msg = f"🔔 [AI 퀀트 전체 분석 리포트]\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 종합상태: {status_emoji}\n"
        msg += f"🔥 위험점수: {danger_score}점 / 100점\n\n"
        
        msg += "1️⃣ 매크로 지표 (Economy)\n"
        msg += f"💵 달러(DXY): {current_dxy:.2f} ({dxy_chg:+.2f}%)\n"
        msg += f"🏦 금리(10Y): {current_tnx:.2f}%\n"
        msg += f"🏦 금리(3M): {current_irx:.2f}% ({irx_chg:+.1f}%)\n"
        msg += f"📉 장단기차 : {yield_spread:.2f}p ({'역전⚠️' if yield_spread<0 else '정상'})\n\n"
        
        msg += "2️⃣ 리스크 자산 흐름 (Flow)\n"
        msg += f"₿ 비트코인 : ${current_btc:,.0f} ({btc_chg:+.2f}%)\n"
        msg += f"📉 반도체괴리: {semi_weakness*100:+.2f}% ({'약세⚠️' if semi_weakness>0.005 else '양호'})\n"
        msg += f"💸 하이일드낙폭: {hyg_drawdown:.2f}% ({'이탈⚠️' if hyg_drawdown<-0.3 else '유입'})\n\n"
        
        msg += "3️⃣ 기술적 분석 (Technical)\n"
        msg += f"📈 나스닥선물: {current_close:,.2f}\n"
        msg += f"☁️ 일목구름 : {cloud_str}\n"
        msg += f"📊 거래량강도: 평소의 {int(vol_ratio*100)}%\n"
        msg += f"📉 RSI (14)  : {rsi_val:.1f}\n\n"
        
        msg += "4️⃣ 시장 심리 (Sentiment)\n"
        msg += f"😱 공포(VIX) : {current_vix:.2f} (추세: {vix_trend:+.1f})\n"
        msg += f"📰 뉴스점수 : {news_score:.2f} (-1~+1)\n"
        msg += "───────────────\n"
        
        msg += "📋 [위험 점수 증가 사유]\n"
        if reasons:
            msg += "\n".join(["🚨 " + r for r in reasons])
        else:
            msg += "✅ 특이사항 없음 (모든 지표 안정적)"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
