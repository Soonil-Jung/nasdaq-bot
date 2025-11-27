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
# ▼▼▼ 사용자 설정 정보 (GitHub Secrets 사용 시 os.environ 유지) ▼▼▼
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(Full-Variables Ver) 가동 중...")
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

        # --- 1. 모든 지표 계산 (Variables Calculation) ---
        
        # [A] 가격 및 변동성 (Price Action)
        current_close = df['Close'].iloc[-1]
        daily_chg = (current_close - df['Close'].iloc[-24]) / df['Close'].iloc[-24] * 100 # 24시간 등락
        hourly_chg = (current_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100  # 1시간 등락
        
        # [B] 거래량 (Volume)
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        avg_vol = df['Vol_MA20'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else current_vol / avg_vol
        
        # [C] 기술적 지표 (Ichimoku & RSI)
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        # [D] 매크로 (Macro)
        current_dxy = df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[-24]) / df['DXY'].iloc[-24] * 100
        
        current_tnx = df['TNX'].iloc[-1] # 10년물
        current_irx = df['IRX'].iloc[-1] # 3개월물 (단기)
        yield_spread = current_tnx - current_irx # 장단기 금리차
        irx_chg = (current_irx - df['IRX'].iloc[-24]) / df['IRX'].iloc[-24] * 100 # 단기금리 변동
        
        # [E] 리스크 자산 (Risk Assets)
        current_btc = df['BTC'].iloc[-1]
        btc_chg = (current_btc - df['BTC'].iloc[-24]) / df['BTC'].iloc[-24] * 100
        
        # 반도체 괴리율 (나스닥 수익률 - 반도체 수익률)
        nq_ret = df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 
        
        # 하이일드 고점 대비 하락률
        hyg_high = df['HYG'].max()
        current_hyg = df['HYG'].iloc[-1]
        hyg_drawdown = (current_hyg - hyg_high) / hyg_high * 100
        
        # [F] 심리 (Sentiment)
        current_vix = df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1] # 추세
        news_score = self.get_news_sentiment()

        # --- 2. 위험 점수 산출 (Scoring) ---
        danger_score = 0
        reasons = []

        # 1. 가격 추세
        if current_close < span_a:
            danger_score += 20
            reasons.append("☁️ 구름대 하단 이탈")
        if daily_chg < -1.5:
            danger_score += 20
            reasons.append(f"📉 일간 추세 하락 ({daily_chg:.2f}%)")
        if hourly_chg < -0.8:
            danger_score += 15
            reasons.append(f"⚡ 1시간 급락 ({hourly_chg:.2f}%)")

        # 2. 거래량
        if vol_ratio > 1.5:
            danger_score += 15
            reasons.append(f"📢 거래량 폭증 ({vol_ratio:.1f}배)")

        # 3. 매크로
        if dxy_chg > 0.3:
            danger_score += 10
            reasons.append(f"💵 달러 강세 (+{dxy_chg:.2f}%)")
        if irx_chg > 2.0:
            danger_score += 10
            reasons.append(f"🏦 단기금리 급등 (+{irx_chg:.1f}%)")
        if yield_spread < -0.8: # 역전 심화 시 점수 반영은 선택사항(여기선 알림만)
            pass 

        # 4. 리스크 자산
        if btc_chg < -3.0: 
            danger_score += 15
            reasons.append(f"📉 비트코인 급락 ({btc_chg:.2f}%)")
        if semi_weakness > 0.005:
            danger_score += 10
            reasons.append(f"📉 반도체 상대적 약세")
        if hyg_drawdown < -0.3:
            danger_score += 15
            reasons.append(f"💸 스마트머니 이탈 ({hyg_drawdown:.2f}%)")

        # 5. 심리
        if vix_trend > 0.5:
            danger_score += 10
            reasons.append(f"😱 공포지수 확산")
        if news_score < -0.2:
            danger_score += 10
            reasons.append(f"📰 뉴스 심리 악화")

        # 점수 Cap
        danger_score = min(danger_score, 100)

        # --- 3. 메시지 작성 (Full Report) ---
        
        # 상태 문자열 정의
        status_emoji = '🔴 위험 (매도)' if danger_score >= 60 else '🟡 주의 (관망)' if danger_score >= 35 else '🟢 안정 (매수)'
        cloud_str = "하단 이탈 🚨" if current_close < span_a else ("구름대 안 ☁️" if current_close < span_b else "구름대 위 ✅")
        spread_str = "정상 ✅" if yield_spread >= 0 else "역전(침체) ⚠️"
        semi_str = "약세 ⚠️" if semi_weakness > 0.005 else "양호 ✅"
        hyg_str = "이탈 ⚠️" if hyg_drawdown < -0.3 else "유입 ✅"
        vix_str = "확산 ↗" if vix_trend > 0 else "안정 ↘"
        
        now_kst = datetime.now() + timedelta(hours=9)
        
        msg = f"🔔 [AI 퀀트 전체 변수 리포트]\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 상태: {status_emoji}\n"
        msg += f"🔥 점수: {danger_score}점 / 100점\n\n"
        
        msg += "1️⃣ 가격 & 거래량 (Technical)\n"
        msg += f"• 나스닥 : {current_close:,.2f} (24h: {daily_chg:+.2f}%)\n"
        msg += f"• 1시간봉 : {hourly_chg:+.2f}% (단기변동)\n"
        msg += f"• 거래강도 : 평소의 {int(vol_ratio*100)}%\n"
        msg += f"• RSI(14) : {rsi_val:.1f}\n"
        msg += f"• 일목구름 : {cloud_str}\n\n"
        
        msg += "2️⃣ 매크로 지표 (Macro)\n"
        msg += f"• 달러(DXY): {current_dxy:.2f} ({dxy_chg:+.2f}%)\n"
        msg += f"• 3개월금리 : {current_irx:.2f}% (Fed기대)\n"
        msg += f"• 10년금리 : {current_tnx:.2f}% (시장금리)\n"
        msg += f"• 장단기차 : {yield_spread:.2f}p ({spread_str})\n\n"
        
        msg += "3️⃣ 리스크 자산 (Risk Asset)\n"
        msg += f"• 비트코인 : ${current_btc:,.0f} ({btc_chg:+.2f}%)\n"
        msg += f"• 반도체비 : {semi_str} (괴리: {semi_weakness*100:.1f}%)\n"
        msg += f"• 하이일드 : {hyg_str} (낙폭: {hyg_drawdown:.2f}%)\n\n"
        
        msg += "4️⃣ 시장 심리 (Sentiment)\n"
        msg += f"• 공포(VIX): {current_vix:.2f} (추세: {vix_str})\n"
        msg += f"• 뉴스점수 : {news_score:.2f} (-1~+1)\n"
        msg += "───────────────\n"
        
        msg += "📋 [위험 점수 반영 내역]\n"
        if reasons:
            msg += "\n".join(["🚨 " + r for r in reasons])
        else:
            msg += "✅ 특이사항 없음 (안정적)"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
