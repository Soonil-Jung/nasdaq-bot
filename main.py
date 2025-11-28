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

# 분석할 종목 리스트 (티커)와 뉴스 검색용 키워드 매핑
TARGET_STOCKS = {
    'AAPL': 'Apple stock',
    'MSFT': 'Microsoft stock',
    'GOOGL': 'Google Alphabet stock',
    'TSLA': 'Tesla stock Elon Musk',
    'NVDA': 'Nvidia stock',
    'AMD': 'AMD stock',
    'PLTR': 'Palantir stock'
}
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(Ultimate Ver) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except: pass
        
        # 매크로 뉴스 키워드
        self.macro_keywords = ['Jerome Powell', 'Fed Rate', 'Recession', 'US Economy', 'Nasdaq']

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
        try: requests.post(url, data=data)
        except: pass

    def get_news_sentiment(self, target_keywords):
        """
        뉴스 감성 분석 함수 (매크로 공통 / 개별 종목 공용)
        target_keywords: 검색할 키워드 리스트 또는 문자열
        """
        try:
            googlenews = GoogleNews(lang='en', period='1d')
            total_score = 0
            count = 0
            worst_title = ""
            worst_link = ""
            min_score = 1.0 

            # 입력이 단일 문자열이면 리스트로 변환
            search_list = [target_keywords] if isinstance(target_keywords, str) else target_keywords

            for key in search_list:
                googlenews.clear()
                googlenews.search(key)
                results = googlenews.results(sort=True)
                if not results: continue
                
                # 상위 2개 뉴스만 샘플링 (속도 최적화)
                for item in results[:2]:
                    try:
                        title = item['title']
                        link = item['link']
                        res = self.nlp(title[:512])[0]
                        score = res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0
                        
                        total_score += score
                        count += 1
                        
                        if score < min_score and score < -0.5:
                            min_score = score
                            worst_title = title
                            worst_link = link
                    except: continue
            
            avg_score = total_score / count if count > 0 else 0
            return avg_score, worst_title, worst_link
            
        except: return 0, "", ""

    def get_market_data(self):
        try:
            macro_tickers = ['NQ=F', 'QQQ', '^VIX', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', 'BTC-USD', '^IRX']
            all_tickers = macro_tickers + list(TARGET_STOCKS.keys())
            
            data = yf.download(all_tickers, period='5d', interval='1h', progress=False)

            if isinstance(data.columns, pd.MultiIndex): 
                dfs = {}
                # 1. 매크로 데이터
                df_macro = pd.DataFrame()
                df_macro['Close'] = data['Close']['NQ=F']
                df_macro['High'] = data['High']['NQ=F']
                df_macro['Low'] = data['Low']['NQ=F']
                df_macro['Volume'] = data['Volume']['QQQ']
                df_macro['VIX'] = data['Close']['^VIX']
                df_macro['DXY'] = data['Close']['DX-Y.NYB']
                df_macro['SOXX'] = data['Close']['SOXX']
                df_macro['HYG'] = data['Close']['HYG']
                df_macro['TNX'] = data['Close']['^TNX']
                df_macro['IRX'] = data['Close']['^IRX']
                df_macro['BTC'] = data['Close']['BTC-USD']
                
                df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None)
                df_macro = df_macro.ffill().bfill().dropna()
                dfs['MACRO'] = df_macro

                # 2. 개별 종목 데이터
                for ticker in TARGET_STOCKS.keys():
                    try:
                        df_stock = pd.DataFrame()
                        df_stock['Close'] = data['Close'][ticker]
                        df_stock['High'] = data['High'][ticker]
                        df_stock['Low'] = data['Low'][ticker]
                        df_stock['Volume'] = data['Volume'][ticker]
                        
                        df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
                        df_stock = df_stock.dropna()
                        dfs[ticker] = df_stock
                    except: continue
                return dfs
            else: return {}
        except: return {}

    def analyze_individual(self, ticker, df_stock, df_macro):
        """개별 종목 분석 (기술적 + 뉴스)"""
        if df_stock.empty: return None

        # [1] 기술적 지표 계산 (차트 기반)
        current_close = df_stock['Close'].iloc[-1]
        
        # 등락률
        daily_chg = (current_close - df_stock['Close'].iloc[-8]) / df_stock['Close'].iloc[-8] * 100 
        if len(df_stock) > 24:
             daily_chg = (current_close - df_stock['Close'].iloc[-7]) / df_stock['Close'].iloc[-7] * 100

        # 상대 강도 (vs NQ)
        qqq_chg = 0
        try:
            qqq_now = df_macro['Close'].iloc[-1]
            qqq_prev = df_macro['Close'].iloc[-7]
            qqq_chg = (qqq_now - qqq_prev) / qqq_prev * 100
        except: pass
        relative_strength = daily_chg - qqq_chg

        # Ichimoku & RSI
        ichimoku = IchimokuIndicator(high=df_stock['High'], low=df_stock['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        rsi_val = RSIIndicator(close=df_stock['Close'], window=14).rsi().iloc[-1]
        
        # 거래량
        df_stock['Vol_MA20'] = df_stock['Volume'].rolling(window=20).mean()
        vol_ratio = 0
        if df_stock['Vol_MA20'].iloc[-1] > 0:
            vol_ratio = df_stock['Volume'].iloc[-1] / df_stock['Vol_MA20'].iloc[-1]

        # [2] 뉴스 분석 (해당 종목 전용)
        # 예: TSLA면 "Tesla stock Elon Musk"로 검색
        search_keyword = TARGET_STOCKS.get(ticker, ticker)
        news_score, worst_news, _ = self.get_news_sentiment(search_keyword)

        # [3] 위험 점수 산출
        danger_score = 0
        reasons = []

        # 변동성 큰 종목(Beta) 필터
        high_beta = ['TSLA', 'NVDA', 'AMD', 'PLTR']
        drop_threshold = -3.5 if ticker in high_beta else -2.0

        if daily_chg < drop_threshold:
            danger_score += 30
            reasons.append(f"📉 폭락 ({daily_chg:.1f}%)")
        
        if relative_strength < -1.5: 
            danger_score += 15
            reasons.append(f"약세 (시장대비 {relative_strength:.1f}%)")
            
        if current_close < span_a:
            danger_score += 20
            reasons.append("☁️ 구름대 이탈")
            
        if rsi_val < 30:
            danger_score += 10
            reasons.append(f"과매도({rsi_val:.0f})")
            
        if vol_ratio > 2.0:
            danger_score += 15
            reasons.append(f"거래량폭발({vol_ratio:.1f}x)")

        # ★ 개별 뉴스 악재 반영
        if news_score < -0.3:
            danger_score += 20
            short_news = worst_news[:15] + "..." if len(worst_news) > 15 else worst_news
            reasons.append(f"📰 악재 뉴스 ({short_news})")

        return {
            "ticker": ticker,
            "score": min(danger_score, 100),
            "change": daily_chg,
            "reasons": reasons
        }

    def analyze_danger(self):
        dfs = self.get_market_data()
        if not dfs or 'MACRO' not in dfs: return
        
        df = dfs['MACRO']

        # --- [PART 1] 전체 시장 분석 (Full Variables) ---
        
        # 기술적 지표
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        # 실시간 호가 조회
        try:
            ticker_nq = yf.Ticker("NQ=F")
            price = ticker_nq.fast_info.get('last_price')
            current_close = price if (price and not np.isnan(price)) else df['Close'].iloc[-1]
        except: current_close = df['Close'].iloc[-1]

        daily_chg = (current_close - df['Close'].iloc[-24]) / df['Close'].iloc[-24] * 100 
        hourly_chg = (current_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
        
        avg_vol = df['Vol_MA20'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else current_vol / avg_vol
        
        # 매크로
        current_dxy = df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[-24]) / df['DXY'].iloc[-24] * 100
        current_tnx = df['TNX'].iloc[-1]
        current_irx = df['IRX'].iloc[-1]
        yield_spread = current_tnx - current_irx
        
        # 리스크 자산
        current_btc = df['BTC'].iloc[-1]
        btc_chg = (current_btc - df['BTC'].iloc[-24]) / df['BTC'].iloc[-24] * 100
        
        nq_ret = current_close / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 
        
        hyg_high = df['HYG'].max()
        current_hyg = df['HYG'].iloc[-1]
        hyg_drawdown = (current_hyg - hyg_high) / hyg_high * 100

        # 매크로 뉴스
        news_score, worst_title, worst_link = self.get_news_sentiment(self.macro_keywords)
        current_vix = df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1]

        # 점수 산출
        danger_score = 0
        reasons = []
        if daily_chg < -1.5: danger_score += 20; reasons.append(f"📉 추세 하락")
        if hourly_chg < -0.8: danger_score += 15; reasons.append(f"⚡ 투매 발생")
        if current_close < span_a: danger_score += 20; reasons.append("☁️ 구름대 이탈")
        if vol_ratio > 1.5: danger_score += 15; reasons.append(f"📢 거래량 폭증")
        if dxy_chg > 0.3: danger_score += 10; reasons.append(f"💵 달러 강세")
        if btc_chg < -3.0: danger_score += 15; reasons.append(f"📉 비트코인 급락")
        if semi_weakness > 0.005: danger_score += 10; reasons.append(f"📉 반도체 약세")
        if hyg_drawdown < -0.3: danger_score += 15; reasons.append(f"💸 스마트머니 이탈")
        if news_score < -0.2: danger_score += 10; reasons.append(f"📰 뉴스 심리 악화")
        danger_score = min(danger_score, 100)

        # --- [PART 2] 개별 종목 분석 ---
        stock_results = []
        for ticker in TARGET_STOCKS.keys():
            if ticker in dfs:
                res = self.analyze_individual(ticker, dfs[ticker], df)
                if res: stock_results.append(res)
        stock_results.sort(key=lambda x: x['score'], reverse=True)

        # --- [PART 3] 메시지 작성 (Full Report) ---
        status_emoji = '🔴 위험' if danger_score >= 60 else '🟡 주의' if danger_score >= 35 else '🟢 안정'
        cloud_str = "하단 이탈 🚨" if current_close < span_a else "구름대 위 ✅"
        
        now_kst = datetime.now() + timedelta(hours=9)
        
        msg = f"🔔 *AI 퀀트 & 종목 정밀 리포트*\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 종합상태: {status_emoji} ({danger_score}점)\n\n"
        
        msg += "*1️⃣ 매크로 & 테크니컬 (Market)*\n"
        msg += f"• 나스닥 : {current_close:,.2f} ({daily_chg:+.2f}%)\n"
        msg += f"• 1시간봉 : {hourly_chg:+.2f}% / 거래 {int(vol_ratio*100)}%\n"
        msg += f"• 달러/금리 : DXY {current_dxy:.2f} / 10Y {current_tnx:.2f}%\n"
        msg += f"• 장단기차 : {yield_spread:.2f}p ({'역전⚠️' if yield_spread<0 else '정상'})\n"
        msg += f"• 구름대 : {cloud_str} / RSI {rsi_val:.1f}\n\n"
        
        msg += "*2️⃣ 리스크 & 심리 (Sentiment)*\n"
        msg += f"• 비트코인 : ${current_btc:,.0f} ({btc_chg:+.2f}%)\n"
        msg += f"• 반도체비 : {'약세⚠️' if semi_weakness>0.005 else '양호'} (괴리 {semi_weakness*100:.1f}%)\n"
        msg += f"• 하이일드 : {'이탈⚠️' if hyg_drawdown<-0.3 else '유입'} (낙폭 {hyg_drawdown:.2f}%)\n"
        msg += f"• 공포지수 : {current_vix:.2f} (추세: {'확산↗' if vix_trend>0 else '진정↘'})\n"
        msg += f"• 뉴스점수 : {news_score:.2f} ({'악재' if news_score<-0.2 else '중립/호재'})\n"
        if worst_title and news_score < -0.2:
            msg += f"  └ 🗞 _{worst_title}_\n"
            
        msg += "\n───────────────\n"
        msg += "*📊 종목별 위험도 랭킹 (개별뉴스 반영)*\n"
        
        for item in stock_results:
            icon = "🔴" if item['score'] >= 60 else "🟡" if item['score'] >= 30 else "🟢"
            reason_str = ", ".join(item['reasons']) if item['reasons'] else "특이사항 없음"
            msg += f"{icon} *{item['ticker']}*: {item['score']}점 ({item['change']:+.1f}%)\n"
            if item['score'] >= 30:
                msg += f"  └ {reason_str}\n"
        
        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
