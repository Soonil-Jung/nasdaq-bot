import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pandas_datareader.data as web
from ta.trend import IchimokuIndicator
from ta.momentum import RSIIndicator
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from GoogleNews import GoogleNews
from datetime import datetime, timedelta

# ======================================================
# ▼▼▼ 사용자 설정 정보 ▼▼▼
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# 분석할 개별 종목 (순서 고정)
TARGET_STOCKS = {
    'GOOGL': 'Google Alphabet stock',
    'MSFT': 'Microsoft stock',
    'TSLA': 'Tesla stock Elon Musk',
    'NVDA': 'Nvidia stock',
    'AMD': 'AMD stock',
    'PLTR': 'Palantir stock',
    'AAPL': 'Apple stock'
}
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 시스템(Final-v24) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except: pass
        
        # ★ [최종 업데이트] 월가 빅마우스 & 연준 핵심 인사 리스트
        self.macro_keywords = [
            'Jerome Powell',    # 연준 의장
            'John Williams',    # (New) 뉴욕 연은 총재 (연준 2인자)
            'Christopher Waller', # (New) 연준 이사 (영향력 1위)
            'Donald Trump',     # 대통령
            'Nick Timiraos',    # 연준 비공식 대변인 (WSJ)
            'Scott Bessent',    # 미 재무장관
            'Kevin Warsh',      # 차기 의장 후보
            'Jamie Dimon',      # JP모건 회장
            'Bill Ackman',      # 헤지펀드 거물
            'Larry Fink',       # 블랙록 회장
            'Michael Burry',    # 빅쇼트
            'Fed Rate',         # 금리 키워드
            'Recession',        # 침체 키워드
            'CPI Inflation'     # 물가 키워드
        ]

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
        try: requests.post(url, data=data)
        except: pass

    # 실시간 데이터 강제 조회 (프리마켓/장전후 포함)
    def get_realtime_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            # 1분봉 데이터 요청 (prepost=True)
            df = stock.history(period='1d', interval='1m', prepost=True)
            if not df.empty:
                return df['Close'].iloc[-1]
            
            # 실패 시 호가 정보 사용
            if stock.fast_info.get('last_price'):
                return stock.fast_info.get('last_price')
        except: pass
        return None

    # 펀더멘털 데이터 (샴의 법칙 & CPI)
    def get_fundamental_data(self):
        try:
            start_date = datetime.now() - timedelta(days=700)
            unrate = web.DataReader('UNRATE', 'fred', start_date)
            cpi = web.DataReader('CPIAUCSL', 'fred', start_date)
            
            # 샴의 법칙 (최근 3개월 이동평균 - 12개월 최저치 >= 0.5%)
            unrate['MA3'] = unrate['UNRATE'].rolling(window=3).mean()
            current_ma3 = unrate['MA3'].iloc[-1]
            low_12m = unrate['UNRATE'].iloc[-14:-1].min()
            sahm_score = current_ma3 - low_12m
            is_recession = sahm_score >= 0.50
            
            # CPI YoY
            cpi_yoy = (cpi['CPIAUCSL'].iloc[-1] - cpi['CPIAUCSL'].iloc[-13]) / cpi['CPIAUCSL'].iloc[-13] * 100
            
            return {
                "unrate": unrate['UNRATE'].iloc[-1],
                "sahm_score": sahm_score,
                "is_recession": is_recession,
                "cpi_yoy": cpi_yoy
            }
        except: return None

    def get_news_sentiment(self, target_keywords):
        try:
            googlenews = GoogleNews(lang='en', period='1d')
            total_score = 0
            count = 0
            worst_title = ""
            worst_link = ""
            worst_source = ""
            min_score = 1.0 

            search_list = [target_keywords] if isinstance(target_keywords, str) else target_keywords

            for key in search_list:
                googlenews.clear()
                googlenews.search(key)
                results = googlenews.results(sort=True)
                if not results: continue
                
                for item in results[:2]:
                    try:
                        title = item['title']
                        link = item['link']
                        media = item['media']
                        res = self.nlp(title[:512])[0]
                        score = res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0
                        total_score += score
                        count += 1
                        if score < min_score and score < -0.5:
                            min_score = score
                            worst_title = title
                            worst_link = link
                            worst_source = media
                    except: continue
            avg_score = total_score / count if count > 0 else 0
            return avg_score, worst_title, worst_link, worst_source
        except: return 0, "", "", ""

    def get_market_data(self):
        try:
            macro_tickers = ['NQ=F', 'QQQ', '^VIX', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', 'BTC-USD', '^IRX']
            all_tickers = macro_tickers + list(TARGET_STOCKS.keys())
            
            # 차트 분석용 1시간봉 (지표 계산)
            data = yf.download(all_tickers, period='5d', interval='1h', progress=False)

            if isinstance(data.columns, pd.MultiIndex): 
                dfs = {}
                df_macro = pd.DataFrame()
                df_macro['Close'] = data['Close']['NQ=F']
                df_macro['High'] = data['High']['NQ=F']
                df_macro['Low'] = data['Low']['NQ=F']
                df_macro['Volume'] = data['Volume']['QQQ']
                # 매크로 지표 정리
                for ticker, col in {'^VIX':'VIX', 'DX-Y.NYB':'DXY', 'SOXX':'SOXX', 'HYG':'HYG', '^TNX':'TNX', '^IRX':'IRX', 'BTC-USD':'BTC'}.items():
                    df_macro[col] = data['Close'][ticker]
                
                df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None)
                df_macro = df_macro.ffill().bfill().dropna()
                dfs['MACRO'] = df_macro

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
        if df_stock.empty: return None

        # [1] 실시간 가격 (프리마켓 반영)
        live_price = self.get_realtime_price(ticker)
        current_price = live_price if live_price else df_stock['Close'].iloc[-1]

        try:
            prev_close = yf.Ticker(ticker).info.get('previousClose')
            if not prev_close: prev_close = df_stock['Close'].iloc[-8]
        except: prev_close = df_stock['Close'].iloc[-8]

        daily_pct = (current_price - prev_close) / prev_close * 100

        # [2] 기술적 지표
        ichimoku = IchimokuIndicator(high=df_stock['High'], low=df_stock['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        rsi_val = RSIIndicator(close=df_stock['Close'], window=14).rsi().iloc[-1]
        
        df_stock['Vol_MA20'] = df_stock['Volume'].rolling(window=20).mean()
        vol_ratio = 0
        if df_stock['Vol_MA20'].iloc[-1] > 0:
            vol_ratio = df_stock['Volume'].iloc[-1] / df_stock['Vol_MA20'].iloc[-1]

        qqq_chg = 0
        try:
            nq_live = self.get_realtime_price('NQ=F')
            if not nq_live: nq_live = df_macro['Close'].iloc[-1]
            qqq_now = nq_live
            qqq_prev = df_macro['Close'].iloc[-24] 
            qqq_chg = (qqq_now - qqq_prev) / qqq_prev * 100
        except: pass
        relative_strength = daily_pct - qqq_chg

        # [3] 뉴스 분석 (언론사 포함)
        search_keyword = TARGET_STOCKS.get(ticker, ticker)
        news_score, worst_news, worst_link, worst_source = self.get_news_sentiment(search_keyword)

        # [4] 점수 산출
        danger_score = 0
        reasons = []
        high_beta = ['TSLA', 'NVDA', 'AMD', 'PLTR']
        drop_threshold = -3.5 if ticker in high_beta else -2.0

        if daily_pct < drop_threshold:
            danger_score += 30
            reasons.append(f"📉 폭락")
        if relative_strength < -1.5: 
            danger_score += 15
            reasons.append(f"상대적 약세")
        if current_price < span_a:
            danger_score += 20
            reasons.append("☁️ 구름대 이탈")
        if rsi_val < 30:
            danger_score += 10
            reasons.append(f"과매도({rsi_val:.0f})")
        if vol_ratio > 2.0:
            danger_score += 15
            reasons.append(f"거래량폭발")
        if news_score < -0.3:
            danger_score += 20
            if worst_news and worst_link:
                clean_title = worst_news[:25] + "..." if len(worst_news) > 25 else worst_news
                source_tag = f"[{worst_source}]" if worst_source else "[News]"
                reasons.append(f"📰 {source_tag} [{clean_title}]({worst_link})")
            else:
                reasons.append(f"📰 악재 뉴스")

        return {
            "ticker": ticker,
            "price": current_price,
            "change": daily_pct,
            "score": min(danger_score, 100),
            "reasons": reasons
        }

    def analyze_danger(self):
        dfs = self.get_market_data()
        if not dfs or 'MACRO' not in dfs: return
        df = dfs['MACRO']

        # --- [PART 1] 매크로 & 펀더멘털 분석 ---
        # 기술적 지표 계산 (차트 기반)
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        # ★ 실시간 데이터 강제 업데이트 (1분봉/호가)
        live_price = self.get_realtime_price('NQ=F')
        current_close = live_price if live_price else df['Close'].iloc[-1]

        daily_chg = (current_close - df['Close'].iloc[-24]) / df['Close'].iloc[-24] * 100 
        hourly_chg = (current_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
        
        avg_vol = df['Vol_MA20'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else current_vol / avg_vol
        
        # 매크로 지표 실시간
        current_dxy = self.get_realtime_price('DX-Y.NYB') or df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[-24]) / df['DXY'].iloc[-24] * 100
        
        current_tnx = self.get_realtime_price('^TNX') or df['TNX'].iloc[-1]
        current_irx = self.get_realtime_price('^IRX') or df['IRX'].iloc[-1]
        yield_spread = current_tnx - current_irx
        irx_chg = (current_irx - df['IRX'].iloc[-24]) / df['IRX'].iloc[-24] * 100
        
        current_btc = self.get_realtime_price('BTC-USD') or df['BTC'].iloc[-1]
        btc_chg = (current_btc - df['BTC'].iloc[-24]) / df['BTC'].iloc[-24] * 100
        
        nq_ret = current_close / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 
        
        hyg_high = df['HYG'].max()
        current_hyg = self.get_realtime_price('HYG') or df['HYG'].iloc[-1]
        hyg_drawdown = (current_hyg - hyg_high) / hyg_high * 100

        current_vix = self.get_realtime_price('^VIX') or df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1]

        # 뉴스 & 펀더멘털
        news_score, worst_title, worst_link, worst_source = self.get_news_sentiment(self.macro_keywords)
        fund_data = self.get_fundamental_data()

        # 점수 산출
        danger_score = 0
        reasons = []
        if daily_chg < -1.5: danger_score += 20; reasons.append(f"📉 추세 하락")
        if hourly_chg < -0.8: danger_score += 15; reasons.append(f"⚡ 투매 발생")
        if current_close < span_a: danger_score += 20; reasons.append("☁️ 구름대 이탈")
        if vol_ratio > 1.5: danger_score += 15; reasons.append(f"📢 거래량 폭증")
        if dxy_chg > 0.3: danger_score += 10; reasons.append(f"💵 달러 강세")
        if irx_chg > 2.0: danger_score += 10; reasons.append(f"🏦 단기금리 급등")
        if btc_chg < -3.0: danger_score += 15; reasons.append(f"📉 비트코인 급락")
        if semi_weakness > 0.005: danger_score += 10; reasons.append(f"📉 반도체 약세")
        if hyg_drawdown < -0.3: danger_score += 15; reasons.append(f"💸 스마트머니 이탈")
        if news_score < -0.2: danger_score += 10; reasons.append(f"📰 뉴스 심리 악화")
        if fund_data and fund_data['is_recession']:
            danger_score += 30; reasons.append(f"🛑 샴의 법칙 발동 (침체)")

        danger_score = min(danger_score, 100)

        # --- [PART 2] 개별 종목 분석 ---
        stock_results = []
        for ticker in TARGET_STOCKS.keys():
            if ticker in dfs:
                res = self.analyze_individual(ticker, dfs[ticker], df)
                if res: stock_results.append(res)

        # --- [PART 3] 메시지 작성 ---
        status_emoji = '🔴 위험' if danger_score >= 60 else '🟡 주의' if danger_score >= 35 else '🟢 안정'
        cloud_str = "하단 이탈 🚨" if current_close < span_a else "구름대 위 ✅"
        spread_str = "정상 ✅" if yield_spread >= 0 else "역전(침체) ⚠️"
        semi_str = "약세 ⚠️" if semi_weakness > 0.005 else "양호 ✅"
        hyg_str = "이탈 ⚠️" if hyg_drawdown < -0.3 else "유입 ✅"
        vix_str = "확산 ↗" if vix_trend > 0 else "안정 ↘"
        
        fund_str = "N/A"
        if fund_data:
            rec_emoji = "🚨 침체 경고" if fund_data['is_recession'] else "안정"
            fund_str = f"실업률 {fund_data['unrate']}% / CPI {fund_data['cpi_yoy']:.1f}% ({rec_emoji})"

        now_kst = datetime.now() + timedelta(hours=9)
        
        msg = f"🔔 *AI 퀀트 시장 정밀 분석*\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 종합상태: {status_emoji} ({danger_score}점)\n\n"
        
        msg += "*1️⃣ 매크로 & 펀더멘털 (Macro)*\n"
        msg += f"• 경제지표 : {fund_str}\n"
        msg += f"• 달러(DXY): {current_dxy:.2f} ({dxy_chg:+.2f}%)\n"
        msg += f"• 금리(10Y): {current_tnx:.2f}% / (3M): {current_irx:.2f}%\n"
        msg += f"• 장단기차 : {yield_spread:.2f}p ({spread_str})\n\n"
        
        msg += "*2️⃣ 기술적 지표 (Technical)*\n"
        msg += f"• 나스닥 : {current_close:,.2f} ({daily_chg:+.2f}%)\n"
        msg += f"• 1시간봉 : {hourly_chg:+.2f}% / 거래 {int(vol_ratio*100)}%\n"
        msg += f"• 구름대 : {cloud_str} / RSI {rsi_val:.1f}\n\n"
        
        msg += "*3️⃣ 리스크 & 심리 (Sentiment)*\n"
        msg += f"• 비트코인 : ${current_btc:,.0f} ({btc_chg:+.2f}%)\n"
        msg += f"• 반도체비 : {semi_str} (괴리 {semi_weakness*100:.1f}%)\n"
        msg += f"• 하이일드 : {hyg_str} (낙폭 {hyg_drawdown:.2f}%)\n"
        msg += f"• 공포지수 : {current_vix:.2f} (추세: {vix_str})\n"
        msg += f"• 뉴스점수 : {news_score:.2f} ({'악재' if news_score<-0.2 else '중립/호재'})\n"
        if worst_title and news_score < -0.2:
            source_tag = f"[{worst_source}]" if worst_source else "[News]"
            msg += f"  └ 🗞 {source_tag} [{worst_title[:20]}...]({worst_link})\n"
            
        msg += "\n*📋 [상세 위험 요인 분석]*\n"
        if reasons:
            msg += "\n".join(["🚨 " + r for r in reasons])
        else:
            msg += "✅ 특이사항 없음 (안정적)"

        msg += "\n\n───────────────\n"
        msg += "*📊 종목별 위험도 (현재가/등락률)*\n"
        
        for item in stock_results:
            icon = "🔴" if item['score'] >= 60 else "🟡" if item['score'] >= 30 else "🟢"
            price_info = f"${item['price']:,.2f} ({item['change']:+.2f}%)"
            msg += f"{icon} *{item['ticker']}*: {price_info} | {item['score']}점\n"
            if item['score'] >= 30:
                reason_str = ", ".join(item['reasons']) if item['reasons'] else ""
                msg += f"  └ {reason_str}\n"
        
        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
