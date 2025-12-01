import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pandas_datareader.data as web
from ta.trend import IchimokuIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from GoogleNews import GoogleNews
from datetime import datetime, timedelta

# ======================================================
# ▼▼▼ 사용자 설정 정보 ▼▼▼
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

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
        print("🤖 AI 시스템(v38-MA-Values) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except: pass
        
        self.macro_keywords = [
            'Federal Reserve', 'The Fed', 'US Fed', 'FOMC', 'US Treasury', 'White House Economy',
            'Jerome Powell', 'Donald Trump', 'Nick Timiraos', 'Scott Bessent',
            'Kevin Warsh', 'Jamie Dimon', 'Bill Ackman', 'Larry Fink', 'Michael Burry',
            'John Williams', 'Christopher Waller',
            'CPI Inflation', 'PCE Inflation', 'PPI Inflation', 'GDP Growth', 'Recession', 'Stagflation',
            'Jobs Report', 'Nonfarm Payrolls', 'Unemployment Rate', 'ADP Report', 'JOLTS',
            'Bloomberg Markets', 'Goldman Sachs', 'Morgan Stanley', 'JP Morgan'
        ]

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
        try: requests.post(url, data=data)
        except: pass

    def get_realtime_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='1d', interval='1m', prepost=True, auto_adjust=True)
            if not df.empty:
                return df['Close'].iloc[-1]
            if stock.fast_info.get('last_price'):
                return stock.fast_info.get('last_price')
        except: pass
        return None

    def get_realtime_chart(self, ticker):
        try:
            df = yf.download(ticker, period='10d', interval='1h', prepost=True, progress=False, ignore_tz=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except: return None

    def get_fundamental_data(self):
        try:
            start_date = datetime.now() - timedelta(days=700)
            unrate = web.DataReader('UNRATE', 'fred', start_date)
            cpi = web.DataReader('CPIAUCSL', 'fred', start_date)
            
            if unrate.empty or cpi.empty: return None

            unrate['MA3'] = unrate['UNRATE'].rolling(window=3).mean()
            current_ma3 = unrate['MA3'].iloc[-1]
            low_12m = unrate['UNRATE'].iloc[-14:-1].min()
            sahm_score = current_ma3 - low_12m
            is_recession = sahm_score >= 0.50
            cpi_yoy = (cpi['CPIAUCSL'].iloc[-1] - cpi['CPIAUCSL'].iloc[-13]) / cpi['CPIAUCSL'].iloc[-13] * 100
            
            return {"unrate": unrate['UNRATE'].iloc[-1], "sahm_score": sahm_score, "is_recession": is_recession, "cpi_yoy": cpi_yoy}
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
                        if '&ved=' in link: link = link.split('&ved=')[0]
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
            
            data = yf.download(all_tickers, period='1mo', interval='1h', prepost=True, progress=False, ignore_tz=True, auto_adjust=True)

            if isinstance(data.columns, pd.MultiIndex): 
                dfs = {}
                df_macro = pd.DataFrame()
                
                if 'Close' not in data.columns or 'NQ=F' not in data['Close'].columns: return {}

                df_macro['Close'] = data['Close']['NQ=F']
                df_macro['High'] = data['High']['NQ=F']
                df_macro['Low'] = data['Low']['NQ=F']
                df_macro['Volume'] = data['Volume']['QQQ']
                
                for ticker, col in {'^VIX':'VIX', 'DX-Y.NYB':'DXY', 'SOXX':'SOXX', 'HYG':'HYG', '^TNX':'TNX', '^IRX':'IRX', 'BTC-USD':'BTC'}.items():
                    if ticker in data['Close'].columns:
                        df_macro[col] = data['Close'][ticker]
                    else: df_macro[col] = np.nan

                df_macro = df_macro.ffill().bfill().dropna()
                dfs['MACRO'] = df_macro

                for ticker in TARGET_STOCKS.keys():
                    if ticker in data['Close'].columns:
                        df_stock = pd.DataFrame()
                        df_stock['Close'] = data['Close'][ticker]
                        df_stock['High'] = data['High'][ticker]
                        df_stock['Low'] = data['Low'][ticker]
                        df_stock['Volume'] = data['Volume'][ticker]
                        df_stock = df_stock.dropna()
                        dfs[ticker] = df_stock
                return dfs
            else: return {}
        except: return {}

    def analyze_individual(self, ticker, df_stock, df_macro):
        if df_stock.empty or len(df_stock) < 30: return None

        live_price = self.get_realtime_price(ticker)
        current_price = live_price if live_price else df_stock['Close'].iloc[-1]

        try:
            prev_close = yf.Ticker(ticker).info.get('previousClose')
            if not prev_close: prev_close = df_stock['Close'].iloc[-8]
        except: prev_close = df_stock['Close'].iloc[-8]

        if prev_close == 0: daily_pct = 0
        else: daily_pct = (current_price - prev_close) / prev_close * 100

        ichimoku = IchimokuIndicator(high=df_stock['High'], low=df_stock['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-26]
        span_b = ichimoku.ichimoku_b().iloc[-26]
        cloud_bottom = min(span_a, span_b)
        
        # [이평선 계산]
        ma20 = SMAIndicator(close=df_stock['Close'], window=20).sma_indicator().iloc[-1]
        ma50 = SMAIndicator(close=df_stock['Close'], window=50).sma_indicator().iloc[-1]
        ma120 = SMAIndicator(close=df_stock['Close'], window=120).sma_indicator().iloc[-1]
        
        # 기울기용
        sma20_prev = SMAIndicator(close=df_stock['Close'], window=20).sma_indicator().iloc[-2]
        sma50_prev = SMAIndicator(close=df_stock['Close'], window=50).sma_indicator().iloc[-2]
        slope20_down = ma20 < sma20_prev
        slope50_down = ma50 < sma50_prev

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
            idx = -24 if len(df_macro) >= 24 else 0
            qqq_prev = df_macro['Close'].iloc[idx] 
            if qqq_prev != 0:
                qqq_chg = (qqq_now - qqq_prev) / qqq_prev * 100
        except: pass
        relative_strength = daily_pct - qqq_chg

        search_keyword = TARGET_STOCKS.get(ticker, ticker)
        news_score, worst_news, worst_link, worst_source = self.get_news_sentiment(search_keyword)

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
        if current_price < cloud_bottom:
            danger_score += 20
            reasons.append("☁️ 구름대 이탈")
        
        # 이평선 위험
        if current_price < ma20 < ma50 < ma120:
            if slope20_down and slope50_down:
                danger_score += 25
                reasons.append("📉 역배열(하락가속)")
            else:
                danger_score += 20
                reasons.append("📉 역배열(완만)")
        elif ma20 < ma50 and current_price < ma20:
            danger_score += 10
            reasons.append("📉 데드크로스")

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
        if not dfs or 'MACRO' not in dfs or dfs['MACRO'].empty: return
        df = dfs['MACRO']
        if len(df) < 30: return 

        now_kst = datetime.now() + timedelta(hours=9)
        weekday = now_kst.weekday() 
        hour = now_kst.hour
        is_weekend_mode = False
        if weekday == 6: is_weekend_mode = True
        elif weekday == 5 and hour >= 9: is_weekend_mode = True
        elif weekday == 0 and hour < 8: is_weekend_mode = True

        live_btc = self.get_realtime_price('BTC-USD')
        current_btc = live_btc if live_btc else df['BTC'].iloc[-1]
        idx_day = -24 if len(df) >= 24 else 0
        btc_chg = (current_btc - df['BTC'].iloc[idx_day]) / df['BTC'].iloc[idx_day] * 100
        news_score, worst_title, worst_link, worst_source = self.get_news_sentiment(self.macro_keywords)

        if is_weekend_mode:
            btc_emoji = "🔥 급등" if btc_chg > 3 else "📉 급락" if btc_chg < -3 else "➡️ 횡보"
            news_emoji = "😊 호재/중립" if news_score >= -0.2 else "🚨 악재 우세"
            msg = f"☕ *주말 시장 핵심 브리핑*\n"
            msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n\n"
            msg += f"*1️⃣ 비트코인 (24h Live)*\n"
            msg += f"• 가격 : ${current_btc:,.0f} ({btc_chg:+.2f}%)\n"
            msg += f"• 추세 : {btc_emoji}\n\n"
            msg += f"*2️⃣ 주말 주요 뉴스*\n"
            msg += f"• 심리점수 : {news_score:.2f} ({news_emoji})\n"
            if worst_title and news_score < -0.2:
                source_tag = f"[{worst_source}]" if worst_source else "[News]"
                msg += f"  └ 🗞 {source_tag} [{worst_title[:30]}...]({worst_link})\n"
            elif news_score >= -0.2:
                msg += "  └ 특이사항 없는 평온한 주말입니다.\n"
            self.send_telegram(msg)
            return

        # [평일 모드]
        nq_chart = self.get_realtime_chart('NQ=F')
        
        if nq_chart is not None and not nq_chart.empty and len(nq_chart) > 30:
            ichimoku = IchimokuIndicator(high=nq_chart['High'], low=nq_chart['Low'], window1=9, window2=26, window3=52)
            span_a = ichimoku.ichimoku_a().iloc[-26]
            span_b = ichimoku.ichimoku_b().iloc[-26]
            
            sma20 = SMAIndicator(close=nq_chart['Close'], window=20).sma_indicator()
            sma50 = SMAIndicator(close=nq_chart['Close'], window=50).sma_indicator()
            sma120 = SMAIndicator(close=nq_chart['Close'], window=120).sma_indicator()
            
            ma20 = sma20.iloc[-1]
            ma50 = sma50.iloc[-1]
            ma120 = sma120.iloc[-1]
            
            slope20_down = ma20 < sma20.iloc[-2]
            slope50_down = ma50 < sma50.iloc[-2]
            
            current_close = nq_chart['Close'].iloc[-1]
            live_price = self.get_realtime_price('NQ=F')
            if live_price: current_close = live_price
        else:
            ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
            span_a = ichimoku.ichimoku_a().iloc[-26]
            span_b = ichimoku.ichimoku_b().iloc[-26]
            
            ma20 = SMAIndicator(close=df['Close'], window=20).sma_indicator().iloc[-1]
            ma50 = SMAIndicator(close=df['Close'], window=50).sma_indicator().iloc[-1]
            ma120 = SMAIndicator(close=df['Close'], window=120).sma_indicator().iloc[-1]
            slope20_down = False
            slope50_down = False
            
            current_close = self.get_realtime_price('NQ=F') or df['Close'].iloc[-1]

        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        cloud_height = cloud_top - cloud_bottom
        
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        idx_hour = -2 if len(df) >= 2 else 0
        daily_chg = (current_close - df['Close'].iloc[idx_day]) / df['Close'].iloc[idx_day] * 100 
        hourly_chg = (current_close - df['Close'].iloc[idx_hour]) / df['Close'].iloc[idx_hour] * 100
        
        avg_vol = df['Vol_MA20'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else current_vol / avg_vol
        
        current_dxy = self.get_realtime_price('DX-Y.NYB') or df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[idx_day]) / df['DXY'].iloc[idx_day] * 100
        
        current_tnx = self.get_realtime_price('^TNX') or df['TNX'].iloc[-1]
        current_irx = self.get_realtime_price('^IRX') or df['IRX'].iloc[-1]
        yield_spread = current_tnx - current_irx
        irx_chg = (current_irx - df['IRX'].iloc[idx_day]) / df['IRX'].iloc[idx_day] * 100
        
        nq_ret = current_close / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 
        
        hyg_high = df['HYG'].max()
        current_hyg = self.get_realtime_price('HYG') or df['HYG'].iloc[-1]
        hyg_drawdown = (current_hyg - hyg_high) / hyg_high * 100

        current_vix = self.get_realtime_price('^VIX') or df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1]
        fund_data = self.get_fundamental_data()

        danger_score = 0
        reasons = []
        if daily_chg < -1.5: danger_score += 20; reasons.append(f"📉 추세 하락")
        if hourly_chg < -0.8: danger_score += 15; reasons.append(f"⚡ 투매 발생")
        
        cloud_status_text = "구름대 위 ✅"
        if current_close < cloud_bottom:
            danger_score += 25
            reasons.append("☁️ 구름대 하단 완전 이탈")
            cloud_status_text = "하단 이탈 (매도) 🚨"
        elif current_close > cloud_top:
            cloud_status_text = "구름대 위 (안정) ✅"
        else:
            if cloud_height > 0:
                pos = (current_close - cloud_bottom) / cloud_height
                if pos < 0.33:
                    danger_score += 10
                    reasons.append("☁️ 구름대 하단 위협")
                    cloud_status_text = "구름대 하단 (불안) ⚡"
                elif pos > 0.66: cloud_status_text = "구름대 상단 (조정) 🌤️"
                else: cloud_status_text = "구름대 중앙 (혼조) 🌫"
            else: cloud_status_text = "구름대 내부 (혼조) 🌫"
            
        ma_status_text = "정배열 ✅"
        if current_close < ma20 < ma50 < ma120:
            if slope20_down and slope50_down:
                danger_score += 25
                reasons.append("📉 역배열(하락가속)")
                ma_status_text = "역배열(가속) 🚨"
            else:
                danger_score += 20
                reasons.append("📉 역배열(하락확정)")
                ma_status_text = "역배열 ⚠️"
        elif ma20 < ma50 and current_close < ma20:
            danger_score += 10
            reasons.append("📉 20/50 데드크로스")
            ma_status_text = "데드크로스 ⚠️"
            
        if vol_ratio > 1.5: danger_score += 15; reasons.append(f"📢 거래량 폭증")
        if dxy_chg > 0.3: danger_score += 10; reasons.append(f"💵 달러 강세")
        if irx_chg > 2.0: danger_score += 10; reasons.append(f"🏦 단기금리 급등")
        if btc_chg < -3.0: danger_score += 15; reasons.append(f"📉 비트코인 급락")
        if semi_weakness > 0.005: danger_score += 10; reasons.append(f"📉 반도체 약세")
        if hyg_drawdown < -0.3: danger_score += 15; reasons.append(f"💸 스마트머니 이탈")
        if news_score < -0.2: danger_score += 10; reasons.append(f"📰 뉴스 심리 악화")
        if fund_data and fund_data['is_recession']: danger_score += 30; reasons.append(f"🛑 샴의 법칙 발동 (침체)")
        danger_score = min(danger_score, 100)

        stock_results = []
        for ticker in TARGET_STOCKS.keys():
            if ticker in dfs:
                res = self.analyze_individual(ticker, dfs[ticker], df)
                if res: stock_results.append(res)

        status_emoji = '🔴 위험' if danger_score >= 60 else '🟡 주의' if danger_score >= 35 else '🟢 안정'
        spread_str = "정상 ✅" if yield_spread >= 0 else "역전(침체) ⚠️"
        semi_str = "약세 ⚠️" if semi_weakness > 0.005 else "양호 ✅"
        hyg_str = "이탈 ⚠️" if hyg_drawdown < -0.3 else "유입 ✅"
        vix_str = "확산 ↗" if vix_trend > 0 else "안정 ↘"
        
        fund_str = "N/A"
        if fund_data:
            rec_emoji = "🚨 침체 경고" if fund_data['is_recession'] else "안정"
            fund_str = f"실업률 {fund_data['unrate']}% / CPI {fund_data['cpi_yoy']:.1f}% ({rec_emoji})"

        msg = f"🔔 *AI 퀀트 시장 정밀 분석*\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 종합상태: {status_emoji} ({danger_score}점)\n\n"
        
        msg += "*1️⃣ 매크로 & 펀더멘털 (Macro)*\n"
        msg += f"• 경제지표 : {fund_str}\n"
        msg += f"• 달러(DXY): {current_dxy:.2f} ({dxy_chg:+.2f}%)\n"
        msg += f
