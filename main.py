import os
import time
import re
import asyncio
import aiohttp
import feedparser
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from ta.trend import IchimokuIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# ======================================================
# ▼▼▼ 사용자 설정 정보 ▼▼▼
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

TARGET_STOCKS = {
    'GOOGL': 'Google Alphabet',
    'MSFT': 'Microsoft',
    'TSLA': 'Tesla',
    'NVDA': 'Nvidia',
    'AMD': 'AMD',
    'PLTR': 'Palantir',
    'AAPL': 'Apple'
}

STOCK_PARAMS = {
    'GOOGL': {'crash': 40, 'rel': 20, 'tech': 20, 'sell': 60},
    'MSFT':  {'crash': 30, 'rel': 10, 'tech': 20, 'sell': 60},
    'TSLA':  {'crash': 40, 'rel': 10, 'tech': 20, 'sell': 60},
    'NVDA':  {'crash': 40, 'rel': 10, 'tech': 20, 'sell': 60},
    'AMD':   {'crash': 30, 'rel': 20, 'tech': 10, 'sell': 60},
    'PLTR':  {'crash': 40, 'rel': 15, 'tech': 20, 'sell': 60},
    'AAPL':  {'crash': 20, 'rel': 20, 'tech': 20, 'sell': 60}
}

# [v2.1 파라미터] 민감도 높음
W_TREND_MACRO = 35 
W_VOL_MACRO = 20
W_MACRO_MACRO = 10 
TH_SELL = 60
TH_BUY = 30
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 퀀트봇 v2.1 (Sensitive + Detail) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"⚠️ AI 모델 로드 실패: {e}")
            self.nlp = None
        
        self.macro_keywords = [
            'Federal Reserve', 'The Fed', 'Jerome Powell', 'FOMC', 
            'US CPI Inflation', 'US Recession', 'US Stagflation', 'US Economy',
            'Nasdaq Crash', 'Wall Street', 'Geopolitical tension'
        ]

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
        try: requests.post(url, data=data, timeout=10)
        except: pass

    async def fetch_feed(self, session, keyword):
        url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    return keyword, await response.text()
        except: pass
        return keyword, None

    async def process_news_async(self, keywords):
        if not self.nlp: return 0, "", "", "", ""
        search_list = [keywords] if isinstance(keywords, str) else keywords
        total_score, count = 0, 0
        worst_info = {"score": 1.0, "title": "", "link": "", "source": "", "summary": ""}

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_feed(session, key) for key in search_list]
            feeds = await asyncio.gather(*tasks)

            for key, xml_data in feeds:
                if not xml_data: continue
                feed = feedparser.parse(xml_data)
                for entry in feed.entries[:3]:
                    try:
                        title = entry.title
                        link = entry.link
                        source = entry.source.title if 'source' in entry else "News"
                        raw_sum = entry.get('summary', '') or entry.get('description', '')
                        clean_sum = BeautifulSoup(raw_sum, "html.parser").get_text().strip()
                        clean_title = BeautifulSoup(title, "html.parser").get_text()
                        res = self.nlp(clean_title[:512])[0]
                        score = res['score'] if res['label'] == 'positive' else -res['score']
                        total_score += score
                        count += 1
                        if score < worst_info["score"] and score < -0.5:
                            worst_info = {"score": score, "title": clean_title, "link": link, "source": source, "summary": clean_sum}
                    except: continue
        
        avg_score = total_score / count if count > 0 else 0
        return avg_score, worst_info["title"], worst_info["link"], worst_info["source"], worst_info["summary"]

    def get_news_sentiment(self, target_keywords):
        try: return asyncio.run(self.process_news_async(target_keywords))
        except: return 0, "", "", "", ""

    def get_realtime_price(self, ticker):
        try: return yf.Ticker(ticker).fast_info.get('last_price', None)
        except: return None

    def get_market_data(self):
        try:
            macro_tickers = ['NQ=F', 'QQQ', '^VIX', '^VIX3M', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', '^IRX', 'BTC-USD']
            all_tickers = macro_tickers + list(TARGET_STOCKS.keys())
            data = yf.download(all_tickers, period='1y', interval='1d', prepost=True, progress=False, ignore_tz=True)
            
            if isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns:
                dfs = {}
                df_macro = pd.DataFrame()
                close_data = data['Close']
                if 'NQ=F' not in close_data.columns: return {}
                
                df_macro['Close'] = close_data['NQ=F']
                df_macro['High'] = data['High']['NQ=F'] if 'High' in data.columns else close_data['NQ=F']
                df_macro['Low'] = data['Low']['NQ=F'] if 'Low' in data.columns else close_data['NQ=F']
                df_macro['Volume'] = data['Volume']['QQQ'] if 'Volume' in data.columns else 0
                
                ticker_map = {'^VIX': 'VIX', '^VIX3M': 'VIX3M', 'DX-Y.NYB': 'DXY', 'SOXX': 'SOXX', 'HYG': 'HYG', '^TNX': 'TNX', '^IRX': 'IRX', 'BTC-USD': 'BTC'}
                for t, col in ticker_map.items():
                    if t in close_data.columns: df_macro[col] = close_data[t]
                
                df_macro = df_macro.ffill().bfill()
                dfs['MACRO'] = df_macro
                
                for ticker in TARGET_STOCKS.keys():
                    if ticker in close_data.columns:
                        df_stock = pd.DataFrame()
                        df_stock['Close'] = close_data[ticker]
                        df_stock['High'] = data['High'][ticker]
                        df_stock['Low'] = data['Low'][ticker]
                        df_stock['Volume'] = data['Volume'][ticker] if 'Volume' in data.columns else 0
                        dfs[ticker] = df_stock.dropna()
                return dfs
            return {}
        except: return {}

    def get_fundamental_data(self):
        try:
            start = datetime.now() - timedelta(days=700)
            unrate = web.DataReader('UNRATE', 'fred', start)
            unrate['MA3'] = unrate['UNRATE'].rolling(3).mean()
            score = unrate['MA3'].iloc[-1] - unrate['UNRATE'].iloc[-14:-1].min()
            return {"unrate": unrate['UNRATE'].iloc[-1], "is_recession": score >= 0.50}
        except: return None

    def analyze_individual(self, ticker, df_stock, df_macro):
        if len(df_stock) < 30: return None
        params = STOCK_PARAMS.get(ticker, {'crash': 30, 'rel': 15, 'tech': 15, 'sell': 60})
        
        curr = df_stock['Close'].iloc[-1]
        prev = df_stock['Close'].iloc[-2]
        chg = (curr - prev) / prev * 100

        ichimoku = IchimokuIndicator(high=df_stock['High'], low=df_stock['Low'], window1=9, window2=26, window3=52)
        cloud = min(ichimoku.ichimoku_a().iloc[-26], ichimoku.ichimoku_b().iloc[-26])
        ma20 = SMAIndicator(close=df_stock['Close'], window=20).sma_indicator().iloc[-1]
        ma50 = SMAIndicator(close=df_stock['Close'], window=50).sma_indicator().iloc[-1]
        rsi = RSIIndicator(close=df_stock['Close'], window=14).rsi().iloc[-1]
        
        rel_str = chg - ((df_macro['Close'].iloc[-1] - df_macro['Close'].iloc[-2]) / df_macro['Close'].iloc[-2] * 100)
        news_score, wn, wl, ws, wsum = self.get_news_sentiment(ticker)

        score = 0
        reasons = []
        if chg < -3.0: score += params['crash']; reasons.append(f"📉 폭락 ({chg:.1f}%)")
        if rel_str < -1.5: score += params['rel']; reasons.append("상대적 약세")
        
        tech = []
        if curr < cloud: tech.append("구름대 이탈")
        if ma20 < ma50 and curr < ma20: tech.append("데드크로스")
        if rsi < 30: tech.append("과매도")
        if tech: score += params['tech']; reasons.append(f"기술적({','.join(tech)})")
            
        if news_score < -0.3:
            score += 20
            reasons.append(f"📰 악재: {wn[:20]}...")

        return {"ticker": ticker, "price": curr, "change": chg, "score": min(score, 100), "threshold": params['sell'], "reasons": reasons}

    def analyze_danger(self):
        dfs = self.get_market_data()
        if not dfs or 'MACRO' not in dfs: return
        df = dfs['MACRO']
        
        now = datetime.now() + timedelta(hours=9)
        weekday = now.weekday()
        hour = now.hour
        is_weekend = (weekday == 6) or (weekday == 5 and hour >= 9) or (weekday == 0 and hour < 8)

        # BTC 데이터 준비
        live_btc = self.get_realtime_price('BTC-USD')
        curr_btc = live_btc if live_btc else df['BTC'].iloc[-1]
        btc_prev = df['BTC'].iloc[-2]
        btc_chg = (curr_btc - btc_prev) / btc_prev * 100
        
        news_score, w_title, w_link, w_src, w_sum = self.get_news_sentiment(self.macro_keywords)

        # [주말 모드]
        if is_weekend:
            btc_emoji = "🔥 급등" if btc_chg > 3 else "📉 급락" if btc_chg < -3 else "➡️ 횡보"
            news_emoji = "😊 호재/중립" if news_score >= -0.2 else "🚨 악재 우세"
            msg = f"☕ *주말 시장 핵심 브리핑*\n📅 {now.strftime('%Y-%m-%d %H:%M')} (KST)\n\n"
            msg += f"*1️⃣ 비트코인 (24h Live)*\n• 가격 : ${curr_btc:,.0f} ({btc_chg:+.2f}%)\n• 추세 : {btc_emoji}\n\n"
            msg += f"*2️⃣ 주말 주요 뉴스*\n• 심리점수 : {news_score:.2f} ({news_emoji})\n"
            if w_title and news_score < -0.2:
                cl_title = re.sub(r'[\[\]\*\_]', '', w_title)[:30] + "..."
                msg += f"  └ 🗞 [{w_src}] [{cl_title}]({w_link})\n"
                if w_sum: msg += f"    📝 {w_sum}\n"
            self.send_telegram(msg)
            return

        # [평일 분석 - v2.1 로직]
        curr = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        chg = (curr - prev) / prev * 100
        
        high_52w = df['Close'].rolling(252).max().iloc[-1]
        drawdown = (curr - high_52w) / high_52w * 100
        
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        ma120 = df['Close'].rolling(120).mean().iloc[-1]
        
        vix = df['VIX'].iloc[-1]
        vix3m = df['VIX3M'].iloc[-1] if 'VIX3M' in df.columns else vix * 1.1
        
        soxx = df['SOXX'].iloc[-1]
        soxx_chg = (soxx - df['SOXX'].iloc[-2]) / df['SOXX'].iloc[-2] * 100
        hyg = df['HYG'].iloc[-1]
        hyg_chg = (hyg - df['HYG'].iloc[-2]) / df['HYG'].iloc[-2] * 100
        
        dxy = df['DXY'].iloc[-1]
        dxy_chg = (dxy - df['DXY'].iloc[-2]) / df['DXY'].iloc[-2] * 100
        
        tnx = df['TNX'].iloc[-1]
        irx = df['IRX'].iloc[-1]
        spread = tnx - irx

        danger_score = 0
        reasons = []
        
        # --- [v2.1 특징: 모든 변수 점수 반영] ---
        
        # 1. 추세 & 낙폭
        if chg < -1.5: danger_score += W_TREND_MACRO; reasons.append(f"📉 지수 급락 ({chg:.2f}%)")
        if drawdown < -20: danger_score += 30; reasons.append(f"📉 폭락장 지속 (고점대비 {drawdown:.1f}%)")
        elif drawdown < -10: danger_score += 15; reasons.append(f"📉 조정장 진입 (고점대비 {drawdown:.1f}%)")
        
        # 2. 공포 (VIX 구조 & 절대값)
        if vix > vix3m * 1.02: danger_score += 35; reasons.append(f"🚨 VIX 역전 (시스템 위기)")
        elif vix > 30: danger_score += 20; reasons.append(f"😱 극단적 공포 ({vix:.1f})")
            
        # 3. 매크로 (달러 & 금리)
        if dxy_chg > 0.5: danger_score += W_MACRO_MACRO; reasons.append("💵 달러 급등")
        if spread < -0.5: danger_score += 10; reasons.append("⚠️ 금리차 역전 심화")
        
        # 4. 뉴스 심리
        if news_score < -0.3: danger_score += 15; reasons.append(f"📰 뉴스 심리 악화")
            
        # 5. 자산군 위험 (모두 반영)
        if btc_chg < -3.0: danger_score += 10; reasons.append(f"📉 비트코인 급락 ({btc_chg:.1f}%)")
        if soxx_chg < (chg - 1.0): danger_score += 10; reasons.append(f"📉 반도체 상대적 약세")
        if hyg_chg < -0.5: danger_score += 10; reasons.append(f"⚠️ 하이일드 자금 이탈")

        # 6. 기술적 역배열
        if ma20 < ma50 and curr < ma20: danger_score += 25; reasons.append("📉 완전 역배열")
        
        # 7. 추세 버퍼
        if curr > ma120: danger_score -= 15

        fund = self.get_fundamental_data()
        if fund and fund['is_recession']: danger_score += 30; reasons.append("🛑 경기 침체 확정")

        danger_score = max(0, min(danger_score, 100))
        
        status = "🟢 안정"
        if danger_score >= TH_SELL: status = "🔴 위험 (현금화 권장)"
        elif danger_score >= TH_BUY: status = "🟡 주의 (비중축소)"
        
        stock_results = []
        for t in TARGET_STOCKS:
            if t in dfs:
                res = self.analyze_individual(t, dfs[t], df)
                if res: stock_results.append(res)
        stock_results.sort(key=lambda x: x['score'], reverse=True)

        # [상태 텍스트: 보합/동행 반영]
        trend_st = "상승(120선 위)✅" if curr > ma120 else "하락(120선 아래)⚠️"
        vix_st = "역전🚨" if vix > vix3m * 1.02 else "정상✅"
        spread_st = "역전⚠️" if spread < 0 else "정상✅"
        recess_st = "침체🛑" if fund and fund['is_recession'] else "정상✅"
        
        nq_emoji = "🔥" if chg > 0.5 else "💧" if chg < -0.5 else "➡️"
        dxy_st = "⚠️악재" if dxy_chg > 0.2 else "✅호재" if dxy_chg < -0.2 else "➡️"
        btc_st = "🔥RiskOn" if btc_chg > 1 else "💧RiskOff" if btc_chg < -1 else "➡️"

        if soxx_chg > chg + 0.2: soxx_st = "🚀주도"
        elif soxx_chg < chg - 0.2: soxx_st = "🐢소외"
        else: soxx_st = "➡️동행"

        if hyg_chg > 0.05: hyg_st = "💰유입"
        elif hyg_chg < -0.05: hyg_st = "💸이탈"
        else: hyg_st = "➡️보합"

        msg = f"🔔 *AI 마켓 워치 (v2.1)*\n📅 {now.strftime('%Y-%m-%d %H:%M')} (KST)\n🚦 시장상태: {status} ({danger_score}점)\n\n"
        
        msg += "*1️⃣ 핵심 위험 요인*\n"
        if reasons: msg += "\n".join(["▪ " + r for r in reasons])
        else: msg += "▪ 특이사항 없음 (양호)"
        
        msg += f"\n\n*2️⃣ 매크로 대시보드*\n"
        msg += f"• 나스닥: {curr:,.0f} ({chg:+.2f}%) {nq_emoji}\n"
        msg += f"• 추세: {trend_st} | 낙폭: {drawdown:.1f}%\n"
        msg += f"• VIX구조: {vix_st} ({vix:.1f}/{vix3m:.1f})\n"
        msg += f"• 달러: {dxy:.2f} ({dxy_chg:+.2f}%) {dxy_st}\n"
        
        # [수정된 부분: 금리 상세 표시]
        msg += f"• 금리차: {spread:.2f}p ({tnx:.2f}%/{irx:.2f}%) {spread_st}\n"
        
        msg += f"• 비트코인: ${curr_btc:,.0f} ({btc_chg:+.2f}%) {btc_st}\n"
        msg += f"• 반도체: ${df['SOXX'].iloc[-1]:.0f} ({soxx_chg:+.2f}%) {soxx_st}\n"
        msg += f"• 하이일드: ${df['HYG'].iloc[-1]:.2f} ({hyg_chg:+.2f}%) {hyg_st}\n"
        
        if fund: msg += f"• 실업률: {fund['unrate']}% ({recess_st})\n"
        
        if w_title:
            cl_title = re.sub(r'[\[\]\*\_]', '', w_title)[:25] + "..."
            src_tag = f"[{w_src}]" if w_src else "[News]"
            msg += f"\n*3️⃣ 주요 뉴스 심리*\n• 점수: {news_score:.2f}\n• 이슈: {src_tag} [{cl_title}]({w_link})\n"
            if w_sum: msg += f"  └ 📝 {w_sum}\n"
            
        msg += "\n*📊 관심 종목 위험도*\n"
        for s in stock_results:
            icon = "🔴" if s['score'] >= s['threshold'] else "🟡" if s['score'] >= 40 else "🟢"
            msg += f"{icon} {s['ticker']}: {s['score']}점 ({s['change']:+.1f}%)\n"
            if s['reasons']: msg += f"  └ {', '.join(s['reasons'])}\n"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
