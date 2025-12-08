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
# ▼▼▼ [STRATEGY UPGRADE] 포트폴리오 리밸런싱 ▼▼▼
# 설명: 기존 Tech 일변도에서 'AI 전력/인프라' 핵심 종목 편입
# ======================================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

TARGET_STOCKS = {
    # [Tech Core]
    'GOOGL': 'Google Alphabet',
    'MSFT': 'Microsoft',
    'NVDA': 'Nvidia',
    'PLTR': 'Palantir',
    # [AI Power & Infra - The New Alpha]
    'NEE': 'NextEra Energy',   # 재생에너지 대장
    'CEG': 'Constellation En', # 원자력 대장
    'ETN': 'Eaton Corp',       # 전력망/변압기
    'XLU': 'Utilities ETF'     # 유틸리티 섹터 지표
}

# 종목별 민감도 설정 (유틸리티는 Tech보다 변동성 허용폭을 좁게 설정)
STOCK_PARAMS = {
    'GOOGL': {'crash': 40, 'rel': 20, 'tech': 20, 'sell': 60},
    'MSFT':  {'crash': 30, 'rel': 10, 'tech': 20, 'sell': 60},
    'NVDA':  {'crash': 40, 'rel': 10, 'tech': 20, 'sell': 60},
    'PLTR':  {'crash': 40, 'rel': 15, 'tech': 20, 'sell': 60},
    # [Defensive Growth] 방어주 성격이 섞인 종목들
    'NEE':   {'crash': 25, 'rel': 15, 'tech': 20, 'sell': 55},
    'CEG':   {'crash': 30, 'rel': 20, 'tech': 20, 'sell': 60},
    'ETN':   {'crash': 30, 'rel': 20, 'tech': 20, 'sell': 60},
    'XLU':   {'crash': 20, 'rel': 10, 'tech': 10, 'sell': 50}
}

W_TREND_MACRO = 35 
W_VOL_MACRO = 20
W_MACRO_MACRO = 10 
TH_SELL = 60
TH_BUY = 30
# ======================================================

class MarketStrategyBot:
    def __init__(self):
        print("🏛️ [Wall St. Strategist Bot v3.0] 가동 중... (Sector Rotation Mode)")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"⚠️ AI 모델 로드 실패: {e}")
            self.nlp = None
        
        # 키워드 확장: 에너지 및 인프라 관련 키워드 추가
        self.macro_keywords = [
            'Federal Reserve', 'Powell', 'US CPI', 'Recession', 
            'AI Bubble', 'Data Center Energy', 'Power Grid Shortage', 'Nuclear Energy'
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
                        
                        # BERT 모델 길이 제한 처리
                        inputs = clean_title[:512]
                        res = self.nlp(inputs)[0]
                        score = res['score'] if res['label'] == 'positive' else -res['score']
                        
                        total_score += score
                        count += 1
                        
                        # 가장 부정적인 뉴스 포착
                        if score < worst_info["score"]:
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
            # QQQ(기술주)와 XLU(유틸리티)를 명시적으로 호출하여 로테이션 분석에 사용
            macro_tickers = ['NQ=F', 'QQQ', '^VIX', '^VIX3M', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', '^IRX', 'BTC-USD']
            all_tickers = macro_tickers + list(TARGET_STOCKS.keys())
            
            # 중복 제거
            all_tickers = list(set(all_tickers))
            
            data = yf.download(all_tickers, period='1y', interval='1d', prepost=True, progress=False, ignore_tz=True)
            
            if isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns:
                dfs = {}
                df_macro = pd.DataFrame()
                close_data = data['Close']
                
                # 매크로 기본 데이터
                if 'NQ=F' in close_data.columns:
                    df_macro['Close'] = close_data['NQ=F']
                    df_macro['High'] = data['High']['NQ=F']
                    df_macro['Low'] = data['Low']['NQ=F']
                else:
                    # 선물이 없으면 QQQ로 대체
                    df_macro['Close'] = close_data['QQQ']
                    df_macro['High'] = data['High']['QQQ']
                    df_macro['Low'] = data['Low']['QQQ']

                ticker_map = {'^VIX': 'VIX', '^VIX3M': 'VIX3M', 'DX-Y.NYB': 'DXY', 'SOXX': 'SOXX', 'HYG': 'HYG', '^TNX': 'TNX', '^IRX': 'IRX', 'BTC-USD': 'BTC', 'QQQ': 'QQQ'}
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
                        dfs[ticker] = df_stock.dropna()
                
                # 유틸리티 데이터가 개별 종목으로 없어도 ETF(XLU) 데이터는 dfs에 저장
                if 'XLU' in close_data.columns:
                    dfs['XLU_DATA'] = pd.DataFrame({'Close': close_data['XLU']})

                return dfs
            return {}
        except Exception as e:
            print(f"Data Fetch Error: {e}")
            return {}

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
            reasons.append(f"📰 악재: {wn[:15]}...")

        return {"ticker": ticker, "price": curr, "change": chg, "score": min(score, 100), "threshold": params['sell'], "reasons": reasons}

    def analyze_market_flow(self):
        dfs = self.get_market_data()
        if not dfs or 'MACRO' not in dfs: return
        df = dfs['MACRO']
        
        now = datetime.now() + timedelta(hours=9)
        
        # [주말 브리핑 로직 생략 - 평일 로직 강화]
        
        curr = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        chg = (curr - prev) / prev * 100
        
        high_52w = df['Close'].rolling(252).max().iloc[-1]
        drawdown = (curr - high_52w) / high_52w * 100
        
        # 지표 계산
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        ma120 = df['Close'].rolling(120).mean().iloc[-1]
        vix = df['VIX'].iloc[-1]
        vix3m = df['VIX3M'].iloc[-1] if 'VIX3M' in df.columns else vix * 1.1
        
        danger_score = 0
        reasons = []
        
        # ==========================================================
        # 1. [NEW STRATEGY] 섹터 로테이션 (Sector Rotation) 감지
        # ==========================================================
        try:
            qqq_curr = df['QQQ'].iloc[-1] if 'QQQ' in df.columns else df['Close'].iloc[-1]
            qqq_prev = df['QQQ'].iloc[-2] if 'QQQ' in df.columns else df['Close'].iloc[-2]
            qqq_chg = (qqq_curr - qqq_prev) / qqq_prev * 100

            if 'XLU_DATA' in dfs:
                xlu_curr = dfs['XLU_DATA']['Close'].iloc[-1]
                xlu_prev = dfs['XLU_DATA']['Close'].iloc[-2]
                xlu_chg = (xlu_curr - xlu_prev) / xlu_prev * 100
            else:
                xlu_chg = 0

            # 로테이션 정의: 기술주 하락(-0.5% 이하) & 유틸리티 상승(+0.3% 이상)
            is_rotation = (qqq_chg < -0.5) and (xlu_chg > 0.3)
            
            # 시스템 붕괴 정의: 기술주 폭락 & 유틸리티 동반 폭락 (피난처 없음)
            is_system_crash = (qqq_chg < -2.0) and (xlu_chg < -1.0)
            
        except:
            is_rotation = False
            is_system_crash = False
            xlu_chg = 0

        # ==========================================================
        # 2. 위험 점수 계산 (Scoring Logic)
        # ==========================================================
        
        # A. 기본 추세
        if chg < -1.5: danger_score += W_TREND_MACRO; reasons.append(f"📉 지수 급락 ({chg:.2f}%)")
        if drawdown < -20: danger_score += 30; reasons.append(f"📉 베어마켓 (MDD {drawdown:.1f}%)")
        
        # B. 공포지수 (VIX)
        if vix > vix3m * 1.02: danger_score += 35; reasons.append(f"🚨 VIX 역전 (변동성 폭발)")
        elif vix > 30: danger_score += 20; reasons.append(f"😱 공포 구간 ({vix:.1f})")
            
        # C. 매크로 유동성 (Liquidity)
        dxy_chg = (df['DXY'].iloc[-1] - df['DXY'].iloc[-2]) / df['DXY'].iloc[-2] * 100
        # [강화] 0.5 -> 0.4로 민감도 상향 (킹달러 경계)
        if dxy_chg > 0.4: danger_score += 15; reasons.append("💵 달러 급등 (유동성 축소)")
        
        spread = df['TNX'].iloc[-1] - df['IRX'].iloc[-1]
        if spread < -0.5: danger_score += 10; reasons.append("⚠️ 장단기 금리 역전")
        
        # D. 뉴스 심리
        news_score, w_title, w_link, w_src, w_sum = self.get_news_sentiment(self.macro_keywords)
        if news_score < -0.3: danger_score += 15; reasons.append(f"📰 거시경제 심리 악화")

        # E. 로테이션 반영 (전략적 가감)
        if is_rotation:
            danger_score -= 15 # 건전한 조정으로 판단하여 점수 차감
            reasons.append(f"🔄 섹터 로테이션 (Tech▼ Power▲)")
        
        if is_system_crash:
            danger_score += 25 # 피할 곳 없는 하락
            reasons.append(f"🆘 시스템 붕괴 (Tech & Util 동반 투매)")

        # 경기침체 데이터 확인
        fund = self.get_fundamental_data()
        if fund and fund['is_recession']: danger_score += 30; reasons.append("🛑 경기 침체 시그널")

        danger_score = max(0, min(danger_score, 100))
        
        # ==========================================================
        # 3. 결과 리포팅 (Reporting)
        # ==========================================================
        status = "🟢 안정"
        if danger_score >= TH_SELL: status = "🔴 위험 (현금확보)"
        elif danger_score >= TH_BUY: status = "🟡 주의 (방어주 이동)"
        
        stock_results = []
        for t in TARGET_STOCKS:
            if t in dfs:
                res = self.analyze_individual(t, dfs[t], df)
                if res: stock_results.append(res)
        stock_results.sort(key=lambda x: x['score'], reverse=True)

        # 이모지 세팅
        trend_st = "상승✅" if curr > ma120 else "하락⚠️"
        xlu_emoji = "🛡️강세" if xlu_chg > 0.5 else "약세"
        
        msg = f"🏛️ *Wall St. Strategist (v3.0)*\n"
        msg += f"📅 {now.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 마켓 국면: {status} ({danger_score}점)\n\n"
        
        msg += "*1️⃣ Market Flow (유동성)*\n"
        if reasons: msg += "\n".join(["▪ " + r for r in reasons])
        else: msg += "▪ 특이사항 없음 (Goldilocks)"
        
        msg += f"\n\n*2️⃣ Sector Dashboard*\n"
        msg += f"• Nasdaq(Tech): {curr:,.0f} ({chg:+.2f}%)\n"
        msg += f"• Utilities(Power): {dfs['XLU_DATA']['Close'].iloc[-1]:.2f} ({xlu_chg:+.2f}%) {xlu_emoji}\n"
        msg += f"• VIX Term: {'정상✅' if vix < vix3m else '역전🚨'}\n"
        msg += f"• Dollar Index: {df['DXY'].iloc[-1]:.2f} ({dxy_chg:+.2f}%)\n"
        
        if w_title:
            cl_title = re.sub(r'[\[\]\*\_]', '', w_title)[:25] + "..."
            msg += f"\n*3️⃣ Smart Money News*\n• 심리: {news_score:.2f}\n• 헤드라인: [{w_src}] {cl_title}\n"
            
        msg += "\n*📊 Alpha Portfolio Watch*\n"
        for s in stock_results:
            icon = "🔴" if s['score'] >= s['threshold'] else "🟡" if s['score'] >= 40 else "🟢"
            # 종목 옆에 섹터 힌트 표시
            sec_hint = "⚡" if s['ticker'] in ['NEE', 'CEG', 'ETN', 'XLU'] else "💻"
            msg += f"{icon} {s['ticker']}{sec_hint}: {s['score']}점 ({s['change']:+.1f}%)\n"
            if s['reasons']: msg += f"  └ {', '.join(s['reasons'])}\n"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = MarketStrategyBot()
    bot.analyze_market_flow()
