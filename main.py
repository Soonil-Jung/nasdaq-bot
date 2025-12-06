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

# 최적화 파라미터
STOCK_PARAMS = {
    'GOOGL': {'crash': 40, 'rel': 20, 'tech': 20, 'sell': 60},
    'MSFT':  {'crash': 30, 'rel': 10, 'tech': 20, 'sell': 60},
    'TSLA':  {'crash': 40, 'rel': 10, 'tech': 20, 'sell': 60},
    'NVDA':  {'crash': 40, 'rel': 10, 'tech': 20, 'sell': 60},
    'AMD':   {'crash': 30, 'rel': 20, 'tech': 10, 'sell': 60},
    'PLTR':  {'crash': 40, 'rel': 15, 'tech': 20, 'sell': 60},
    'AAPL':  {'crash': 20, 'rel': 20, 'tech': 20, 'sell': 60}
}

W_TREND_MACRO = 30
W_VOL_MACRO = 15
W_MACRO_MACRO = 10
TH_SELL = 80
TH_BUY = 40
# ======================================================

class DangerAlertBot:
    def __init__(self):
        print("🤖 AI 퀀트 시스템(Pro Ver.) 가동 중...")
        try:
            # FinBERT: 금융 특화 감성 분석 모델
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"⚠️ AI 모델 로드 실패: {e}")
            self.nlp = None
        
        self.macro_keywords = [
            'Federal Reserve', 'The Fed', 'Jerome Powell', 'FOMC', 
            'CPI Inflation', 'Recession', 'Stagflation', 'US Economy',
            'Geopolitical tension', 'Market Crash', 'Liquidity crisis'
        ]

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ 텔레그램 토큰 없음")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
        try: requests.post(url, data=data, timeout=10)
        except Exception as e: print(f"텔레그램 전송 실패: {e}")

    # ------------------------------------------------------------------
    # [뉴스 분석 엔진] 비동기 RSS 피드 + FinBERT (속도/신뢰성 개선)
    # ------------------------------------------------------------------
    async def fetch_feed(self, session, keyword):
        """구글 뉴스 RSS 비동기 요청"""
        url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    return keyword, xml_data
        except: pass
        return keyword, None

    async def process_news_async(self, keywords):
        """뉴스 데이터 병렬 처리 및 AI 감성 분석"""
        if not self.nlp: return 0, "", "", ""
        
        search_list = [keywords] if isinstance(keywords, str) else keywords
        total_score = 0
        count = 0
        worst_info = {"score": 1.0, "title": "", "link": "", "source": ""}

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_feed(session, key) for key in search_list]
            feeds = await asyncio.gather(*tasks)

            for key, xml_data in feeds:
                if not xml_data: continue
                feed = feedparser.parse(xml_data)
                
                # 키워드 당 최신 3개 기사만 분석 (속도 최적화)
                for entry in feed.entries[:3]:
                    try:
                        title = entry.title
                        link = entry.link
                        source = entry.source.title if 'source' in entry else "News"
                        
                        # HTML 태그 제거
                        clean_title = BeautifulSoup(title, "html.parser").get_text()
                        # AI 분석 (최대 512토큰)
                        res = self.nlp(clean_title[:512])[0]
                        
                        score = 0
                        if res['label'] == 'positive': score = res['score']
                        elif res['label'] == 'negative': score = -res['score']
                        
                        total_score += score
                        count += 1
                        
                        # 가장 부정적인 뉴스 추적
                        if score < worst_info["score"] and score < -0.5:
                            worst_info = {
                                "score": score,
                                "title": clean_title,
                                "link": link,
                                "source": source
                            }
                    except: continue
        
        avg_score = total_score / count if count > 0 else 0
        return avg_score, worst_info["title"], worst_info["link"], worst_info["source"]

    def get_news_sentiment(self, target_keywords):
        """동기 래퍼 함수"""
        try:
            return asyncio.run(self.process_news_async(target_keywords))
        except Exception as e:
            print(f"뉴스 분석 오류: {e}")
            return 0, "", "", ""

    # ------------------------------------------------------------------
    # [데이터 수집] 매크로 & 개별 종목
    # ------------------------------------------------------------------
    def get_realtime_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.fast_info.get('last_price', None)
        except: return None

    def get_market_data(self):
        """주요 지표 일괄 다운로드"""
        try:
            # ^VIX3M: 3개월 VIX (구조 분석용)
            # ^MMTW: 나스닥 종목 중 20일 이평선 상회 비율 (시장 너비) - 티커 확인 필요 (대안: ^NYA200R 등)
            # 여기서는 안정적인 데이터 확보를 위해 VIX3M 추가
            macro_tickers = ['NQ=F', 'QQQ', '^VIX', '^VIX3M', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', '^IRX', 'BTC-USD']
            all_tickers = macro_tickers + list(TARGET_STOCKS.keys())
            
            data = yf.download(all_tickers, period='1mo', interval='1h', prepost=True, progress=False, ignore_tz=True)
            
            # 멀티인덱스 처리
            if isinstance(data.columns, pd.MultiIndex):
                dfs = {}
                df_macro = pd.DataFrame()
                
                # Close 데이터 추출
                if 'Close' in data.columns:
                    close_data = data['Close']
                    # 필수 데이터 체크
                    if 'NQ=F' not in close_data.columns: return {}
                    
                    df_macro['Close'] = close_data['NQ=F']
                    df_macro['High'] = data['High']['NQ=F'] if 'High' in data.columns else close_data['NQ=F']
                    df_macro['Low'] = data['Low']['NQ=F'] if 'Low' in data.columns else close_data['NQ=F']
                    df_macro['Volume'] = data['Volume']['QQQ'] if 'Volume' in data.columns and 'QQQ' in data['Volume'].columns else 0
                    
                    # 매크로 지표 매핑
                    ticker_map = {
                        '^VIX': 'VIX', '^VIX3M': 'VIX3M', 'DX-Y.NYB': 'DXY', 
                        'SOXX': 'SOXX', 'HYG': 'HYG', '^TNX': 'TNX', '^IRX': 'IRX', 'BTC-USD': 'BTC'
                    }
                    for t, col in ticker_map.items():
                        if t in close_data.columns:
                            df_macro[col] = close_data[t]
                    
                    df_macro = df_macro.ffill().bfill()
                    dfs['MACRO'] = df_macro
                    
                    # 개별 종목 데이터 추출
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
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return {}

    def get_fundamental_data(self):
        try:
            start_date = datetime.now() - timedelta(days=700)
            unrate = web.DataReader('UNRATE', 'fred', start_date)
            # Sahm Rule 계산
            unrate['MA3'] = unrate['UNRATE'].rolling(window=3).mean()
            curr_ma3 = unrate['MA3'].iloc[-1]
            low_12m = unrate['UNRATE'].iloc[-14:-1].min()
            sahm_score = curr_ma3 - low_12m
            return {"unrate": unrate['UNRATE'].iloc[-1], "sahm_score": sahm_score, "is_recession": sahm_score >= 0.50}
        except: return None

    # ------------------------------------------------------------------
    # [분석 엔진] 위험 감지 로직 (Logic Improved)
    # ------------------------------------------------------------------
    def analyze_individual(self, ticker, df_stock, df_macro):
        if df_stock.empty or len(df_stock) < 30: return None

        params = STOCK_PARAMS.get(ticker, {'crash': 30, 'rel': 15, 'tech': 15, 'sell': 60})
        
        current_price = df_stock['Close'].iloc[-1]
        prev_close = df_stock['Close'].iloc[-8] # 약 하루 전 (1시간봉 기준 8개)
        daily_pct = (current_price - prev_close) / prev_close * 100

        # 일목균형표
        ichimoku = IchimokuIndicator(high=df_stock['High'], low=df_stock['Low'], window1=9, window2=26, window3=52)
        cloud_bottom = min(ichimoku.ichimoku_a().iloc[-26], ichimoku.ichimoku_b().iloc[-26])
        
        # 이평선
        ma20 = SMAIndicator(close=df_stock['Close'], window=20).sma_indicator().iloc[-1]
        ma50 = SMAIndicator(close=df_stock['Close'], window=50).sma_indicator().iloc[-1]
        
        # RSI & 거래량
        rsi = RSIIndicator(close=df_stock['Close'], window=14).rsi().iloc[-1]
        vol_ma = df_stock['Volume'].rolling(window=20).mean().iloc[-1]
        vol_ratio = df_stock['Volume'].iloc[-1] / vol_ma if vol_ma > 0 else 0

        # 상대강도
        nq_chg = (df_macro['Close'].iloc[-1] - df_macro['Close'].iloc[-8]) / df_macro['Close'].iloc[-8] * 100
        rel_strength = daily_pct - nq_chg

        # 뉴스 분석
        news_score, worst_n, worst_l, worst_s = self.get_news_sentiment(ticker)

        # 점수 계산
        score = 0
        reasons = []

        if daily_pct < -3.0: score += params['crash']; reasons.append(f"📉 폭락 ({daily_pct:.1f}%)")
        if rel_strength < -1.5: score += params['rel']; reasons.append("상대적 약세")
        
        tech_bad = []
        if current_price < cloud_bottom: tech_bad.append("구름대 이탈")
        if ma20 < ma50 and current_price < ma20: tech_bad.append("데드크로스")
        if rsi < 30: tech_bad.append("과매도")
        if vol_ratio > 2.0 and daily_pct < 0: tech_bad.append("투매 거래량")
        
        if tech_bad:
            score += params['tech']
            reasons.append(f"기술적({','.join(tech_bad)})")
            
        if news_score < -0.3:
            score += 20
            src = f"[{worst_s}]" if worst_s else ""
            reasons.append(f"📰 악재: {src} {worst_n[:15]}...")

        score = max(0, min(score, 100))
        return {"ticker": ticker, "price": current_price, "change": daily_pct, "score": score, "threshold": params['sell'], "reasons": reasons}

    def analyze_danger(self):
        dfs = self.get_market_data()
        if not dfs or 'MACRO' not in dfs: 
            print("데이터 수집 실패")
            return
        df = dfs['MACRO']
        
        now = datetime.now() + timedelta(hours=9) # KST
        
        # 기본 데이터
        curr_close = df['Close'].iloc[-1]
        idx_day = -24 if len(df) >= 24 else 0
        daily_chg = (curr_close - df['Close'].iloc[idx_day]) / df['Close'].iloc[idx_day] * 100
        
        # [신규] 고급 매크로 분석
        vix = df['VIX'].iloc[-1]
        vix3m = df['VIX3M'].iloc[-1] if 'VIX3M' in df.columns else vix * 1.1
        # VIX 백워데이션 (단기 공포 > 중기 공포) = 폭락 전조
        is_backwardation = vix > (vix3m * 1.02)
        vix_ratio = vix / vix3m
        
        # 뉴스 분석
        news_score, w_title, w_link, w_src = self.get_news_sentiment(self.macro_keywords)
        
        # 위험 점수 계산
        danger_score = 0
        reasons = []
        
        # 1. 추세
        if daily_chg < -1.5: danger_score += W_TREND_MACRO; reasons.append(f"📉 지수 급락 ({daily_chg:.2f}%)")
        
        # 2. VIX 구조 (핵심 업그레이드)
        if is_backwardation:
            danger_score += 25
            reasons.append(f"🚨 VIX 백워데이션 (공포확산 {vix_ratio:.2f}배)")
        elif vix > 30:
            danger_score += 15
            reasons.append(f"😱 공포지수 위험권 ({vix:.1f})")
            
        # 3. 매크로
        dxy_chg = (df['DXY'].iloc[-1] - df['DXY'].iloc[idx_day]) / df['DXY'].iloc[idx_day] * 100
        if dxy_chg > 0.3: danger_score += W_MACRO_MACRO; reasons.append("💵 달러 급등")
        
        tnx = df['TNX'].iloc[-1]
        irx = df['IRX'].iloc[-1]
        spread = tnx - irx
        if spread < -0.5: danger_score += 10; reasons.append("⚠️ 장단기 금리차 역전 심화")
        
        # 4. 심리/뉴스
        if news_score < -0.25: 
            danger_score += W_VOL_MACRO
            reasons.append(f"📰 뉴스 심리 악화 ({news_score:.2f})")
            
        # 5. 기술적 (이평선)
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        if ma20 < ma50 and curr_close < ma20:
            danger_score += 15
            reasons.append("📉 완전 역배열 진입")

        # 샴의 법칙
        fund = self.get_fundamental_data()
        if fund and fund['is_recession']:
            danger_score += 30
            reasons.append(f"🛑 경기 침체 신호 (샴의 법칙)")

        danger_score = max(0, min(danger_score, 100))
        
        # 상태 결정
        status = "🟢 안정"
        if danger_score >= TH_SELL: status = "🔴 위험 (현금화 권장)"
        elif danger_score >= TH_BUY: status = "🟡 주의 (관망)"
        
        # 개별 종목 분석
        stock_results = []
        for t in TARGET_STOCKS:
            if t in dfs:
                res = self.analyze_individual(t, dfs[t], df)
                if res: stock_results.append(res)
        stock_results.sort(key=lambda x: x['score'], reverse=True)

        # 메시지 작성
        msg = f"🔔 *AI 마켓 워치 (Advanced)*\n📅 {now.strftime('%Y-%m-%d %H:%M')} (KST)\n🚦 시장상태: {status} ({danger_score}점)\n\n"
        
        msg += "*1️⃣ 핵심 위험 요인*\n"
        if reasons: msg += "\n".join(["▪ " + r for r in reasons])
        else: msg += "▪ 특이사항 없음 (양호)"
        
        msg += f"\n\n*2️⃣ 매크로 대시보드*\n• 나스닥: {curr_close:,.0f} ({daily_chg:+.2f}%)\n• VIX구조: {'⚠️ 역전' if is_backwardation else '✅ 정상'} ({vix:.1f}/{vix3m:.1f})\n• 달러: {df['DXY'].iloc[-1]:.2f}\n• 금리차: {spread:.2f}p\n"
        
        if fund: msg += f"• 실업률: {fund['unrate']}%\n"
        
        if w_title:
            cl_title = re.sub(r'[\[\]\*\_]', '', w_title)[:25] + "..."
            src_tag = f"[{w_src}]" if w_src else "[News]"
            msg += f"\n*3️⃣ 주요 뉴스 심리*\n• 점수: {news_score:.2f}\n• 이슈: {src_tag} [{cl_title}]({w_link})\n"
            
        msg += "\n*📊 관심 종목 위험도*\n"
        for s in stock_results:
            icon = "🔴" if s['score'] >= s['threshold'] else "🟡" if s['score'] >= 40 else "🟢"
            msg += f"{icon} {s['ticker']}: {s['score']}점 ({s['change']:+.1f}%)\n"
            if s['reasons']: msg += f"  └ {', '.join(s['reasons'])}\n"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
