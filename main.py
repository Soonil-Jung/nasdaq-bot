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
        print("🤖 AI 시스템(Link-Support Ver) 가동 중...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except: pass
        self.keywords = ['Jerome Powell', 'Donald Trump', 'Fed Rate', 'Recession', 'Nasdaq']

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        # disable_web_page_preview=True: 메시지 내 링크 미리보기 이미지가 너무 크게 뜨는 것 방지
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
        try: requests.post(url, data=data)
        except: pass

    def get_news_sentiment(self):
        # [수정] 점수, 제목, 그리고 '링크'까지 반환
        try:
            googlenews = GoogleNews(lang='en', period='1d')
            total_score = 0
            count = 0
            
            worst_news_title = ""
            worst_news_link = ""
            min_score = 1.0 

            for keyword in self.keywords:
                googlenews.clear()
                googlenews.search(keyword)
                results = googlenews.results(sort=True)
                if not results: continue
                
                for item in results[:2]:
                    try:
                        title = item['title']
                        link = item['link'] # 기사 링크 가져오기
                        
                        res = self.nlp(title[:512])[0]
                        score = res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0
                        
                        total_score += score
                        count += 1
                        
                        # 가장 부정적인 뉴스 기록
                        if score < min_score and score < -0.5:
                            min_score = score
                            worst_news_title = f"[{keyword}] {title}"
                            worst_news_link = link
                            
                    except: continue
            
            avg_score = total_score / count if count > 0 else 0
            return avg_score, worst_news_title, worst_news_link
            
        except: return 0, "", ""

    def get_market_data(self):
        try:
            tickers = ['NQ=F', 'QQQ', '^VIX', 'DX-Y.NYB', 'SOXX', 'HYG', '^TNX', 'BTC-USD', '^IRX']
            # 1분봉 제거 -> 다시 1시간봉(1h)으로 복귀 (가볍고 안정적)
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
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        rsi_val = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        
        # [수정] 1분봉 다운로드 로직 제거하고, 가벼운 호가 조회(fast_info)만 유지
        try:
            ticker_nq = yf.Ticker("NQ=F")
            realtime_price = ticker_nq.fast_info.get('last_price')
            current_close = realtime_price if (realtime_price and not np.isnan(realtime_price)) else df['Close'].iloc[-1]
        except:
            current_close = df['Close'].iloc[-1]

        # [A] 가격 변동성
        daily_chg = (current_close - df['Close'].iloc[-24]) / df['Close'].iloc[-24] * 100 
        hourly_chg = (current_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100  
        
        # [B] 거래량
        avg_vol = df['Vol_MA20'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        vol_ratio = 0 if avg_vol == 0 else current_vol / avg_vol
        
        # [C] 매크로
        current_dxy = df['DXY'].iloc[-1]
        dxy_chg = (current_dxy - df['DXY'].iloc[-24]) / df['DXY'].iloc[-24] * 100
        
        current_tnx = df['TNX'].iloc[-1]
        current_irx = df['IRX'].iloc[-1]
        yield_spread = current_tnx - current_irx
        irx_chg = (current_irx - df['IRX'].iloc[-24]) / df['IRX'].iloc[-24] * 100
        
        # [D] 리스크 자산
        current_btc = df['BTC'].iloc[-1]
        btc_chg = (current_btc - df['BTC'].iloc[-24]) / df['BTC'].iloc[-24] * 100
        
        nq_ret = current_close / df['Close'].iloc[-5] - 1
        soxx_ret = df['SOXX'].iloc[-1] / df['SOXX'].iloc[-5] - 1
        semi_weakness = nq_ret - soxx_ret 

        hyg_high = df['HYG'].max()
        current_hyg = df['HYG'].iloc[-1]
        hyg_drawdown = (current_hyg - hyg_high) / hyg_high * 100

        # ★ 뉴스 정보 가져오기 (링크 포함)
        news_score, worst_news_title, worst_news_link = self.get_news_sentiment()

        # --- 2. 위험 점수 산출 ---
        danger_score = 0
        reasons = []

        # [A] 가격 추세
        if daily_chg < -1.5:
            danger_score += 20
            reasons.append(f"📉 *추세 하락*: 24시간 동안 **{daily_chg:.2f}%** 하락했습니다.")
        if hourly_chg < -0.8:
            danger_score += 15
            reasons.append(f"⚡ *투매 발생*: 1시간 만에 **{hourly_chg:.2f}%** 급락했습니다.")

        # [B] 구름대
        cloud_str = "구름대 위 (안정)"
        if current_close < span_a:
            danger_score += 20
            reasons.append("☁️ *지지선 붕괴*: 일목균형표 구름대 하단을 이탈했습니다.")
            cloud_str = "하단 이탈 🚨"
        elif current_close < span_b: 
            cloud_str = "구름대 내부 (혼조)"

        # [C] 거래량
        if vol_ratio > 1.5:
            danger_score += 15
            reasons.append(f"📢 *패닉 셀링*: 거래량이 평소의 **{vol_ratio:.1f}배**로 폭발했습니다.")

        # [D] 매크로
        if dxy_chg > 0.3:
            danger_score += 10
            reasons.append(f"💵 *달러 강세*: 달러 인덱스가 **+{dxy_chg:.2f}%** 급등했습니다.")
        if irx_chg > 2.0:
            danger_score += 10
            reasons.append(f"🏦 *긴축 공포*: 단기 금리가 **+{irx_chg:.1f}%** 치솟았습니다.")

        # [E] 리스크 자산
        if btc_chg < -3.0: 
            danger_score += 15
            reasons.append(f"📉 *코인 급락*: 위험자산 회피로 비트코인이 **{btc_chg:.2f}%** 하락했습니다.")
        if semi_weakness > 0.005:
            danger_score += 10
            reasons.append(f"📉 *주도주 균열*: 반도체 섹터가 나스닥보다 약세입니다.")
        if hyg_drawdown < -0.3:
            danger_score += 15
            reasons.append(f"💸 *자금 이탈*: 스마트머니(하이일드)가 **{hyg_drawdown:.2f}%** 빠져나갔습니다.")

        # [F] 심리 & 뉴스
        current_vix = df['VIX'].iloc[-1]
        vix_trend = current_vix - df['VIX'].rolling(window=5).mean().iloc[-1]
        if vix_trend > 0.5:
            danger_score += 10
            reasons.append(f"😱 *공포 확산*: 변동성 지수(VIX)가 상승 추세입니다.")
        
        # ★ 뉴스 악재 발생 시 링크 제공
        if news_score < -0.2:
            danger_score += 10
            news_msg = f"📰 *뉴스 심리 악화*: AI 점수 **{news_score:.2f}** (부정)"
            if worst_news_title:
                # 텔레그램 마크다운 링크 형식: [텍스트](URL)
                # 링크가 너무 길 수 있으니 '기사 원문 보기'로 표시
                news_msg += f"\n  └ 원인: {worst_news_title}"
                if worst_news_link:
                    news_msg += f"\n  └ 🔗 [기사 원문 보기]({worst_news_link})"
            reasons.append(news_msg)

        danger_score = min(danger_score, 100)

        # --- 3. 메시지 작성 ---
        status_emoji = '🔴 위험 (매도)' if danger_score >= 60 else '🟡 주의 (관망)' if danger_score >= 35 else '🟢 안정 (매수)'
        spread_str = "정상 ✅" if yield_spread >= 0 else "역전(침체) ⚠️"
        semi_str = "약세 ⚠️" if semi_weakness > 0.005 else "양호 ✅"
        hyg_str = "이탈 ⚠️" if hyg_drawdown < -0.3 else "유입 ✅"
        vix_str = "확산 ↗" if vix_trend > 0 else "안정 ↘"
        
        now_kst = datetime.now() + timedelta(hours=9)
        
        msg = f"🔔 *AI 퀀트 시장 정밀 분석*\n"
        msg += f"📅 {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
        msg += f"🚦 종합상태: {status_emoji}\n"
        msg += f"🔥 위험점수: *{danger_score}점* / 100점\n\n"
        
        msg += "*1️⃣ 가격 & 거래량 (Technical)*\n"
        msg += f"• 나스닥 : {current_close:,.2f} (24h: {daily_chg:+.2f}%)\n"
        msg += f"• 1시간봉 : {hourly_chg:+.2f}% (단기변동)\n"
        msg += f"• 거래강도 : 평소의 {int(vol_ratio*100)}%\n"
        msg += f"• RSI(14) : {rsi_val:.1f}\n"
        msg += f"• 일목구름 : {cloud_str}\n\n"
        
        msg += "*2️⃣ 매크로 지표 (Macro)*\n"
        msg += f"• 달러(DXY): {current_dxy:.2f} ({dxy_chg:+.2f}%)\n"
        msg += f"• 3개월금리 : {current_irx:.2f}% ({irx_chg:+.1f}%)\n"
        msg += f"• 10년금리 : {current_tnx:.2f}% (시장금리)\n"
        msg += f"• 장단기차 : {yield_spread:.2f}p ({spread_str})\n\n"
        
        msg += "*3️⃣ 리스크 자산 (Risk Asset)*\n"
        msg += f"• 비트코인 : ${current_btc:,.0f} ({btc_chg:+.2f}%)\n"
        msg += f"• 반도체비 : {semi_str} (괴리: {semi_weakness*100:.1f}%)\n"
        msg += f"• 하이일드 : {hyg_str} (낙폭: {hyg_drawdown:.2f}%)\n\n"
        
        msg += "*4️⃣ 시장 심리 (Sentiment)*\n"
        msg += f"• 공포(VIX): {current_vix:.2f} (추세: {vix_str})\n"
        msg += f"• 뉴스점수 : {news_score:.2f} (-1~+1)\n"
        msg += "───────────────\n"
        
        msg += "*📋 [상세 위험 요인 분석]*\n"
        if reasons:
            msg += "\n".join(["🚨 " + r for r in reasons])
        else:
            msg += "✅ 특이사항 없음 (안정적)"

        self.send_telegram(msg)

if __name__ == "__main__":
    bot = DangerAlertBot()
    bot.analyze_danger()
