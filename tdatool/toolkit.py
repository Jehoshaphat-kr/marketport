import tdatool as tt
import pandas as pd
import numpy as np
import os, requests
from bs4 import BeautifulSoup as Soup
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter
from datetime import timedelta


class technical:

    @staticmethod
    def fetch(ticker:str, src:str) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 시계열 데이터프레임
        :param ticker: 종목코드
        :param src: 주가 데이터 소스
        :return:
        """
        __price__ = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        ) if src == 'online' else pd.read_csv(
            os.path.join(tt.root, f'warehouse/series/{ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        __price__.index = pd.to_datetime(__price__.index)
        return __price__

    @staticmethod
    def calc_filter(series:pd.Series, window:list) -> pd.DataFrame:
        """
        주가 가이드(필터) 데이터프레임
        :param series: 필터 기준 주가(시가, 고가, 저가, 종가)
        :param window: 필터 대상 거래일
        :return:
        """
        # FIR: SMA
        objs = {f'SMA{win}D': series.rolling(window=win).mean() for win in window}

        # FIR: EMA
        objs.update({f'EMA{win}D': series.ewm(span=win).mean() for win in window})
        for win in window:
            # IIR: BUTTERWORTH
            cutoff = (252 / win) / (252 / 2)
            coeff_a, coeff_b = butter(N=1, Wn=cutoff, btype='lowpass', analog=False, output='ba')
            objs[f'IIR{win}D'] = pd.Series(data=filtfilt(coeff_a, coeff_b, series), index=series.index)

            # FIR: KAISER
            N, beta = kaiserord(ripple={5:10, 10:12, 20:20, 60:60, 120:80}[win], width=75 / (252 / 2))
            taps = firwin(N, cutoff, window=('kaiser', beta))
            objs[f'FIR{win}D'] = pd.Series(data=lfilter(taps, 1.0, series), index=series.index)
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def calc_trend(dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        주가 필터 기반 추세 데이터프레임
        :param dataframe: 주가 가이드 데이터프레임
        :return:
        """
        combination = [
            ['중장기IIR', 'IIR60D', 'EMA120D'], ['중기IIR', 'IIR60D', 'EMA60D'], ['중단기IIR', 'IIR20D', 'EMA60D'],
            ['중장기FIR', 'FIR60D', 'EMA120D'], ['중기FIR', 'FIR60D', 'EMA60D'], ['중단기FIR', 'FIR20D', 'EMA60D'],
            ['중장기SMA', 'SMA60D', 'SMA120D'], ['중단기SMA', 'SMA20D', 'SMA60D'],
            ['중장기EMA', 'EMA60D', 'EMA120D'], ['중단기EMA', 'EMA20D', 'EMA60D']
        ]
        objs = {}
        for label, numerator, denominator in combination:
            basis = dataframe[numerator] - dataframe[denominator]
            objs[label] = basis
            objs[f'd{label}'] = basis.diff()
            objs[f'd2{label}'] = basis.diff().diff()
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def calc_macd(series:pd.Series) -> pd.DataFrame:
        """
        MACD 데이터프레임
        :param series:
        :return:
        """
        macd = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return pd.concat(objs={'MACD': macd, 'MACD-Sig': signal, 'MACD-Hist': hist}, axis=1)

    @staticmethod
    def calc_trade_points(dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        주요 지표 매매 적합 판단지점
        :param dataframe: 주요 지표
        :return:
        """
        objs = {}
        cols = [col for col in dataframe if not col.startswith('d') and not 'Hist' in col and not 'Sig' in col]
        for col in cols:
            is_macd = True if col.startswith('MACD') else False
            data = []
            tr = dataframe['MACD' if is_macd else col].values
            sr = dataframe['MACD-Sig' if is_macd else f'd{col}'].values
            for n, date in enumerate(dataframe.index[1:]):
                if (is_macd and tr[n-1] < sr[n-1] and tr[n] > sr[n]) or (not is_macd and sr[n-1] < 0 < sr[n]):
                    data.append([date, tr[n], 'Buy', 'triangle-up', 'red'])
                elif (is_macd and tr[n-1] > sr[n-1] and tr[n] < sr[n]) or (not is_macd and sr[n-1] > 0 > sr[n]):
                    data.append([date, tr[n], 'Sell', 'triangle-down', 'blue'])
                elif not is_macd and tr[n - 1] < 0 < tr[n]:
                    data.append([date, tr[n], 'Golden-Cross', 'star', 'gold'])
                elif not is_macd and tr[n - 1] > 0 > tr[n]:
                    data.append([date, tr[n], 'Dead-Cross', 'x', 'black'])
            objs[f'det{col}'] = pd.DataFrame(data=data, columns=['날짜', 'value', 'bs', 'symbol', 'color']).set_index(keys='날짜')
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def calc_horizontal_line(dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        수평 지지/저항선 데이터프레임
        :param dataframe: 가격 데이터프레임
        :return:
        """
        frm = dataframe[dataframe.index >= (dataframe.index[-1] - timedelta(180))].copy()
        low = frm['저가']
        high = frm['고가']
        spread = (high - low).mean()

        def is_support(i):
            return low[i] < low[i - 1] < low[i - 2] and low[i] < low[i + 1] < low[i + 2]

        def is_resistance(i):
            return high[i] > high[i - 1] > high[i - 2] and high[i] > high[i + 1] > high[i + 2]

        def is_far_from_level(l, lines):
            return np.sum([abs(l - x) < spread for x in lines]) == 0

        levels = []
        data = []
        for n, date in enumerate(frm.index[2: len(frm) - 2]):
            if is_support(n) and is_far_from_level(l=low[n], lines=levels):
                sample = (n, low[n])
                levels.append(sample)
                data.append(list(sample) + list((date, f'지지선@{date.strftime("%Y%m%d")[2:]}')))
            elif is_resistance(n) and is_far_from_level(l=frm['고가'][n], lines=levels):
                sample = (n, high[n])
                levels.append(sample)
                data.append(list(sample) + list((date, f'저항선@{date.strftime("%Y%m%d")[2:]}')))
        return pd.DataFrame(data=data, columns=['ID', '가격', '날짜', '종류']).set_index(keys='날짜')


class fundamental:
    @staticmethod
    def fetch_info(ticker) -> str:
        """
        기업 소개
        :param ticker: 종목코드
        :return:
        """
        link = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701"

        html = requests.get(link % ticker).content
        soup = Soup(html, 'lxml')
        texts = soup.find('ul', id='bizSummaryContent').find_all('li')
        return '\n'.join([text.text.replace('&nbsp;', ' ') for text in texts])

    def fetch_statement(self, ticker) -> tuple:
        """
        1. 기업 개요
        연결 제무제표 또는 별도 제무제표 유효성 판정
        매출 추정치 존재 유무로 판정 (매출 추정치 존재 시 유효 판정)
        index 11 = 연간 연결 제무제표
        index 14 = 연간 별도 제무제표
        :param ticker: 종목코드
        :return:
        """
        link = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701"
        table = pd.read_html(link % ticker, encoding='utf-8')
        is_separate = table[11].iloc[0].isnull().sum() > table[14].iloc[0].isnull().sum()

        a = table[14] if is_separate else table[11]
        q = table[15] if is_separate else table[12]
        return self.reform_statement(df=a), self.reform_statement(df=q), self.reform_consensus(df=table[7])

    def fetch_summary(self, ticker) -> pd.DataFrame:
        """
        1. 기업 정보
        :param ticker: 종목코드
        :return: 
        """
        link = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"
        table = pd.read_html(link % ticker, encoding='utf-8')
        return pd.concat(objs=[
            self.reform_sga(df=table[6]),
            self.reform_sga(df=table[7]),
            self.reform_rnd(df=table[8]),
        ], axis=1).sort_index()

    def fetch_factor(self, ticker):
        link1 = "http://comp.fnguide.com/SVO2/common/chartListPopup2.asp?oid=div5_img&cid=05_05&gicode=A"
        link2 = "&filter=D&term=Y&etc=0&etc2=0&titleTxt=%EB%A9%80%ED%8B%B0%ED%8C%A9%ED%84%B0%20%EC%8A%A4%ED%83%80%EC%9D%BC%20%EB%B6%84%EC%84%9D&dateTxt=undefined&unitTxt="
        html = requests.get(link1 + ticker + link2)
        print(html.text)
        return

    @staticmethod
    def reform_statement(df:pd.DataFrame) -> pd.DataFrame:
        """
        기업 기본 재무정보
        :param df: 원 데이터프레임
        :return:
        """
        cols = df.columns.tolist()
        df.set_index(keys=[cols[0]], inplace=True)
        df.index.name = None
        df.columns = df.columns.droplevel()
        return df.T

    @staticmethod
    def reform_rnd(df:pd.DataFrame) -> pd.DataFrame:
        """
        R&D 투자 데이터프레임
        :param df:
        :return:
        """
        df.set_index(keys=['회계연도'], inplace=True)
        df.index.name = None
        df = df[['R&D 투자 총액 / 매출액 비중.1', '무형자산 처리 / 매출액 비중.1', '당기비용 처리 / 매출액 비중.1']]
        df = df.rename(columns={'R&D 투자 총액 / 매출액 비중.1': 'R&D투자비중',
                                  '무형자산 처리 / 매출액 비중.1': '무형자산처리비중',
                                  '당기비용 처리 / 매출액 비중.1': '당기비용처리비중'})
        return df

    @staticmethod
    def reform_sga(df:pd.DataFrame) -> pd.DataFrame:
        """
        SG & A: Sales, General, and Administrative(판관비) 또는 매출원가 데이터프레임
        :param df: 원 데이터프레임
        :return:
        """
        df.set_index(keys=['항목'], inplace=True)
        df.index.name = None
        return df.T

    @staticmethod
    def reform_consensus(df:pd.DataFrame) -> pd.DataFrame:
        """
        투자 컨센서스 데이터프레임
        :param df: 원 데이터프레임
        :return:
        """
        df['투자의견'] = df['투자의견'].astype(str)
        df['목표주가'] = df['목표주가'].apply(lambda x: "{:,}원".format(x))
        df['EPS'] = df['EPS'].apply(lambda x: "{:,}원".format(x))
        df['PER'] = df['PER'].astype(str)
        return df

if __name__ == "__main__":
    api = fundamental()
    # api.fetch_statement(ticker='000660')
    # api.fetch_info(ticker='000660')
    api.fetch_factor(ticker='000660')