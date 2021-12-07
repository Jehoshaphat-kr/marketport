import requests, json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as Soup
from pykrx import stock
from datetime import datetime, timedelta
from urllib.request import urlopen


today = datetime.today()
class fetch:

    def __init__(self, ticker:str):
        # Parameter
        self.ticker = ticker
        self.name = stock.get_market_ticker_name(ticker=ticker)
        self.url1 = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701"
        self.url2 = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"

        # Fetch CompanyGuide SnapShot
        self.obj1 = pd.read_html(self.url1 % ticker, encoding='utf-8')
        self.obj2 = pd.read_html(self.url2 % ticker, encoding='utf-8')
        self.is_separate = self.obj1[11].iloc[0].isnull().sum() > self.obj1[14].iloc[0].isnull().sum()

        # Initialize DataFrames
        self._marketcap_ = pd.DataFrame()
        self._foreigner_ = pd.DataFrame()
        self._consensus_ = pd.DataFrame()
        self._factors_ = pd.DataFrame()
        self._shorts_ = pd.DataFrame()
        self._product_ = pd.DataFrame()
        self._rnd_ = pd.DataFrame()
        return

    @property
    def summary(self) -> str:
        """
        사업 소개
        :return:
        """
        html = requests.get(self.url1 % self.ticker).content
        soup = Soup(html, 'lxml')
        texts = soup.find('ul', id='bizSummaryContent').find_all('li')
        text = '\n\n '.join([text.text for text in texts])
        return ' ' + text[0] + ''.join([
            '.\n' if text[n] == '.' and not text[n - 1].isdigit() else text[n] for n in range(1, len(text) - 1)
        ])

    @property
    def factors(self) -> pd.DataFrame:
        """
        멀티 팩터 데이터프레임
        :return:
        """
        if self._factors_.empty:
            url = f"http://cdn.fnguide.com/SVO2/json/chart/05_05/A{self.ticker}.json"
            data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
            header = pd.DataFrame(data['CHART_H'])['NAME'].tolist()
            self._factors_ = pd.DataFrame(data['CHART_D']).rename(
                columns=dict(zip(['NM', 'VAL1', 'VAL2'], ['팩터'] + header))
            ).set_index(keys='팩터')
        return self._factors_

    @property
    def foreigner(self) -> pd.DataFrame:
        """
        외국인 보유 비중 데이터프레임
        :return:
        """
        if self._foreigner_.empty:
            url = f"http://cdn.fnguide.com/SVO2/json/chart/01_01/chart_A{self.ticker}_1Y.json"
            data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
            self._foreigner_ = pd.DataFrame(data["CHART"])[['TRD_DT', 'J_PRC', 'FRG_RT']].rename(columns={
                'TRD_DT': '날짜', 'J_PRC': '종가', 'FRG_RT': '외국인보유비중'
            }).set_index(keys='날짜')
            self._foreigner_.index = pd.to_datetime(self._foreigner_.index)
        return self._foreigner_

    @property
    def consensus(self) -> pd.DataFrame:
        """
        컨센선스 Consensus 데이터프레임
        :return:
        """
        if self._consensus_.empty:
            url = f"http://cdn.fnguide.com/SVO2/json/chart/01_02/chart_A{self.ticker}.json"
            data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
            self._consensus_ = pd.DataFrame(data['CHART']).rename(columns={
                'TRD_DT': '날짜', 'VAL1': '투자의견', 'VAL2': '목표주가', 'VAL3': '종가'
            }).set_index(keys='날짜')
            self._consensus_.index = pd.to_datetime(self._consensus_.index)
            self._consensus_['목표주가'] = self._consensus_['목표주가'].apply(lambda x:x if x else np.nan)
        return self._consensus_

    @property
    def short(self) -> pd.DataFrame:
        """
        차입공매도 비중 데이터프레임
        :return:
        """
        if self._shorts_.empty:
            url = f"http://cdn.fnguide.com/SVO2/json/chart/11_01/chart_A{self.ticker}_SELL1Y.json"
            data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
            self._shorts_ = pd.DataFrame(data['CHART']).rename(columns={
                'TRD_DT':'날짜', 'VAL':'차입공매도비중', 'ADJ_PRC':'수정 종가'
            }).set_index(keys='날짜')
            self._shorts_.index = pd.to_datetime(self._shorts_.index)
        return self._shorts_

    @property
    def annual(self) -> pd.DataFrame:
        """
        연간 실적/비율/배수 데이터프레임
        :return:
        """
        return self.reform_statement(self.obj1[14] if self.is_separate else self.obj1[11])

    @property
    def quarter(self) -> pd.DataFrame:
        """
        분기 실적/비율/배수 데이터프레임
        :return:
        """
        return self.reform_statement(self.obj1[15] if self.is_separate else self.obj1[12])

    @property
    def product(self) -> pd.DataFrame:
        """
        주요 매출 상품:: [reform] 리폼 필요
        :return:
        """
        if self._product_.empty:
            df = self.obj2[2]
            df.set_index(keys='제품명', inplace=True)
            df = df[df.columns[-1]].dropna()
            df.drop(index=df[df < 0].index, inplace=True)
            df[df.index[-1]] += (100 - df.sum())
            df.name = '비중'
            self._product_ = df.copy()
        return self._product_

    @property
    def sgna(self) -> pd.DataFrame:
        """
        판관비 Sales, General and Administrative (SG & A) 데이터프레임
        :return:
        """
        df = self.obj2[4].copy()
        df.set_index(keys=['항목'], inplace=True)
        df.index.name = None
        return df.T

    @property
    def cost(self) -> pd.DataFrame:
        """
        매출 원가율 데이터프레임
        :return:
        """
        df = self.obj2[5].copy()
        df.set_index(keys=['항목'], inplace=True)
        df.index.name = None
        return df.T

    @property
    def rnd(self) -> pd.DataFrame:
        """
        R&D 투자현황
        :return:
        """
        if self._rnd_.empty:
            df = self.obj2[8]
            df.set_index(keys=['회계연도'], inplace=True)
            df.index.name = None
            df = df[['R&D 투자 총액 / 매출액 비중.1', '무형자산 처리 / 매출액 비중.1', '당기비용 처리 / 매출액 비중.1']]
            df = df.rename(columns={'R&D 투자 총액 / 매출액 비중.1': 'R&D투자비중',
                                    '무형자산 처리 / 매출액 비중.1': '무형자산처리비중',
                                    '당기비용 처리 / 매출액 비중.1': '당기비용처리비중'})
            if '관련 데이터가 없습니다.' in df.index:
                df.drop(index=['관련 데이터가 없습니다.'], inplace=True)
            self._rnd_ = df.copy()
        return self._rnd_

    def reform_statement(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        기업 기본 재무정보
        :param df: 원 데이터프레임
        :return:
        """
        df_copy = df.copy()
        cols = df_copy.columns.tolist()
        df_copy.set_index(keys=[cols[0]], inplace=True)
        df_copy.index.name = None
        df_copy.columns = df_copy.columns.droplevel()
        df_copy = df_copy.T

        key = [i[:-3] if i.endswith(')') else i for i in df_copy.index]
        if self._marketcap_.empty:
            from_date = (today - timedelta(365 * 7)).strftime("%Y%m%d")
            to_date = today.strftime("%Y%m%d")
            self._marketcap_ = stock.get_market_cap_by_date(
                fromdate=from_date, todate=to_date, ticker=self.ticker, freq='m'
            )
            self._marketcap_['ID'] = [date.strftime("%Y/%m") for date in self._marketcap_.index]
            self._marketcap_['시가총액'] = (self._marketcap_['시가총액'] / 100000000).astype(int)
        cap = self._marketcap_[self._marketcap_['ID'].isin(key)][['ID', '시가총액']].copy().set_index(keys='ID')
        df_copy = df_copy.join(cap, how='left')
        df_copy['PSR'] = round(df_copy['시가총액'] / df_copy[df_copy.columns[0]], 2)
        peg = (df_copy['PER'] / (100*df_copy['EPS(원)'].pct_change())).values
        df_copy['PEG'] = [round(v, 2) if v > 0 else 0 for v in peg]
        return df_copy


if __name__ == "__main__":
    api = fetch(ticker='009970')
    print(api.name)
    print("# 사업 소개")
    print(api.summary)

    # print("# 연간 실적")
    # print(api.annual)
    # print(api.annual['PEG'])

    # print("# 분기 실적")
    # print(api.quarter)
    
    print("# 외국인 보유 비율")
    print(api.foreigner)

    print("# 차입 공매도 현황")
    print(api.short)
    # print(api.consensus)
    # print(api.factors)
    # print(api.product)
    # print(api.sgna)
    # print(api.cost)
    # print(api.rnd)