import requests, json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as Soup
from pykrx import stock
from datetime import datetime, timedelta
from urllib.request import urlopen


today = datetime.today()
class finances:

    def __init__(self, ticker:str, meta=None):
        """
        :param ticker: 종목코드
        :param meta: DataFrame
        """
        # Parameter
        self.ticker = ticker
        self.name = meta.loc[ticker, '종목명'] if type(meta) == type(pd.DataFrame()) else stock.get_market_ticker_name(ticker=ticker)

        self.url1 = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701"
        self.url2 = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"

        # Fetch CompanyGuide SnapShot
        self.obj1 = list()
        self.obj2 = list()
        self.is_init = False
        self.is_link = True

        # Initialize DataFrames
        self._multiples_ = pd.DataFrame()
        self._foreigner_ = pd.DataFrame()
        self._consensus_ = pd.DataFrame()
        self._relmulti_ = pd.DataFrame()
        self._relyield_ = pd.DataFrame()
        self._balance_ = pd.DataFrame()
        self._product_ = pd.DataFrame()
        self._factors_ = pd.DataFrame()
        self._shorts_ = pd.DataFrame()
        self._rnd_ = pd.DataFrame()
        return

    def _init_object_(self):
        """
        초기 웹 데이터프레임 다운로드
        :return:
        """
        if not self.is_init:
            self.obj1 = pd.read_html(self.url1 % self.ticker, encoding='utf-8')
            self.obj2 = pd.read_html(self.url2 % self.ticker, encoding='utf-8')
            self.is_link = self.obj1[11].iloc[0].isnull().sum() > self.obj1[14].iloc[0].isnull().sum()
            self.is_init = True
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

        syllables = []
        for n in range(1, len(text) - 2):
            if text[n] == '.':
                if text[n-1].isdigit() or text[n+1].isdigit() or text[n+1].isalpha():
                    syllables.append('.')
                else:
                    syllables.append('.\n')
            else:
                syllables.append(text[n])
        return ' ' + text[0] + ''.join(syllables) + text[-2] + text[-1]

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
    def relyield(self) -> pd.DataFrame:
        """
        상대수익률 (3M, 1Y)
        :return:
        """
        if self._relyield_.empty:
            objs = {}
            for period in ['1Y', '3M']:
                url = f"http://cdn.fnguide.com/SVO2/json/chart/01_01/chart_A{self.ticker}_{period}.json"
                data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
                header = pd.DataFrame(data["CHART_H"])[['ID', 'PREF_NAME']]
                header = header[header['PREF_NAME'] != ""]
                inner = pd.DataFrame(data["CHART"])[
                    ['TRD_DT'] + header['ID'].tolist()
                ].set_index(keys='TRD_DT').rename(columns=header.set_index(keys='ID').to_dict()['PREF_NAME'])
                inner.index = pd.to_datetime(inner.index)
                objs[period] = inner
            self._relyield_ = pd.concat(objs=objs, axis=1)
        return self._relyield_

    @property
    def relmultiple(self) -> pd.DataFrame:
        """
        상대 배수(Multiple)
        :return:
        """
        if self._relmulti_.empty:
            url = f"http://cdn.fnguide.com/SVO2/json/chart/01_04/chart_A{self.ticker}_D.json"
            data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
            objs = {}
            for label, index in (('PER', '02'), ('EV/EBITA', '03')):
                header1 = pd.DataFrame(data[f'{index}_H'])[['ID', 'NAME']].set_index(keys='ID')
                header1['NAME'] = header1['NAME'].astype(str).str.replace("'", "20")
                header1 = header1.to_dict()['NAME']
                header1.update({'CD_NM':'이름'})

                inner1 = pd.DataFrame(data[index])[list(header1.keys())].rename(columns=header1).set_index(keys='이름')
                inner1.index.name = None
                for col in inner1.columns:
                    inner1[col] = inner1[col].apply(lambda x: np.nan if x == '-' else x)
                objs[label] = inner1.T
            self._relmulti_ = pd.concat(objs=objs, axis=1)
        return self._relmulti_

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
    def balance(self) -> pd.DataFrame:
        """
        대차 잔고 비중 데이터프레임
        :return:
        """
        if self._balance_.empty:
            url = f"http://cdn.fnguide.com/SVO2/json/chart/11_01/chart_A{self.ticker}_BALANCE1Y.json"
            data = json.loads(urlopen(url).read().decode('utf-8-sig', 'replace'))
            self._balance_ = pd.DataFrame(data['CHART'])[['TRD_DT', 'BALANCE_RT', 'ADJ_PRC']].rename(columns={
                'TRD_DT': '날짜', 'BALANCE_RT': '대차잔고비중', 'ADJ_PRC': '수정 종가'
            }).set_index(keys='날짜')
            self._balance_.index = pd.to_datetime(self._balance_.index)
        return self._balance_

    @property
    def annual(self) -> pd.DataFrame:
        """
        연간 실적/비율/배수 데이터프레임
        :return:
        """
        self._init_object_()
        df_copy = (self.obj1[14] if self.is_link else self.obj1[11]).copy()
        cols = df_copy.columns.tolist()
        df_copy.set_index(keys=[cols[0]], inplace=True)
        df_copy.index.name = None
        df_copy.columns = df_copy.columns.droplevel()
        df_copy = df_copy.T
        return df_copy

    @property
    def quarter(self) -> pd.DataFrame:
        """
        분기 실적/비율/배수 데이터프레임
        :return:
        """
        self._init_object_()
        df_copy = (self.obj1[15] if self.is_link else self.obj1[12]).copy()
        cols = df_copy.columns.tolist()
        df_copy.set_index(keys=[cols[0]], inplace=True)
        df_copy.index.name = None
        df_copy.columns = df_copy.columns.droplevel()
        df_copy = df_copy.T
        return df_copy

    @property
    def product(self) -> pd.DataFrame:
        """
        주요 매출 상품:: [reform] 리폼 필요
        :return:
        """
        self._init_object_()
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
        self._init_object_()
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
        self._init_object_()
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
        self._init_object_()
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

    @property
    def multiples(self) -> pd.DataFrame:
        """
        PY KRX 투자 배수 / EPS 데이터프레임 (반복 호출 불가)
        :return:
        """
        if self._multiples_.empty:
            fromdate = (today - timedelta(365 * 3)).strftime("%Y%m%d")
            todate = today.strftime("%Y%m%d")
            ratio = stock.get_market_fundamental_by_date(fromdate=fromdate, todate=todate, ticker=self.ticker)
            marketcap = stock.get_market_cap_by_date(fromdate=fromdate, todate=todate, ticker=self.ticker)

            quarter = self.quarter.copy()
            key = '매출액'
            key = '순영업수익' if '순영업수익' in quarter.columns else key
            key = '보험료수익' if '보험료수익' in quarter.columns else key
            q_sales = quarter[key].dropna()
            q_sales = q_sales[q_sales.index.str.startswith(str(today.year))]

            sales = self.annual[key].dropna()
            sales = sales[~sales.index.str.endswith(')')]
            sales = sales.append(to_append=pd.Series(data={f'{today.year}/12': int(4 * q_sales.sum() / len(q_sales))}))
            sales.index = sales.index + '/30'
            sales.index = pd.to_datetime(sales.index)
            sales.name = key
            sales = sales.astype(float)
            sales = sales[sales.index >= marketcap.index[0]]

            multiples = pd.concat(objs=[sales, marketcap], axis=1)[[key, '시가총액']]
            multiples['매출액'] = multiples[key].interpolate(method='nearest')
            multiples['시가총액'] = multiples['시가총액']/100000000
            multiples['PSR'] = multiples['시가총액'] / multiples[key]
            self._multiples_ = multiples.join(ratio).dropna()
        return self._multiples_


if __name__ == "__main__":
    api = finances(ticker='006400')
    # api = finances(ticker='000660')
    print(api.name)
    # print("# 사업 소개")
    print(api.summary)

    # print("# 연간 실적")
    # print(api.annual)
    # print(api.annual['매출액'])

    # print("# 분기 실적")
    # print(api.quarter)
    
    # print("# 외국인 보유 비율")
    # print(api.foreigner)

    # print("# 차입 공매도 현황")
    # print(api.short)

    # print("# 대차잔고비중")
    # print(api.balance)
    # print(api.consensus)
    # print(api.factors)
    # print(api.product)
    # print(api.sgna)
    # print(api.cost)
    # print(api.rnd)
    # print(api.multiples)
    # print(api.relyield)
    # print(api.relmultiple)