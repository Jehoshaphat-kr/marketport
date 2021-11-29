import requests, tdatool
import pandas as pd
from bs4 import BeautifulSoup as Soup
from pykrx import stock
from datetime import datetime, timedelta


today = datetime.today()
class fundamental:

    def __init__(self, ticker:str):
        # Parameter
        self.ticker = ticker
        self.name = tdatool.meta.loc[ticker, '종목명']
        url1 = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701"
        url2 = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"

        # Fetch Company Business Summary
        html = requests.get(url1 % ticker).content
        soup = Soup(html, 'lxml')
        texts = soup.find('ul', id='bizSummaryContent').find_all('li')
        self.business_summary = '\n'.join([text.text.replace('&nbsp;', ' ') for text in texts])

        # Fetch CompanyGuide SnapShot
        self.obj1 = pd.read_html(url1 % ticker, encoding='utf-8')
        self.obj2 = pd.read_html(url2 % ticker, encoding='utf-8')
        self.is_separate = self.obj1[11].iloc[0].isnull().sum() > self.obj1[14].iloc[0].isnull().sum()

        # Fetch Market-Cap
        from_date = (today - timedelta(365 * 7)).strftime("%Y%m%d")
        to_date = today.strftime("%Y%m%d")
        self.cap = stock.get_market_cap_by_date(fromdate=from_date, todate=to_date, ticker=ticker, freq='m')
        self.cap['ID'] = [date.strftime("%Y/%m") for date in self.cap.index]
        self.cap['시가총액'] = (self.cap['시가총액'] / 100000000).astype(int)
        return

    @property
    def annual_statement(self) -> pd.DataFrame:
        """
        연간 실적/비율/배수 데이터프레임
        :return:
        """
        return self.reform_statement(self.obj1[14] if self.is_separate else self.obj1[11])

    @property
    def quarter_statement(self) -> pd.DataFrame:
        """
        분기 실적/비율/배수 데이터프레임
        :return:
        """
        return self.reform_statement(self.obj1[15] if self.is_separate else self.obj1[12])

    @property
    def consensus(self) -> pd.DataFrame:
        """
        컨센선스 Consensus 데이터프레임
        :return:
        """
        return self.reform_consensus(self.obj1[7])

    @property
    def sales_product(self) -> pd.DataFrame:
        """
        주요 매출 상품:: [reform] 리폼 필요
        :return:
        """
        return self.obj2[2]

    @property
    def market_share(self) -> pd.DataFrame:
        """
        주요 제품 시장 점유율:: [reform] 리폼 필요
        :return:
        """
        return self.obj2[3]

    @property
    def sg_a(self) -> pd.DataFrame:
        """
        판관비 Sales, General and Administrative (SG & A) 데이터프레임
        :return:
        """
        return self.reform_sga(self.obj2[4])

    @property
    def sales_cost(self) -> pd.DataFrame:
        """
        매출 원가율 데이터프레임
        :return:
        """
        return self.reform_sga(self.obj2[5])

    @property
    def rnd_invest(self) -> pd.DataFrame:
        """
        R&D 투자현황
        :return:
        """
        return self.reform_rnd(df=self.obj2[8])

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
        cap = self.cap[self.cap['ID'].isin(key)][['ID', '시가총액']].copy().set_index(keys='ID')
        df_copy = df_copy.join(cap, how='left')
        df_copy['PSR'] = round(df_copy['시가총액'] / df_copy[df_copy.columns[0]], 2)
        peg = (df_copy['PER'] / (100*df_copy['EPS(원)'].pct_change())).values
        df_copy['PEG'] = [round(v, 2) if v > 0 else 0 for v in peg]
        return df_copy

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
        if '관련 데이터가 없습니다.' in df.index:
            df.drop(index=['관련 데이터가 없습니다.'], inplace=True)
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
    # api = fundamental(ticker='000660')
    # print(api.annual_statement)
    # print(api.annual_statement['PEG'])
    # print(api.quarter_statement)
    # print(api.consensus)
    # print(api.sales_product)
    # print(api.market_share)
    # print(api.sg_a)
    # print(api.sales_cost)
    # print(api.rnd_invest)

    url = 'http://comp.fnguide.com/SVO2/common/chartListPopup2.asp?oid=div5_img&cid=05_05&gicode=A000660&filter=D&term=Y&etc=0&etc2=0&titleTxt=%EB%A9%80%ED%8B%B0%ED%8C%A9%ED%84%B0%20%EC%8A%A4%ED%83%80%EC%9D%BC%20%EB%B6%84%EC%84%9D&dateTxt=undefined&unitTxt='
    a = requests.get(url=url)
    print(a.text)