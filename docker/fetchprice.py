from datetime import datetime, timedelta
from pykrx import stock
from random import sample
import pandas as pd
import os, time

__root__ = os.path.dirname(os.path.dirname(__file__))
class dock:

    dir_warehouse = os.path.join(__root__, 'warehouse')
    dir_storage = os.path.join(__root__, 'warehouse/series')

    today = datetime.today()
    tic = datetime(2010, 1, 2).strftime("%Y%m%d")

    def __init__(self, date:datetime=None):
        print("=" * 50)
        print("|" + " " * 15 + "가격 정보 업데이트" + " " * 15 + "|")
        print("=" * 50)
        today = self.today if not date else date
        self.toc = today.strftime("%Y%m%d")
        self.ago = (today - timedelta(1)).strftime("%Y%m%d")

        print(f"PROP 날짜: {today.strftime('%Y-%m-%d')}")
        self.meta = pd.read_csv(
            filepath_or_buffer=os.path.join(self.dir_warehouse, 'meta-stock.csv'),
            encoding='utf-8',
            index_col='종목코드'
        )
        self.meta.index = self.meta.index.astype(str).str.zfill(6)

        self.market = pd.DataFrame()
        self.tickers_stored = [f.replace('.csv', '') for f in os.listdir(self.dir_storage)]
        self.tickers_ipo = []
        self.tickers_spl = []
        return

    def get_market(self) -> None:
        """
        업데이트 날짜의 시장 전체 가격 정보
        :return:
        """
        print("Proc 01: {} 시장 가격 정보 다운로드".format(self.toc))
        labels = ['시가', '고가', '저가', '종가', '거래량']
        df_stk = stock.get_market_ohlcv_by_ticker(self.toc, market='ALL')[labels]
        df_etf = stock.get_etf_ohlcv_by_ticker(self.toc)[labels]
        df_market = pd.concat([df_stk, df_etf], axis=0)
        self.market = df_market.groupby(level=0).last()
        return

    def get_ipo_or_split(self) -> None:
        """
        신규 상장 혹은 주식 분할 이슈 종목 선별
        :return:
        """
        print("Proc 02: 신규 상장/액면분할 이슈 종목 선별")
        prev = stock.get_market_cap_by_ticker(date=self.ago, market='ALL').rename(columns={'상장주식수': '전일'})
        curr = stock.get_market_cap_by_ticker(date=self.toc, market='ALL').rename(columns={'상장주식수': '금일'})
        comp = pd.concat(objs=[prev['전일'], curr['금일']], axis=1)
        self.tickers_ipo = comp[comp['전일'].isna()].index.tolist()
        self.tickers_spl = comp[comp['전일'] != comp['금일']].index.tolist()
        return

    def fetch(self, ticker:str) -> pd.DataFrame:
        """
        단일 종목코드에 대한 가격 데이터프레임 다운로드
        :param ticker: 종목코드
        :return:
        """
        error_message = ''
        for trial_cnt in range(5):
            try:
                data = stock.get_market_ohlcv_by_date(fromdate=self.tic, todate=self.toc, ticker=ticker)
                data.reset_index(level=0, inplace=True)
                return data
            except Exception as e:
                time.sleep(5)
                if trial_cnt == 4:
                    error_message = e
        print("[{}] : ".format(ticker), error_message)
        return pd.DataFrame()

    def update(self, debug:bool=False) -> None:
        """
        전체 종목코드에 대한 가격 데이터프레임 다운로드 후 저장
        :param debug:
        :return:
        """
        self.get_market()
        self.get_ipo_or_split()

        fails = []

        ''' IPO / 액면분할 종목 다운로드 '''
        new = self.tickers_ipo + self.tickers_spl
        if new:
            print("  - 신규 상장/액면분할 종목 재편 중")
        else:
            print("  - 신규 상장/액면분할 종목 없음")
        for n, ticker in enumerate(new):
            if debug: print('    {:3.2f}%: {}'.format(100 * (n + 1) / len(new), ticker))
            data = self.fetch(ticker=ticker)
            if data.empty:
                fails.append(ticker)
                continue

            data.to_csv(os.path.join(self.dir_storage, '{}.csv'.format(ticker)), encoding='utf-8', index=False)
            if not (n+1) % 12:
                time.sleep(3)
        ''' ---------------------------------------------------------------------------------- '''

        print("Proc 03: 정규 종목 가격 업데이트")
        meta = self.meta.copy()
        market = self.market.copy()
        toc_label = datetime.strptime(self.toc, "%Y%m%d").strftime("%Y-%m-%d")
        for n, ticker in enumerate(meta.index):
            if debug: print('    {:3.2f}%: {}'.format(100 * (n + 1) / len(meta), ticker))

            # 기존 다운로드 이력이 없는 경우 신규 추가
            if not ticker in self.tickers_stored:
                data = self.fetch(ticker=ticker)
                if data.empty:
                    fails.append(ticker)
                    continue
                data.to_csv(os.path.join(self.dir_storage, '{}.csv'.format(ticker)), encoding='utf-8', index=False)
                if not (n+1) % 12:
                    time.sleep(3)

            # 기존 다운로드 이력이 있는 경우 업데이트
            else:
                data = pd.read_csv(
                    filepath_or_buffer=os.path.join(self.dir_storage, '{}.csv'.format(ticker)),
                    encoding='utf-8',
                    index_col='날짜'
                )
                if data.index[-1].replace('-', '') == self.toc:
                    continue
                specify = market[market.index == ticker].copy()
                specify.index = [toc_label]
                updated = pd.concat(objs=[data, specify], axis=0)
                updated.index.name = '날짜'
                updated.reset_index(level=0, inplace=True)
                updated.to_csv(os.path.join(self.dir_storage, '{}.csv'.format(ticker)), encoding='utf-8', index=False)
        ''' ---------------------------------------------------------------------------------- '''

        print('** 랜덤 샘플 케이스')
        tickers = sample(meta.index.tolist(), 10)
        for ticker in tickers:
            name = meta.loc[ticker, '종목명']
            df = pd.read_csv(os.path.join(self.dir_storage, '{}.csv'.format(ticker)), index_col='날짜')
            print("  - UPDATED @ %s: [%s] %s" % (str(df.index[-1]), ticker, name))

        print('** 업데이트 실패 항목 ** ')
        if fails:
            print(meta[meta.index.isin(fails)])
        else:
            print('  - 없음')
        return



if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    
    docker = dock()
    docker.update(debug=True)


    ''' TEST SET MANIPULATE '''
    # set_date = datetime(2021, 10, 6)
    # for n, e in enumerate(docker.tickers_stored):
    #     # if n == 3: break
    #     df = pd.read_csv(
    #         filepath_or_buffer=os.path.join(docker.dir_storage, '{}.csv'.format(e)),
    #         encoding='utf-8',
    #     )
    #     if not '날짜' in df.columns:
    #         print(e)
    #         continue
    #
    #     df.set_index(keys='날짜', inplace=True)
    #     df.index = pd.to_datetime(df.index)
    #     if df.index[-1] == set_date:
    #         continue
    #
    #     df = df[df.index <= set_date]
    #     if df.empty:
    #         continue
    #
    #     df.reset_index(level=0, inplace=True)
    #     df.to_csv(os.path.join(docker.dir_storage, '{}.csv'.format(e)), encoding='utf-8', index=False)

