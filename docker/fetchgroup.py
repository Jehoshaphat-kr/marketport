from datetime import datetime
from selenium import webdriver
import pandas as pd
import os, time, requests


__root__ = os.path.dirname(os.path.dirname(__file__))
class dock:
    dir_warehouse = os.path.join(__root__, 'warehouse')
    dir_storage = os.path.join(__root__, 'warehouse/group')
    dir_handler = os.path.join(__root__, 'warehouse/group/handler')

    driver = None
    meta = pd.read_csv(
        filepath_or_buffer=os.path.join(dir_warehouse, 'meta-stock.csv'),
        encoding='utf-8',
        index_col='종목코드'
    )
    wics_meta = pd.read_csv(
        filepath_or_buffer=os.path.join(dir_handler, 'WICSMETA.csv'),
        encoding='utf-8',
        index_col='SectorCode'
    )
    wi26_meta = pd.read_csv(
        filepath_or_buffer=os.path.join(dir_handler, 'WI26META.csv'),
        encoding='utf-8',
        index_col='SectorCode'
    )
    url = 'http://www.wiseindex.com/Index/Index#/%s.0.Components'
    wics = pd.DataFrame()
    wi26 = pd.DataFrame()

    def __init__(self):
        return

    def wise_init(self, wise:str='ALL') -> None:
        """
        CHROME 크롤러 초기화
        :param wise: 분류 방법 선택 실행
        :return:
        """
        print("=" * 50)
        print("|" + " " * 10 + "WISE 산업/업종 분류 다운로드" + " " * 10 + "|")
        print("=" * 50)
        print("PROP 날짜: {}".format(datetime.today().strftime("%Y-%m-%d")))
        option = webdriver.ChromeOptions()
        option.add_argument('--headless')
        option.add_argument('--no-sandbox')
        option.add_argument('--disable-dev-shm-usage')
        option.add_argument('--disable-gpu')

        self.wise = wise
        self.driver = webdriver.Chrome(executable_path='chromedriver.exe', options=option)
        return

    def wise_date(self) -> None:
        """
        WISE 산업 분류 날짜 확인
        :return: 
        """
        response = requests.get(self.url % self.wics_meta.index[0])
        source = response.text
        i_start = source.find("기준일")
        i_end = source[i_start:].find("</p>")
        source_date = datetime.strptime(source[i_start + 6: i_start + i_end], "%Y.%m.%d").strftime("%Y-%m-%d")
        print("PROP 날짜: {} <-- Source".format(source_date))
        return

    def wise_update(self) -> None:
        """
        WISE 산업 분류 업데이트
        :return:
        """
        for i, wise, frm in [('01', 'WICS', self.wics_meta), ('02', 'WI26', self.wi26_meta)]:
            if not self.wise == 'ALL' and not self.wise == wise:
                continue
            print("Proc {}: {} 분류 다운로드".format(i, wise))
            buff = []
            for n, code in enumerate(frm.index):
                name = frm.loc[code, 'Sector']
                print("Proc {}-{}: {:.2f}% {}".format(i, str(n+1).zfill(2), 100 * (n+1)/len(frm), name), end=" ")

                flag = False
                for retry in range(1, 6):
                    try:
                        self.driver.get(url=self.url % code)
                        self.driver.refresh()
                        time.sleep(3)

                        fetch = pd.read_html(self.driver.page_source, header=0)[-1]
                        fetch.drop(columns=['섹터명'], inplace=True)
                        if wise == "WICS":
                            fetch['산업'] = frm.loc[code, 'Industry']
                        fetch['섹터'] = name
                        buff.append(fetch)
                        print("SUCCESS")
                        flag = False
                        break
                    except Exception as e:
                        print('RETRY(%d),' % retry, end=' ')
                        time.sleep(5)
                if flag:
                    print("FAILED ***")
                time.sleep(2)

            pd.concat(objs=buff, axis=0, ignore_index=True).to_csv(
                os.path.join(self.dir_storage, wise + '.csv'),
                encoding='utf-8',
                index=False
            )
        self.driver.close()
        return

    def wise_postproc(self) -> None:
        """
        WISE 산업 분류 후처리 및 저장
        :return:
        """
        time.sleep(1)
        print("Proc 03: 산업분류 후처리")
        meta = self.meta.copy()
        meta.reset_index(level=0, inplace=True)
        meta.set_index(keys='종목명', inplace=True)
        for n, wise in enumerate(['WICS', 'WI26']):
            if not self.wise == 'ALL' and not self.wise == wise:
                continue
            print(f"Proc 03-{str(n+1).zfill(2)}: {wise} 분류 후처리")
            frm = pd.read_csv(
                filepath_or_buffer=os.path.join(self.dir_storage, wise + '.csv'),
                index_col='종목명',
                encoding='utf-8'
            )
            frm = frm.join(meta[['종목코드']], how='left')
            frm_na = frm[frm['종목코드'].isna()].copy()
            if not frm_na.empty:
                print("종목코드 미상 항목 발생")
                print("RENAME.csv @handler 업데이트 필요")
                print(frm_na)
                print('-' * 70)
                return
            frm.reset_index(level=0, inplace=True)
            frm.to_csv(os.path.join(self.dir_storage, wise + '.csv'), index=False)
        return

    def theme_update(self) -> None:
        """
        테마 파일 CSV 변환
        :return:
        """
        print("** 테마 분류 업데이트 **")
        meta = self.meta.reset_index(level=0).copy()
        meta.set_index(keys='종목명', inplace=True)
        meta['종목코드'] = meta['종목코드'].astype(str).str.zfill(6)

        theme = pd.read_excel(os.path.join(self.dir_handler, 'THEME.xlsx'), index_col='종목명')
        theme.drop(columns=['출처'], inplace=True)
        theme = theme.join(meta[['종목코드']], how='left')

        no_ticker = theme[theme['종목코드'].isna()]
        if not no_ticker.empty:
            print("종목코드 미상 항목 발생: 업데이트 중단")
            print(no_ticker)
            print("RENAME.csv @handler 업데이트 필요")
        else:
            theme.reset_index(level=0, inplace=True)
            theme.to_csv(os.path.join(self.dir_storage, 'THEME.csv'), index=False, encoding='utf-8')
        print("-" * 70)
        return

    def etf_check(self) -> None:
        """
        ETF 최신화 여부 확인
        :return: 
        """
        etf_online = self.meta[self.meta['거래소'] == 'ETF'].copy()
        etf_offline = pd.read_excel(os.path.join(self.dir_handler, 'TDATETF.xlsx'), index_col='종목코드')

        to_be_delete = etf_offline[~etf_offline.index.isin(etf_online.index)]
        to_be_update = etf_online[~etf_online.index.isin(etf_offline.index)]

        ''' NOTICE '''
        for kind, frm in [('삭제', to_be_delete), ('추가', to_be_update)]:
            print("** ETF 분류 {} 필요 항목 **".format(kind))
            if frm.empty:
                print("--> 없음")
            else:
                print(frm)
                if kind == '추가':
                    os.startfile(self.dir_handler)
            print("-" * 70)
        return

    def etf_update(self) -> None:
        """
        ETF 수기 파일 CSV 변환
        :return:
        """
        print("** ETF 분류 업데이트 **")
        etf_offline = pd.read_excel(os.path.join(self.dir_handler, 'TDATETF.xlsx'))
        etf_offline.to_csv(os.path.join(self.dir_storage, 'ETF.csv'), index=False, encoding='utf-8')
        print("-" * 70)
        return


'''
Partial Code:: @dock.wise_update()
WISE 산업 분류 크롤링 시, pandas.read_html 사용 불가 상태 일 때 아래 코드 추가 필요
------------------------------------------ < 아래 > ------------------------------------------ 
    # if fetch.empty:
    #     xpath = '//*[@id="ng-app"]/div/div/div[2]/div[2]/div[3]/div[2]/div/div/table/tbody/tr'
    #     cps = self.driver.find_elements_by_xpath(xpath)
    #     raw_names = [cp.text for cp in cps]
    #     sector_name = [code['Sector']] * len(cps)
    #     fetch = pd.DataFrame(data={"종목명": raw_names, "섹터": sector_name})
----------------------------------------------------------------------------------------------
'''


if __name__ == "__main__":

    docker = dock()

    ''' WICS/WI26 '''
    docker.wise_init(wise='WICS')
    docker.wise_date()
    docker.wise_update()
    docker.wise_postproc()

    ''' ETF '''
    # docker.etf_check()
    # docker.etf_update()

    ''' THEME '''
    # docker.theme_update()
