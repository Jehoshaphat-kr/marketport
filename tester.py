# import tdatool as tt
#
# if __name__ == "__main__":
#     ticker = '025900'
#
#     fundamental = tt.Fundamental(ticker=ticker)
#     print(f'{fundamental.name}({fundamental.ticker}) 기본적 분석')
#
#     fundamental.show_business(show=True, save=True)
#     fundamental.show_summary(show=False, save=True)
#     fundamental.show_sales(show=False, save=True)
#     fundamental.show_multiple(show=False, save=True)
#
#     technical = tt.Technical(ticker=ticker, period=5, src='local')
#     print(f'{technical.name}({technical.ticker}) 기술적 분석')
#     technical.show_price(show=False, save=True)
#     technical.show_trend(show=False, save=True)


import requests, time
import pandas as pd
from datetime import datetime


if __name__ == '__main__':

    url = 'http://www.wiseindex.com/Index/Index#/%s.0.Components'
    response = requests.get(url % 'G1010')
    source = response.text
    i_start = source.find("기준일")
    i_end = source[i_start:].find("</p>")
    date = datetime.strptime(source[i_start + 6: i_start + i_end], "%Y.%m.%d").strftime("%Y%m%d")

    df = pd.DataFrame(columns=['code', 'name', 'ls', 'ms'])

    wics = {1010: '에너지',
               1510: '소재',
               2010: '자본재',
               2020: '상업서비스와공급품',
               2030: '운송',
               2510: '자동차와부품',
               2520: '내구소비재와의류',
               2530: '호텔,레스토랑,레저 등',
               2550: '소매(유통)',
               2560: '교육서비스',
               3010: '식품과기본식료품소매',
               3020: '식품,음료,담배',
               3030: '가정용품과개인용품',
               3510: '건강관리장비와서비스',
               3520: '제약과생물공학',
               4010: '은행',
               4020: '증권',
               4030: '다각화된금융',
               4040: '보험',
               4050: '부동산',
               4510: '소프트웨어와서비스',
               4520: '기술하드웨어와장비',
               4530: '반도체와반도체장비',
               4535: '전자와 전기제품',
               4540: '디스플레이',
               5010: '전기통신서비스',
               5020: '미디어와엔터테인먼트',
               5510: '유틸리티'}

    retry = []
    for n, code in enumerate(wics):
        print(f'{100*(n+1)/len(wics):.2f}% ::{code} {wics[code]}')
        response = requests.get(
            url=f'http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={date}&sec_cd=G{code}'
        )
        try:
            json_list = response.json()
            for json in json_list['list']:
                ls = json['SEC_NM_KOR']
                ms = json['IDX_NM_KOR'][5:]
                code = json['CMP_CD']
                name = json['CMP_KOR']
                df = df.append({'code': code, 'name': name, 'ls': ls, 'ms': ms}, ignore_index=True)
        except ConnectionAbortedError as e:
            print(f'Error: HTTP {response.status_code} while fetch WICS {code} / {wics[code]}')
            retry.append(code)


        time.sleep(1)

        print(df)