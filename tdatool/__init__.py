from tdatool.charts import Technical, Fundamental
import os
import pandas as pd


root = os.path.dirname(os.path.dirname(__file__))
meta = pd.read_csv(
    'http://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/market/market.csv',
    encoding='utf-8',
    index_col='종목코드'
).join(
    other=pd.read_csv(
        'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/group/WI26.csv',
        encoding='utf-8',
        index_col='종목코드'
    ).drop(columns=['종목명']),
    how='left'
)[['섹터', '종목명', '종가', '시가총액', 'R1D', 'R1W', 'R1M', 'PER', 'DIV']].rename(
    columns={'R1D': '1일등락', 'R1W': '1주등락', 'R1M': '1개월등락'}
)
for col in meta.columns:
    if '등락' in col:
        meta[col] = round(meta[col], 2).astype(str) + '%'
meta['PER'] = round(meta['PER'], 2)
meta['종가'] = meta['종가'].apply(lambda p: '{:,}원'.format(int(p)))
cap = (meta["시가총액"] / 100000000).astype(int).astype(str)
meta['시가총액'] = cap.apply(lambda v: v + "억" if len(v) < 5 else v[:-4] + '조 ' + v[-4:] + '억')
meta.index = meta.index.astype(str).str.zfill(6)