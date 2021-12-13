from typing import ItemsView
import pandas as pd
import plotly.graph_objects as go


colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

def reform(span) -> list:
    """
    날짜 형식 변경 (from)datetime --> (to)YY/MM/DD
    :param span: 날짜 리스트
    :return:
    """
    return [f'{d.year}/{d.month}/{d.day}' for d in span]

def trace_price(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    기본 주가 차트 요소
    :param df: 주가 데이터프레임
    :return: dict() :: key = ['일봉', '시가', '고가', '저가', '종가']
    """
    require = ['시가', '고가', '저가', '종가']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for price data')

    objects = dict()
    objects['일봉'] = go.Candlestick(
        name='일봉', x=df.index, open=df['시가'], high=df['고가'], low=df['저가'], close=df['종가'],
        increasing_line=dict(color='red'), decreasing_line=dict(color='royalblue'),
        visible=True, showlegend=True,
    )

    for col in require:
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            line=dict(color='grey'),
            visible='legendonly', showlegend=True,
            meta=reform(span=df.index),
            hovertemplate=col + ': %{y:,}원<br>날짜: %{meta}<extra></extra>',
        )
    return objects.items()

def trace_volume(df:pd.DataFrame, is_main:bool=False) -> ItemsView[str, go.Bar]:
    """
    거래량 차트 요소
    :param df: 거래량 데이터프레임
    :param is_main: 거래량 주요 분석 대상 여부
    :return:
    """
    require = ['거래량']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for volume data')

    objects = dict()
    objects['거래량'] = go.Bar(
        name='거래량', x=df.index, y=df['거래량'],
        marker=dict(color=['blue' if d < 0 else 'red' for d in df['거래량'].pct_change().fillna(1)]),
        visible=True, showlegend=is_main,
        meta=reform(span=df.index),
        hovertemplate='날짜:%{meta}<br>거래량:%{y:,d}<extra></extra>'
    )
    return objects.items()

def trace_filters(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    주가 필터 차트 요소
    :param df: 필터 데이터프레임
    :return: dict() :: key = ['SMA(n)D', 'EMA(n)D', 'IIR(n)D']
    """
    meta = reform(span=df.index)
    objects = dict()
    for col in df.columns:
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            visible='legendonly', showlegend=True,
            meta=meta,
            hovertemplate=col + '<br>값: %{y:,d}원<br>날짜: %{meta}<extra></extra>',
        )
    return objects.items()

def trace_bollinger(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    볼린저밴드 차트 요소
    :param df: 볼린저밴드 데이터프레임
    :return: dict() :: key = ['상한선', '기준선', '하한선', '상한지시', '하한지시', '밴드폭', '신호']
    """
    require = ['상한선', '기준선', '하한선']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for bollinger-band data')

    objects = dict()
    meta = reform(span=df.index)
    for n, col in enumerate(['상한선', '기준선', '하한선']):
        objects[col] = go.Scatter(
            name='볼린저밴드', x=df.index, y=df[col],
            mode='lines', line=dict(color='rgb(184, 247, 212)'), fill='tonexty' if n else None,
            visible='legendonly', showlegend=False if n else True, legendgroup='볼린저밴드',
            meta=meta,
            hovertemplate=col + '<br>날짜: %{meta}<br>값: %{y:,d}원<extra></extra>',
        )
    for n, col in enumerate(['상한지시', '하한지시']):
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            mode='markers', marker=dict(
                symbol=f'triangle-{"up" if n else "down"}', color='red' if n else 'royalblue', size=9
            ), visible='legendonly', showlegend=True,
            hoverinfo='skip'
        )
    for col in ['밴드폭', '신호']:
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            visible=True, showlegend=True,
            meta=meta,
            hovertemplate=col + '<br>날짜: %{meta}<br>값: %{y:.2f}<extra></extra>'
        )
    return objects.items()

def trace_macd(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    MACD 차트 요소
    :param df: MACD 데이터프레임
    :return:
    """
    objects = dict()

    return objects.items()

def trace_pivot(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    피벗 지점 차트 요소
    :param df: 피벗 지점 데이터프레임
    :return: dict() :: key = [고점, 저점]
    """
    require = ['고점', '저점']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for pivot data')

    objects = dict()
    for n, col in enumerate(df.columns):
        sr = df[col].dropna()
        objects[col] = go.Scatter(
            name=f'{col}피벗', x=sr.index, y=sr,
            mode='markers', marker=dict(
                symbol='circle', color='blue' if col == '고점' else 'red', size=8, opacity=0.7
            ), visible='legendonly', showlegend=True,
            meta=reform(span=sr.index),
            hovertemplate='날짜: %{meta}<br>' + col + '피벗: %{y:,d}원<extra></extra>'
        )
    return objects.items()

def trace_trend(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    직선 추세 차트 요소
    :param df: 직선 추세 데이터프레임
    :return: dict() :: key = ['1Y평균저항선', '1Y평균지지선', '6M평균저항선', '6M평균지지선', '3M평균저항선', '3M평균지지선',
                              '1Y표준저항선', '1Y표준지지선', '6M표준저항선', '6M표준지지선', '3M표준저항선', '3M표준지지선']
    """
    require = ['1Y평균저항선', '1Y평균지지선', '6M평균저항선', '6M평균지지선', '3M평균저항선', '3M평균지지선',
               '1Y표준저항선', '1Y표준지지선', '6M표준저항선', '6M표준지지선', '3M표준저항선', '3M표준지지선']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for trend line data')

    meta = reform(span=df.index)
    objects = {}
    for n, col in enumerate(df.columns):
        key = f"{col[:4]}추세"
        objects[col] = go.Scatter(
            name=key, x=df.index, y=df[col],
            mode='lines', line=dict(color='royalblue' if col.endswith('저항선') else 'red'),
            visible='legendonly', showlegend=False if col.endswith('지지선') else True, legendgroup=key,
            meta=meta,
            hovertemplate='날짜: %{meta}<br>' + col + ': %{y:,d}원<extra></extra>'
        )
    return objects.items()

def trace_rsi(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    RSI 차트 요소
    :param df: RSI 데이터프레임
    :return: dict() :: key = ['RSI', 'STOCH-RSI', 'STOCH-RSI-Sig']
    """
    require = ['RSI', 'STOCH-RSI', 'STOCH-RSI-Sig']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for trend line data')

    meta = reform(span=df.index)
    objects = dict()
    for n, col in enumerate(df.columns):
        name = col.replace('STOCH-RSI', 'RSI(S)') if col.startswith('STOCH') else col
        objects[col] = go.Scatter(
            name=name, x=df.index, y=df[col],
            visible=True, showlegend=True,
            meta=meta,
            hovertemplate='날짜: %{meta}<br>' + name + ': %{y:.2f}<extra></extra>'
        )
    for n, label in enumerate(['upper', 'lower']):
        objects[label] = go.Scatter(
            name='RIS(S)-Range', x=df.index, y=[20 if n else 80] * len(df),
            mode='lines', line=dict(color='rgb(184, 247, 212)'), fill='tonexty' if n else None,
            visible='legendonly', showlegend=True if n else False, legendgroup='Range',
            hoverinfo='skip'
        )
    return objects.items()

def trace_product(df:pd.DataFrame) -> ItemsView[str, go.Pie]:
    """
    Company Guide 제품 구성 차트 요소
    :param df: 제품 구성 데이터프레임
    :return: dict() :: Key 필요 Column 없음 (자동 설정)
    """
    objects = dict()
    objects['Item'] = go.Pie(
        name='Product', labels=df.index, values=df,
        textinfo='label+percent', insidetextorientation='radial',
        visible=True, showlegend=False,
        hoverinfo='label+percent'
    )
    return objects.items()

def trace_asset(df:pd.DataFrame) -> ItemsView[str, go.Bar]:
    """
    Company Guide 자산 차트 요소
    :param df: 자산 데이터프레임
    :return: dict() :: Key ['자산', '부채']
    """
    require = ['자산총계', '부채총계', '자본총계']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for asset data')

    objects = dict()
    asset = df['자산총계'].fillna(0).astype(int)
    debt = df['부채총계'].fillna(0).astype(int)
    capital = df['자본총계'].fillna(0).astype(int)
    objects['자산'] = go.Bar(
        name='자산', x=df.index, y=asset,
        marker=dict(color='green'), opacity=0.9,
        text=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in asset],
        meta=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in debt],
        customdata=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in capital],
        visible=True, showlegend=False, offsetgroup=0,
        texttemplate=' ',
        hovertemplate='자산: %{text}억원<br>부채: %{meta}억원<br>자본: %{customdata}억원<extra></extra>',
    )

    objects['부채'] = go.Bar(
        name='부채', x=df.index, y=debt,
        marker=dict(color='red'), opacity=0.8,
        visible=True, showlegend=False, offsetgroup=0,
        hoverinfo='skip',
    )
    return objects.items()

def trace_sales(df:pd.DataFrame, is_annual:bool=True) -> ItemsView[str, go.Bar]:
    """
    Company Guide 연간/분기 실적 차트 요소
    :param df: 연간/분기 실적 데이터프레임
    :param is_annual: 연간 데이터 여부
    :return: dict() :: Key ['시가총액', '매출액', '영업이익', '당기순이익']
    """
    key = '매출액'
    key = '순영업수익' if '순영업수익' in df.columns else key
    key = '보험료수익' if '보험료수익' in df.columns else key
    require = [key, '영업이익', '당기순이익']

    objects = dict()
    for n, col in enumerate(require):
        y = df[col].fillna(0).astype(int)
        objects[col] = go.Bar(
            name=f'연간{col}' if is_annual else f'분기{col}', x=df.index, y=y,
            marker=dict(color=colors[n]),  opacity=0.9,
            legendgroup=col,
            meta=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in y],
            hovertemplate=col + ': %{meta}억원<extra></extra>',
        )
    return objects.items()

def trace_sales_ratio(df:pd.DataFrame) -> ItemsView[str, go.Bar]:
    """
    Company Guide 연간 실적 배수 차트 요소
    :param df: 연간 실적 배수 데이터프레임
    :return: dict() :: Key ['ROE', 'ROA', '영업이익률']
    """
    require = ['ROA', 'ROE', '영업이익률']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for annual statement data')

    objects = dict()
    for n, col in enumerate(require):
        objects[col] = go.Bar(
            name=f'연간{col}', x=df.index, y=df[col],
            marker=dict(color=colors[n]), opacity=0.9,
            legendgroup=col,
            hovertemplate=col + ': %{y}%<extra></extra>',
        )
    return objects.items()

def trace_multiple(df:pd.DataFrame, require:list) -> ItemsView[str, go.Scatter]:
    """
    PyKrx 투자 배수 차트 요소
    :param df: 투자 배수 시계열 데이터프레임
    :param require: 지표/근거 지표 입력 (e.g. PSR/매출/시가총액)
    :return: dict() Key
    """
    meta = reform(span=df.index)
    objects = dict()
    for n, col in enumerate(require):
        is_base = col in ['매출액', 'EPS', 'BPS', 'DPS']
        cd = [
            str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in df[col]
        ] if col == '매출액' else df[col]
        hover = ': %{customdata:,}원' if is_base else ': %{customdata:.2f}'
        hover = ': %{customdata}원' if col == '매출액' else hover
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            visible=True, showlegend=True,
            meta=meta, customdata=cd,
            hovertemplate='날짜: %{meta}<br>' + col + hover + '<extra></extra>'
        )
    return objects.items()


def trace_factors(df:pd.DataFrame) -> ItemsView[str, go.Scatterpolar]:
    """
    Company Guide Multi-Factor 멀티팩터 차트 요소
    :param df: 멀티팩터 데이터프레임
    :return: dict() :: Key 필요 Column 없음 (자동 설정)
    """
    objects = dict()
    for n, col in enumerate(df.columns):
        objects[col] = go.Scatterpolar(
            name=col, r=df[col].astype(float), theta=df.index,
            fill='toself',
            visible='legendonly' if n else True, showlegend=True,
            hovertemplate=col + '<br>팩터: %{theta}<br>값: %{r}<extra></extra>'
        )
    return objects.items()

def trace_consensus(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    Company Guide Consensus 차트 요소
    :param df: 컨센서스 데이터프레임
    :return: dict() :: Key ['목표주가', '종가']
    """
    require = ['목표주가', '종가']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for consensus data')

    objects = dict()
    for col in require:
        sr = df[col].dropna()
        objects[col] = go.Scatter(
            name=col, x=sr.index, y=sr.astype(int),
            visible=True, showlegend=False if col == '종가' else True,
            meta=reform(df.index),
            hovertemplate='날짜: %{meta}<br>' + col + ': %{y:,}원<extra></extra>'
        )
    return objects.items()

def trace_foreigners(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    Company Guide 외국인 비중 차트 요소
    :param df: 외국인 비중 데이터프레임
    :return: dict() :: Key ['종가', '외국인보유비중']
    """
    require = ['종가', '외국인보유비중']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for foreigner data')

    objects = dict()
    for col in df.columns:
        flag_price = col.startswith('종가')
        form = ': %{y:,}원' if flag_price else ': %{y}%'
        objects[col] = go.Scatter(
            name=col + '(지분율)' if flag_price else col, x=df.index, y=df[col].astype(int if flag_price else float),
            visible=True, showlegend=False if col == '종가' else True,
            meta=reform(df.index),
            hovertemplate='날짜: %{meta}<br>' + col + form + '<extra></extra>'
        )
    return objects.items()

def trace_shorts(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    Company Guide 차입 공매도 비중 차트 요소
    :param df: 차입공매도 비중 데이터프레임
    :return: dict() :: Key ['차입공매도비중', '수정 종가']
    """
    require = ['수정 종가', '차입공매도비중']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for shorts data')

    objects = dict()
    for col in df.columns:
        is_price = col.endswith('종가')
        form = ': %{y:,}원' if is_price else ': %{y}%'
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col].astype(int if is_price else float),
            visible=True, showlegend=False if col.endswith('종가') else True,
            meta=reform(df.index),
            hovertemplate='날짜: %{meta}<br>' + col + form + '<extra></extra>'
        )
    return objects.items()

def trace_balance(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    Company Guide 대차잔고비중 차트 요소
    :param df: 대차잔고 비중 데이터프레임
    :return: dict() :: key ['대차잔고비중', '수정 종가']
    """
    require = ['수정 종가', '대차잔고비중']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for loan balance data')

    objects = dict()
    for col in df.columns:
        is_price = col.endswith('종가')
        form = ': %{y:,}원' if is_price else ': %{y}%'
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col].astype(int if is_price else float),
            visible=True, showlegend=False if col.endswith('종가') else True,
            meta=reform(df.index),
            hovertemplate='날짜: %{meta}<br>' + col + form + '<extra></extra>'
        )
    return objects.items()