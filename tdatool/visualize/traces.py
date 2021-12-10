from typing import ItemsView
import pandas as pd
import plotly.graph_objects as go


def reform(span) -> list:
    """
    날짜 형식 변경 (from)datetime --> (to)YY/MM/DD
    :param span: 날짜 리스트
    :return:
    """
    return [f'{d.year}/{d.month}/{d.day}' for d in span]

def price(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
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

def volume(df:pd.DataFrame, is_main:bool=False) -> ItemsView[str, go.Bar]:
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

def filters(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
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

def bollinger(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    볼린저밴드 차트 요소
    :param df: 볼린저밴드 데이터프레임
    :param group: 범례 그룹 여부
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

def macd(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    MACD 차트 요소
    :param df: MACD 데이터프레임
    :return:
    """
    objects = dict()

    return objects.items()

def pivot(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
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

def trend(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
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

def rsi(df:pd.DataFrame) -> ItemsView[str, go.Scatter]:
    """
    RSI 차트 요소
    :param df: RSI 데이터프레임
    :return:
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