"""
시계열 예측 분석 대시보드
Time Series Forecasting & Trust Assessment Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ══════════════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="시계열 예측 신뢰 분석기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08); border-left: 4px solid #667eea;
        margin-bottom: 0.7rem;
    }
    .metric-card.green  { border-left-color: #10b981; }
    .metric-card.yellow { border-left-color: #f59e0b; }
    .metric-card.red    { border-left-color: #ef4444; }
    .trust-score {
        font-size: 3.5rem; font-weight: 900;
        text-align: center; margin: 1rem 0;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #374151;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════════════════════════

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def safe_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def calc_mase(y_true, y_pred, y_train):
    y_true  = np.array(y_true,  dtype=float)
    y_pred  = np.array(y_pred,  dtype=float)
    y_train = np.array(y_train, dtype=float)
    naive_errors = np.abs(np.diff(y_train))
    scale = naive_errors.mean() if len(naive_errors) > 0 and naive_errors.mean() != 0 else 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def compute_metrics(y_true, y_pred, y_train, model_fit=None):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape  = safe_mape(y_true, y_pred)
    smape = safe_smape(y_true, y_pred)
    mase  = calc_mase(y_true, y_pred, y_train)
    aic, bic = np.nan, np.nan
    if model_fit is not None:
        try: aic = float(model_fit.aic)
        except: pass
        try: bic = float(model_fit.bic)
        except: pass
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, sMAPE=smape,
                MASE=mase, AIC=aic, BIC=bic)

def ljung_box_pvalue(residuals, lags=10):
    # [버그수정] 잔차 개수 < lags 이면 lags 자동 축소
    try:
        res = np.array(residuals, dtype=float)
        res = res[~np.isnan(res)]
        if len(res) < 4:
            return np.nan
        actual_lags = min(lags, len(res) // 2)
        if actual_lags < 1:
            return np.nan
        result = acorr_ljungbox(res, lags=[actual_lags], return_df=True)
        return float(result['lb_pvalue'].iloc[0])
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════
# 날짜/빈도 탐지
# ══════════════════════════════════════════════════════════════════

def try_parse_datetime(series):
    """
    [버그수정] infer_datetime_format 완전 제거 (pandas 2.2에서 TypeError 발생).
    errors='coerce' 방식 + 주요 포맷 순차 시도.
    """
    # 방법 1: pandas 자동 파싱
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().mean() > 0.8:
            return parsed, float(parsed.notna().mean())
    except Exception:
        pass

    # 방법 2: 자주 쓰이는 포맷 직접 시도
    formats = [
        '%Y-%m', '%Y/%m', '%Y.%m',
        '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
        '%m/%d/%Y', '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S',
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            if parsed.notna().mean() > 0.8:
                return parsed, float(parsed.notna().mean())
        except Exception:
            continue

    return None, 0.0

def auto_detect_columns(df):
    """
    [버그수정] 수치형 열이 날짜로 오탐지되는 문제 방지.
    순수 숫자 열은 날짜 후보에서 제외.
    """
    date_col     = None
    numeric_cols = []

    for col in df.columns:
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            numeric_cols.append(col)
            continue
        if date_col is None:
            parsed, rate = try_parse_datetime(col_data.astype(str))
            if parsed is not None and rate > 0.8:
                date_col = col
            else:
                try:
                    converted = pd.to_numeric(col_data, errors='coerce')
                    if converted.notna().mean() > 0.5:
                        numeric_cols.append(col)
                except Exception:
                    pass

    return date_col, numeric_cols

def load_and_validate(df, date_col, val_col):
    """
    [버그수정]
    - infer_datetime_format 제거 → try_parse_datetime 사용
    - infer_freq 실패 시 None 안전 처리
    - interpolate/ffill/bfill 체인 분리
    """
    df = df[[date_col, val_col]].copy()

    parsed, _ = try_parse_datetime(df[date_col].astype(str))
    if parsed is None:
        raise ValueError(f"'{date_col}' 열을 날짜로 변환할 수 없습니다.")
    df[date_col] = parsed

    df = df.dropna(subset=[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)

    dup_count = int(df.duplicated(subset=[date_col]).sum())
    df = df.groupby(date_col, as_index=False)[val_col].mean()
    df = df.set_index(date_col)

    # [버그수정] infer_freq가 None 반환해도 안전하게 처리
    try:
        inferred = pd.infer_freq(df.index)
        if inferred:
            df.index.freq = inferred
    except Exception:
        pass

    missing_before = int(df[val_col].isna().sum())
    df[val_col] = df[val_col].interpolate(method='linear')
    df[val_col] = df[val_col].ffill()
    df[val_col] = df[val_col].bfill()

    return df, dup_count, missing_before

def detect_freq(index):
    """
    [버그수정] .days 속성 대신 total_seconds() / 86400 으로 일수 계산
    """
    if len(index) < 2:
        return 'Monthly', 'MS', 12
    try:
        diffs = pd.Series(index).diff().dropna()
        if len(diffs) == 0:
            return 'Monthly', 'MS', 12
        secs  = diffs.apply(lambda x: x.total_seconds())
        days  = secs.median() / 86400
        if days <= 1.5:
            return 'Daily',     'D',  7
        elif days <= 8:
            return 'Weekly',    'W',  52
        elif days <= 35:
            return 'Monthly',   'MS', 12
        elif days <= 100:
            return 'Quarterly', 'QS', 4
        else:
            return 'Yearly',    'YS', 1
    except Exception:
        return 'Monthly', 'MS', 12

def generate_future_index(last_date, h, freq_str):
    freq_map = {'D': 'D', 'W': 'W', 'MS': 'MS', 'QS': 'QS', 'YS': 'YS'}
    freq = freq_map.get(freq_str, 'MS')
    try:
        return pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]
    except Exception:
        return pd.date_range(start=last_date, periods=h + 1, freq='MS')[1:]


# ══════════════════════════════════════════════════════════════════
# 통계 분석
# ══════════════════════════════════════════════════════════════════

def run_adf(series):
    try:
        clean = series.dropna()
        if len(clean) < 8:
            return dict(statistic=np.nan, pvalue=np.nan,
                        critical={'1%': np.nan, '5%': np.nan, '10%': np.nan},
                        is_stationary=False)
        result = adfuller(clean, autolag='AIC')
        return dict(statistic=float(result[0]), pvalue=float(result[1]),
                    critical=result[4], is_stationary=bool(result[1] < 0.05))
    except Exception:
        return dict(statistic=np.nan, pvalue=np.nan,
                    critical={'1%': np.nan, '5%': np.nan, '10%': np.nan},
                    is_stationary=False)

def get_decomposition(series, period):
    try:
        if len(series) < period * 2:
            return None
        return seasonal_decompose(series, model='additive', period=period,
                                  extrapolate_trend='freq')
    except Exception:
        return None

def auto_arima_order(series, d_val):
    """
    [버그수정] 차분 후 시리즈 길이 부족 방어 + nlags 안전 상한 설정
    """
    try:
        diff_s = series.dropna()
        for _ in range(d_val):
            diff_s = diff_s.diff().dropna()
        if len(diff_s) < 8:
            return (1, d_val, 1)
        max_lags = min(10, len(diff_s) // 3)
        if max_lags < 2:
            return (1, d_val, 1)
        acf_r  = acf(diff_s,  nlags=max_lags, alpha=0.05)
        pacf_r = pacf(diff_s, nlags=max_lags, alpha=0.05)
        acf_arr  = acf_r[0][1:]
        pacf_arr = pacf_r[0][1:]
        conf = 1.96 / np.sqrt(len(diff_s))
        p = next((i + 1 for i, v in enumerate(pacf_arr) if abs(v) < conf), 1)
        q = next((i + 1 for i, v in enumerate(acf_arr)  if abs(v) < conf), 1)
        return (min(p, 3), d_val, min(q, 3))
    except Exception:
        return (1, d_val, 1)


# ══════════════════════════════════════════════════════════════════
# 모델 훈련
# ══════════════════════════════════════════════════════════════════

def naive_forecast(train, h):
    return np.array([float(train.iloc[-1])] * h), None

def moving_average_forecast(train, h, window=4):
    w = min(window, len(train))
    return np.array([float(train.iloc[-w:].mean())] * h), None

def ses_forecast(train, h):
    try:
        m = SimpleExpSmoothing(train, initialization_method='estimated').fit(optimized=True)
        return m.forecast(h).values.astype(float), m
    except Exception:
        return naive_forecast(train, h)

def holt_forecast(train, h):
    try:
        m = ExponentialSmoothing(train, trend='add',
                                 initialization_method='estimated').fit(optimized=True)
        return m.forecast(h).values.astype(float), m
    except Exception:
        return ses_forecast(train, h)

def holtwinters_forecast(train, h, sp):
    if len(train) < sp * 2 + 1:
        return holt_forecast(train, h)
    try:
        m = ExponentialSmoothing(train, trend='add', seasonal='add',
                                 seasonal_periods=sp,
                                 initialization_method='estimated').fit(optimized=True)
        return m.forecast(h).values.astype(float), m
    except Exception:
        return holt_forecast(train, h)

def arima_forecast(train, h, order=(1, 1, 1)):
    # [버그수정] 여러 차수를 순서대로 시도하는 폴백 체인
    for o in [order, (1, 1, 1), (0, 1, 1), (1, 1, 0), (0, 1, 0)]:
        try:
            fit = ARIMA(train, order=o).fit()
            return fit.forecast(h).values.astype(float), fit
        except Exception:
            continue
    return naive_forecast(train, h)


# ══════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════

COLORS = {
    'Naive':       '#94a3b8',
    'MovAvg':      '#64748b',
    'SES':         '#3b82f6',
    'Holt':        '#8b5cf6',
    'HoltWinters': '#f59e0b',
    'ARIMA':       '#10b981',
}

def plot_raw(series, val_col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series[val_col],
                             mode='lines', name='원본 데이터',
                             line=dict(color='#667eea', width=2)))
    fig.update_layout(title='원본 시계열 데이터', xaxis_title='날짜',
                      yaxis_title=val_col, height=350,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig

def plot_decomp(decomp):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=['원본 (Observed)', '추세 (Trend)',
                                        '계절성 (Seasonal)', '잔차 (Residual)'])
    for row, (comp, color) in enumerate([
        (decomp.observed, '#667eea'), (decomp.trend,    '#f59e0b'),
        (decomp.seasonal, '#10b981'), (decomp.resid,    '#ef4444'),
    ], start=1):
        fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode='lines',
                                 showlegend=False, line=dict(color=color, width=1.5)),
                      row=row, col=1)
    fig.update_layout(height=680, title='시계열 분해 (가법 모형)',
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig

def plot_acf_pacf(series, lags=24):
    # [버그수정] 데이터 길이에 따라 lags 자동 조정
    try:
        n = len(series.dropna())
        safe_lags = min(lags, n // 3)
        if safe_lags < 4:
            return None
        acf_r  = acf(series.dropna(),  nlags=safe_lags, alpha=0.05)
        pacf_r = pacf(series.dropna(), nlags=safe_lags, alpha=0.05)
        acf_v, pacf_v = acf_r[0], pacf_r[0]
        conf = 1.96 / np.sqrt(n)
        fig = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
        for ci, (vals, name) in enumerate([(acf_v, 'ACF'), (pacf_v, 'PACF')], start=1):
            fig.add_trace(go.Bar(x=list(range(len(vals))), y=vals,
                                 marker_color='#667eea', showlegend=False),
                          row=1, col=ci)
            for sign in [1, -1]:
                fig.add_hline(y=sign * conf, line_dash='dash',
                              line_color='red', opacity=0.6, row=1, col=ci)
        fig.update_layout(height=320, title='ACF / PACF (자기상관 분석)',
                          margin=dict(l=40, r=20, t=50, b=40))
        return fig
    except Exception:
        return None

def plot_forecast(series, val_col, test_results, future_results, best_model, train_end):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series[val_col],
                             name='실제 데이터', mode='lines',
                             line=dict(color='#1e293b', width=2.5)))
    for name, info in test_results.items():
        is_best = (name == best_model)
        fig.add_trace(go.Scatter(
            x=info['index'], y=info['pred'],
            name=f'{name} (검증)', mode='lines',
            line=dict(color=COLORS.get(name, '#888'),
                      dash='solid' if is_best else 'dot',
                      width=2.5 if is_best else 1.2),
            opacity=1.0 if is_best else 0.6,
        ))
    if best_model and best_model in future_results:
        fi = future_results[best_model]
        fig.add_trace(go.Scatter(
            x=fi['index'], y=fi['pred'],
            name=f'{best_model} (미래 예측)', mode='lines+markers',
            line=dict(color=COLORS.get(best_model, '#667eea'), width=3, dash='dash'),
            marker=dict(size=5),
        ))
        if len(fi['index']) > 0:
            fig.add_vrect(x0=str(fi['index'][0]), x1=str(fi['index'][-1]),
                          fillcolor='rgba(102,126,234,0.07)',
                          layer='below', line_width=0,
                          annotation_text='예측 구간', annotation_position='top left')
    fig.add_vline(x=int(train_end.timestamp() * 1000), line_dash='dash', line_color='gray', opacity=0.8,
                  annotation_text='훈련 | 테스트', annotation_position='top right')
    fig.update_layout(title='예측 결과 대시보드', height=480,
                      xaxis_title='날짜', yaxis_title=val_col,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1),
                      margin=dict(l=40, r=20, t=70, b=40))
    return fig

def plot_residuals(residuals, model_name):
    res = np.array(residuals, dtype=float)
    res = res[~np.isnan(res)]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['잔차 시계열', '잔차 히스토그램', 'QQ Plot'])
    fig.add_trace(go.Scatter(y=res, mode='lines+markers',
                             marker=dict(size=4), line=dict(color='#667eea'),
                             showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.7, row=1, col=1)
    fig.add_trace(go.Histogram(x=res, nbinsx=20, marker_color='#764ba2',
                               showlegend=False), row=1, col=2)
    # QQ Plot
    try:
        from scipy.stats import norm as sp_norm
        sorted_res = np.sort(res)
        n = len(sorted_res)
        p = np.array([(i - 0.5) / n for i in range(1, n + 1)])
        q_th = sp_norm.ppf(p)
        fig.add_trace(go.Scatter(x=q_th, y=sorted_res, mode='markers',
                                 marker=dict(color='#10b981', size=5),
                                 showlegend=False), row=1, col=3)
        slope = np.std(sorted_res)
        mu    = np.mean(sorted_res)
        fig.add_trace(go.Scatter(x=[q_th[0], q_th[-1]],
                                 y=[slope * q_th[0] + mu, slope * q_th[-1] + mu],
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 showlegend=False), row=1, col=3)
    except Exception:
        pass
    fig.update_layout(title=f'잔차 진단 — {model_name}', height=320,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig

def plot_model_comparison(metrics_df, metric='RMSE'):
    if metric not in metrics_df.columns:
        return go.Figure()
    vals  = metrics_df[metric].values
    valid = [v for v in vals if not np.isnan(v)]
    if not valid:
        return go.Figure()
    min_v  = min(valid)
    colors = ['#10b981' if (not np.isnan(v) and v == min_v) else '#667eea' for v in vals]
    fig = go.Figure(go.Bar(
        x=metrics_df.index.tolist(), y=vals, marker_color=colors,
        text=[f'{v:.3f}' if not np.isnan(v) else 'N/A' for v in vals],
        textposition='outside',
    ))
    fig.update_layout(title=f'모델별 {metric} (낮을수록 좋음)',
                      yaxis_title=metric, height=360,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


# ══════════════════════════════════════════════════════════════════
# 예측 신뢰 리포트
# ══════════════════════════════════════════════════════════════════

def compute_trust_report(n, missing_before, adf_result, metrics_df,
                         best_model, horizon, seasonal_period, residuals):
    scores, reasons = {}, {}

    # 1. 데이터 품질
    missing_pct = missing_before / max(n, 1) * 100
    q = 100
    if n < 24:          q -= 20
    if n < 12:          q -= 20
    if missing_pct > 5:  q -= 15
    if missing_pct > 20: q -= 20
    scores['데이터 품질']  = max(0, q)
    reasons['데이터 품질'] = f"총 {n}개 데이터 | 결측률 {missing_pct:.1f}%"

    # 2. 정상성 (ADF)
    if np.isnan(adf_result['pvalue']):
        scores['정상성'] = 60
        reasons['정상성'] = "ADF 검정 실행 불가 (데이터 부족)"
    elif adf_result['is_stationary']:
        scores['정상성'] = 90
        reasons['정상성'] = f"ADF p={adf_result['pvalue']:.4f} → 정상 시계열 ✅"
    else:
        scores['정상성'] = 65
        reasons['정상성'] = f"ADF p={adf_result['pvalue']:.4f} → 비정상 (차분 적용) △"

    # 3. Naive 대비 개선
    if (best_model and 'Naive' in metrics_df.index
            and best_model in metrics_df.index):
        nr = metrics_df.loc['Naive', 'RMSE']
        br = metrics_df.loc[best_model, 'RMSE']
        if nr > 0 and not np.isnan(nr) and not np.isnan(br):
            imp = (nr - br) / nr * 100
            if imp > 20:
                scores['Naive 대비'] = 95
                reasons['Naive 대비'] = f"{best_model}이 Naive보다 {imp:.1f}% 정확 ✅"
            elif imp > 0:
                scores['Naive 대비'] = 75
                reasons['Naive 대비'] = f"{best_model}이 Naive보다 {imp:.1f}% 개선 △"
            else:
                scores['Naive 대비'] = 40
                reasons['Naive 대비'] = f"⚠️ {best_model}이 Naive보다 나쁨"
        else:
            scores['Naive 대비'] = 70
            reasons['Naive 대비'] = "비교 불가"
    else:
        scores['Naive 대비'] = 70
        reasons['Naive 대비'] = "비교 정보 없음"

    # 4. 잔차 독립성 (Ljung-Box)
    lb_pval = ljung_box_pvalue(residuals)
    if np.isnan(lb_pval):
        scores['잔차 독립성'] = 65
        reasons['잔차 독립성'] = "Ljung-Box 검정 불가 (데이터 부족)"
    elif lb_pval > 0.05:
        scores['잔차 독립성'] = 90
        reasons['잔차 독립성'] = f"Ljung-Box p={lb_pval:.3f} → 잔차 = 백색잡음 ✅"
    else:
        scores['잔차 독립성'] = 45
        reasons['잔차 독립성'] = f"⚠️ Ljung-Box p={lb_pval:.3f} → 잔차에 패턴 남음"

    # 5. 예측 기간
    ratio = horizon / max(n, 1)
    if ratio <= 0.15:
        scores['예측 기간'] = 95
        reasons['예측 기간'] = f"{horizon}스텝 (데이터의 {ratio*100:.0f}%) → 적절 ✅"
    elif ratio <= 0.3:
        scores['예측 기간'] = 75
        reasons['예측 기간'] = f"{horizon}스텝 (데이터의 {ratio*100:.0f}%) → 다소 김 △"
    elif ratio <= 0.5:
        scores['예측 기간'] = 55
        reasons['예측 기간'] = f"{horizon}스텝 (데이터의 {ratio*100:.0f}%) → 주의 ⚠️"
    else:
        scores['예측 기간'] = 30
        reasons['예측 기간'] = f"⚠️ {horizon}스텝이 너무 김 — 신뢰도 급감"

    total = float(np.mean(list(scores.values())))
    return total, scores, reasons

def trust_color(score):
    if score >= 75:
        return '#10b981', '🟢 신뢰 가능'
    elif score >= 55:
        return '#f59e0b', '🟡 조건부 신뢰'
    else:
        return '#ef4444', '🔴 주의 필요'


# ══════════════════════════════════════════════════════════════════
# 샘플 데이터
# ══════════════════════════════════════════════════════════════════

def get_airpassengers():
    idx  = pd.date_range('1949-01', periods=144, freq='MS')
    vals = [
        112,118,132,129,121,135,148,148,136,119,104,118,
        115,126,141,135,125,149,170,170,158,133,114,140,
        145,150,178,163,172,178,199,199,184,162,146,166,
        171,180,193,181,183,218,230,242,209,191,172,194,
        196,196,236,235,229,243,264,272,237,211,180,201,
        204,188,235,227,234,264,302,293,259,229,203,229,
        242,233,267,269,270,315,364,347,312,274,237,278,
        284,277,317,313,318,374,413,405,355,306,271,306,
        315,301,356,348,355,422,465,467,404,347,305,336,
        340,318,362,348,363,435,491,505,404,359,310,337,
        360,342,406,396,420,472,548,559,463,407,362,405,
        417,391,419,461,472,535,622,606,508,461,390,432,
    ]
    return pd.DataFrame({'Month': idx, 'Passengers': vals})


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-title">📈 시계열 예측 신뢰 분석기</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">단순한 AI 예측을 넘어 — 예측을 믿어도 되는지 판단해 드립니다</div>',
        unsafe_allow_html=True,
    )

    # ── 사이드바 ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")
        data_mode = st.radio("데이터 소스", ["CSV 업로드", "샘플 데이터 (AirPassengers)"])
        uploaded  = None
        if data_mode == "CSV 업로드":
            uploaded = st.file_uploader("CSV 파일 업로드", type=['csv'])

        st.divider()
        st.subheader("모델 설정")
        forecast_horizon = st.slider("예측 기간 (스텝)", 1, 60, 12)
        test_ratio       = st.slider("테스트 비율 (%)", 10, 40, 20)
        use_auto_arima   = st.checkbox("ARIMA 차수 자동 추정", value=True)

        st.divider()
        st.subheader("📚 강의 개념 맵")
        st.markdown("""
        - ✅ 데이터 수집 & 전처리
        - ✅ 결측값 처리 / 중복 제거
        - ✅ ADF 정상성 검정
        - ✅ ACF / PACF
        - ✅ 시계열 분해 (가법 모형)
        - ✅ SES / Holt / Holt-Winters
        - ✅ ARIMA
        - ✅ 예측 평가 지표
        - ✅ 잔차 진단 (Ljung-Box)
        """)

    # ── 데이터 로드 ─────────────────────────────────────────────
    df_raw = None
    date_col, numeric_cols, val_col = None, [], None

    if data_mode == "샘플 데이터 (AirPassengers)":
        df_raw       = get_airpassengers()
        date_col     = 'Month'
        numeric_cols = ['Passengers']
        val_col      = 'Passengers'
        st.sidebar.success("✅ AirPassengers 로드 완료")

    elif uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            date_col, numeric_cols = auto_detect_columns(df_raw)
        except Exception as e:
            st.error(f"❌ CSV 읽기 오류: {e}")
            st.stop()

        if not numeric_cols:
            st.error("❌ 수치형 열이 없습니다.")
            st.stop()
        val_col = numeric_cols[0]

    else:
        st.info("👈 왼쪽에서 CSV를 업로드하거나 샘플 데이터를 선택하세요.")
        st.markdown("""
        ### 🚀 이 앱이란?
        단변량 시계열 CSV를 업로드하면 자동으로:

        1. **데이터 품질 진단** — 결측값, 중복, 빈도 감지
        2. **ADF 정상성 검정** — 차분 필요 여부 자동 판단
        3. **시계열 분해** — 추세 / 계절성 / 잔차 분리
        4. **6개 모델 동시 학습** — Naive / MovAvg / SES / Holt / HW / ARIMA
        5. **예측 신뢰 리포트** — 5가지 기준으로 0~100점 채점
        """)
        return

    # ── 열 선택 UI ──────────────────────────────────────────────
    with st.expander("🔧 열 설정 확인 / 변경", expanded=(date_col is None)):
        all_cols = list(df_raw.columns)
        c1, c2 = st.columns(2)
        with c1:
            di = all_cols.index(date_col) if date_col in all_cols else 0
            date_col = st.selectbox("날짜 열 선택", all_cols, index=di)
        with c2:
            val_cands = [c for c in all_cols if c != date_col]
            vi = val_cands.index(val_col) if val_col in val_cands else 0
            val_col = st.selectbox("값 열 선택 (예측 대상)", val_cands, index=vi)

    # ── 전처리 ──────────────────────────────────────────────────
    try:
        series, dup_count, missing_before = load_and_validate(df_raw, date_col, val_col)
    except Exception as e:
        st.error(f"❌ 전처리 오류: {e}")
        st.stop()

    n = len(series)
    if n < 16:
        st.error(f"❌ 데이터가 너무 적습니다 (현재 {n}행, 최소 16행 필요).")
        st.stop()

    freq_label, freq_str, seasonal_period = detect_freq(series.index)
    adf_result  = run_adf(series[val_col])
    d_val       = 1 if not adf_result['is_stationary'] else 0
    arima_order = auto_arima_order(series[val_col], d_val) if use_auto_arima else (1, d_val, 1)
    decomp      = get_decomposition(series[val_col], seasonal_period)

    test_n  = max(4, int(n * test_ratio / 100))
    train_n = n - test_n
    train_s = series[val_col].iloc[:train_n]
    test_s  = series[val_col].iloc[train_n:]
    train_end = series.index[train_n - 1]

    # ── 모델 훈련 ────────────────────────────────────────────────
    with st.spinner("🔄 모델 훈련 중..."):
        runners = {
            'Naive':       lambda: naive_forecast(train_s, test_n),
            'MovAvg':      lambda: moving_average_forecast(train_s, test_n),
            'SES':         lambda: ses_forecast(train_s, test_n),
            'Holt':        lambda: holt_forecast(train_s, test_n),
            'HoltWinters': lambda: holtwinters_forecast(train_s, test_n, seasonal_period),
            'ARIMA':       lambda: arima_forecast(train_s, test_n, arima_order),
        }
        test_results = {}
        all_metrics  = {}
        best_model   = None
        best_rmse    = np.inf

        for name, fn in runners.items():
            try:
                pred, fit = fn()
                pred = np.array(pred, dtype=float)
                test_results[name] = {'pred': pred, 'index': test_s.index}
                m = compute_metrics(test_s.values, pred, train_s.values, fit)
                all_metrics[name] = m
                if not np.isnan(m['RMSE']) and m['RMSE'] < best_rmse:
                    best_rmse  = m['RMSE']
                    best_model = name
            except Exception as err:
                st.warning(f"⚠️ {name} 실패: {err}")

        metrics_df = pd.DataFrame(all_metrics).T if all_metrics else pd.DataFrame()

        future_runners = {
            'Naive':       lambda: naive_forecast(series[val_col], forecast_horizon),
            'SES':         lambda: ses_forecast(series[val_col], forecast_horizon),
            'Holt':        lambda: holt_forecast(series[val_col], forecast_horizon),
            'HoltWinters': lambda: holtwinters_forecast(series[val_col], forecast_horizon, seasonal_period),
            'ARIMA':       lambda: arima_forecast(series[val_col], forecast_horizon, arima_order),
        }
        future_idx    = generate_future_index(series.index[-1], forecast_horizon, freq_str)
        future_results = {}
        for name, fn in future_runners.items():
            try:
                pf, _ = fn()
                future_results[name] = {'pred': np.array(pf, dtype=float), 'index': future_idx}
            except Exception:
                pass

        residuals = np.array([])
        if best_model and best_model in test_results:
            residuals = test_s.values - test_results[best_model]['pred']

        trust_total, trust_scores, trust_reasons = compute_trust_report(
            n, missing_before, adf_result, metrics_df,
            best_model, forecast_horizon, seasonal_period, residuals
        )

    # ── KPI 바 ──────────────────────────────────────────────────
    st.divider()
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("데이터 수", n)
    with k2: st.metric("감지된 빈도", freq_label)
    with k3: st.metric("계절 주기", seasonal_period)
    with k4: st.metric("최적 모델", best_model or "없음")
    with k5:
        tc, tl = trust_color(trust_total)
        st.metric("예측 신뢰 점수", f"{trust_total:.0f}/100", tl)
    st.divider()

    # ── 탭 ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 데이터 리포트", "🔬 분해 & 진단", "📈 예측 개요",
        "🏆 모델 비교", "🛡️ 예측 신뢰 리포트", "💾 다운로드",
    ])

    # ── 탭 1: 데이터 리포트 ─────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">🔍 자동 분석 요약</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card {'green' if missing_before==0 else 'yellow'}">
            <b>결측값 처리</b><br>발견 {missing_before}개 → 선형 보간으로 대체
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card {'green' if dup_count==0 else 'yellow'}">
            <b>중복 타임스탬프</b><br>{dup_count}개 → 평균으로 집계
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card green">
            <b>날짜 범위</b><br>{series.index[0].date()} ~ {series.index[-1].date()}
            </div>""", unsafe_allow_html=True)

        st.plotly_chart(plot_raw(series, val_col), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">📊 기술통계</div>',
                        unsafe_allow_html=True)
            st.dataframe(series[val_col].describe().to_frame("값").style.format("{:.3f}"),
                         use_container_width=True)
        with c2:
            st.markdown('<div class="section-header">🔬 ADF 정상성 검정</div>',
                        unsafe_allow_html=True)
            if np.isnan(adf_result['pvalue']):
                st.warning("데이터 부족으로 ADF 검정 불가")
            else:
                clr = 'green' if adf_result['is_stationary'] else 'yellow'
                msg = "✅ 정상 시계열" if adf_result['is_stationary'] else "⚠️ 비정상 시계열"
                st.markdown(f"""<div class="metric-card {clr}">
                <b>결론: {msg}</b><br>
                검정 통계량: {adf_result['statistic']:.4f}<br>
                p-value: {adf_result['pvalue']:.4f}<br>
                임계값 (5%): {adf_result['critical'].get('5%', 'N/A')}
                </div>""", unsafe_allow_html=True)
                if adf_result['is_stationary']:
                    st.info("p < 0.05 → 정상 시계열. 평균과 분산이 시간에 따라 안정적입니다.")
                else:
                    st.info(f"p > 0.05 → 비정상 시계열.\nARIMA에서 d={d_val} 차분이 자동 적용됩니다.")

        st.markdown(f"""<div class="metric-card">
        <b>🤖 ARIMA 추천 차수: {arima_order}</b><br>
        &nbsp; p={arima_order[0]}: PACF 기반 AR 차수 &nbsp;|&nbsp;
        d={arima_order[1]}: ADF 기반 차분 횟수 &nbsp;|&nbsp;
        q={arima_order[2]}: ACF 기반 MA 차수
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📋 데이터 미리보기 (상위 20행)</div>',
                    unsafe_allow_html=True)
        st.dataframe(series.head(20), use_container_width=True)

    # ── 탭 2: 분해 & 진단 ───────────────────────────────────────
    with tab2:
        if decomp is not None:
            st.plotly_chart(plot_decomp(decomp), use_container_width=True)
            resid_var    = float(np.var(decomp.resid.dropna()))
            trend_var    = float(np.var(decomp.trend.dropna()))
            seasonal_var = float(np.var(decomp.seasonal.dropna()))
            ts = max(0.0, 1 - resid_var / (trend_var    + resid_var + 1e-9))
            ss = max(0.0, 1 - resid_var / (seasonal_var + resid_var + 1e-9))
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("추세 강도",   f"{ts:.1%}")
            with c2: st.metric("계절성 강도", f"{ss:.1%}")
            with c3: st.metric("계절 주기",   f"{seasonal_period} 스텝")
            if ss > 0.4:
                st.success(f"**계절성 강함** ({ss:.1%}) → Holt-Winters 또는 SARIMA 권장")
            else:
                st.info("계절성 약함 → Holt 또는 ARIMA가 적합할 수 있습니다.")
        else:
            st.warning(f"시계열 분해 불가: 데이터({n}개) < 계절주기({seasonal_period})×2")

        st.markdown('<div class="section-header">📉 ACF / PACF</div>',
                    unsafe_allow_html=True)
        acf_fig = plot_acf_pacf(series[val_col])
        if acf_fig:
            st.plotly_chart(acf_fig, use_container_width=True)
            st.caption("ACF → MA 차수(q) | PACF → AR 차수(p) | 빨간 점선 = 95% 신뢰구간")
        else:
            st.warning("데이터 부족으로 ACF/PACF 계산 불가")

        if not adf_result['is_stationary'] and not np.isnan(adf_result['pvalue']):
            st.markdown('<div class="section-header">📊 1차 차분 시계열</div>',
                        unsafe_allow_html=True)
            diff_s   = series[val_col].diff().dropna()
            diff_adf = run_adf(diff_s)
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=diff_s.index, y=diff_s.values,
                                       mode='lines', line=dict(color='#f59e0b', width=1.5)))
            fig_d.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
            fig_d.update_layout(title='1차 차분 시계열', height=260,
                                 margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig_d, use_container_width=True)
            if diff_adf['is_stationary']:
                st.success(f"✅ 1차 차분 후 정상 (p={diff_adf['pvalue']:.4f})")
            elif not np.isnan(diff_adf['pvalue']):
                st.warning(f"⚠️ 1차 차분 후에도 비정상 (p={diff_adf['pvalue']:.4f})")

    # ── 탭 3: 예측 개요 ─────────────────────────────────────────
    with tab3:
        if forecast_horizon > n * 0.5:
            st.warning(f"⚠️ 예측 {forecast_horizon}스텝 > 데이터({n}개)의 50% — 장기 예측은 불확실합니다.")
        if forecast_horizon > seasonal_period * 2:
            st.warning(f"⚠️ 예측 기간이 계절 주기({seasonal_period})의 2배 초과.")
        st.info(f"📅 1스텝 = 1{freq_label} | 예측 기간 {forecast_horizon}스텝 | 최적 모델: **{best_model}**")

        if test_results:
            st.plotly_chart(
                plot_forecast(series, val_col, test_results, future_results, best_model, train_end),
                use_container_width=True
            )
        else:
            st.error("모든 모델이 실패했습니다.")

        if best_model and best_model in future_results:
            st.markdown(f'<div class="section-header">📋 미래 예측값 — {best_model}</div>',
                        unsafe_allow_html=True)
            fi = future_results[best_model]
            st.dataframe(pd.DataFrame({'날짜': fi['index'],
                                       '예측값': np.round(fi['pred'], 2)}),
                         use_container_width=True)

        if len(residuals) >= 4:
            st.markdown(f'<div class="section-header">🔬 잔차 진단 — {best_model}</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(plot_residuals(residuals, best_model), use_container_width=True)
            lb_p = ljung_box_pvalue(residuals)
            if not np.isnan(lb_p):
                if lb_p > 0.05:
                    st.success(f"✅ Ljung-Box p={lb_p:.4f} — 잔차 = 백색잡음")
                else:
                    st.warning(f"⚠️ Ljung-Box p={lb_p:.4f} — 잔차에 패턴 남음")

    # ── 탭 4: 모델 비교 ─────────────────────────────────────────
    with tab4:
        st.markdown("> **MASE < 1** 이어야 Naive보다 의미 있는 예측입니다.")
        if not metrics_df.empty:
            disp_cols = [c for c in ['MAE','RMSE','MAPE','sMAPE','MASE','AIC','BIC']
                         if c in metrics_df.columns]
            disp = metrics_df[disp_cols].copy()

            def highlight_best(s):
                valid = s.dropna()
                if valid.empty: return ['' for _ in s]
                min_v = valid.min()
                return ['background-color:#d1fae5;font-weight:bold'
                        if (not np.isnan(v) and v == min_v) else '' for v in s]

            st.dataframe(
                disp.style.apply(highlight_best,
                                 subset=[c for c in ['MAE','RMSE','MAPE','sMAPE','MASE']
                                         if c in disp.columns])
                          .format("{:.4f}", na_rep="N/A"),
                use_container_width=True
            )
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_model_comparison(metrics_df, 'RMSE'),
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_model_comparison(metrics_df, 'MASE'),
                                use_container_width=True)

            if 'Naive' in metrics_df.index:
                st.markdown('<div class="section-header">📊 Naive 대비 개선율</div>',
                            unsafe_allow_html=True)
                nr = metrics_df.loc['Naive', 'RMSE']
                rows = []
                for m in metrics_df.index:
                    if m == 'Naive': continue
                    r   = metrics_df.loc[m, 'RMSE']
                    imp = (nr - r) / nr * 100 if nr > 0 and not np.isnan(r) else 0.0
                    rows.append({'모델': m, 'RMSE': r, 'Naive 대비 개선(%)': imp,
                                 '판정': '✅ 개선' if imp > 0 else '❌ Naive보다 나쁨'})
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows).set_index('모델').style.format(
                            {'RMSE': '{:.4f}', 'Naive 대비 개선(%)': '{:.2f}'}
                        ),
                        use_container_width=True
                    )

            with st.expander("📖 평가 지표 설명"):
                st.markdown("""
| 지표 | 의미 | 주의사항 |
|------|------|---------|
| **MAE** | 평균 절대 오차. 직관적, 이상치에 강건 | 스케일 의존 |
| **RMSE** | 제곱 오차 기반. 큰 오차에 민감 | 이상치 영향 큼 |
| **MAPE** | 퍼센트(%) 오차 | 0 근처 폭발 |
| **sMAPE** | 대칭 MAPE | 0값 주의 |
| **MASE** | Naive 대비 상대 오차 **< 1 = Naive보다 좋음** | 핵심 지표 |
| **AIC/BIC** | 정보기준, 낮을수록 좋음 | ARIMA 계열만 |
                """)
        else:
            st.error("모델 결과 없음")

    # ── 탭 5: 예측 신뢰 리포트 ─────────────────────────────────
    with tab5:
        tc, tl = trust_color(trust_total)
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem;
                    background:linear-gradient(135deg,#f8fafc,#e2e8f0);
                    border-radius:16px;margin-bottom:1rem;">
            <div style="font-size:0.95rem;color:#64748b;margin-bottom:0.2rem;">종합 예측 신뢰 점수</div>
            <div class="trust-score" style="color:{tc};">{trust_total:.0f}</div>
            <div style="font-size:1.1rem;font-weight:700;">{tl}</div>
            <div style="color:#64748b;font-size:0.88rem;margin-top:0.5rem;">
                최적 모델: <b>{best_model}</b> &nbsp;|&nbsp; 예측 기간: <b>{forecast_horizon}스텝</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔍 세부 평가</div>', unsafe_allow_html=True)
        for item, score in trust_scores.items():
            reason   = trust_reasons.get(item, '')
            cc       = 'green' if score >= 75 else ('yellow' if score >= 55 else 'red')
            bc       = '#10b981' if score >= 75 else ('#f59e0b' if score >= 55 else '#ef4444')
            st.markdown(f"""
            <div class="metric-card {cc}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <b>{item}</b>
                    <span style="font-size:1.3rem;font-weight:800;color:{bc};">{score:.0f}점</span>
                </div>
                <div style="color:#6b7280;font-size:0.88rem;margin-top:0.3rem;">{reason}</div>
                <div style="margin-top:0.5rem;background:#e5e7eb;border-radius:999px;height:6px;">
                    <div style="width:{min(score,100):.0f}%;background:{bc};border-radius:999px;height:6px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">💬 자동 자연어 해석</div>',
                    unsafe_allow_html=True)
        interps = []
        if trust_total >= 75:
            interps.append(f"✅ **이 예측은 신뢰할 수 있습니다.** {best_model} 모델이 주요 기준을 충족합니다.")
        elif trust_total >= 55:
            interps.append("⚠️ **조건부로 참고할 수 있습니다.** 일부 기준에서 경고가 있습니다.")
        else:
            interps.append("🔴 **신뢰하기 어렵습니다.** 데이터 보강 또는 모델 재검토가 필요합니다.")

        if not adf_result['is_stationary'] and not np.isnan(adf_result['pvalue']):
            interps.append(f"📌 **비정상 시계열**: d={d_val} 차분이 ARIMA에 적용되었습니다.")

        if best_model and best_model in metrics_df.index:
            mv = metrics_df.loc[best_model, 'MASE']
            if not np.isnan(mv):
                sym = "✅" if mv < 1 else "⚠️"
                interps.append(f"📌 **MASE={mv:.3f}** {'< 1 → Naive보다 정확' if mv < 1 else '≥ 1 → Naive보다 나쁨'} {sym}")

        lb_p = ljung_box_pvalue(residuals)
        if not np.isnan(lb_p):
            if lb_p > 0.05:
                interps.append(f"📌 **잔차 진단 통과** (p={lb_p:.3f}) ✅")
            else:
                interps.append(f"📌 **잔차에 패턴 남음** (p={lb_p:.3f}) — 모델 개선 여지 있음")

        if forecast_horizon > seasonal_period:
            interps.append(f"📌 **장기 예측 주의**: {forecast_horizon}스텝 > 계절주기({seasonal_period})")

        for interp in interps:
            st.markdown(interp)

        st.markdown('<div class="section-header">📚 강의 개념 연결</div>',
                    unsafe_allow_html=True)
        adf_str   = '정상' if adf_result['is_stationary'] else f'비정상→d={d_val}'
        decomp_str = '성공' if decomp else '데이터 부족'
        lb_str    = f'p={lb_p:.3f} ✅' if (not np.isnan(lb_p) and lb_p > 0.05) else '확인 필요'
        st.markdown(f"""
| 강의 개념 | 적용 내용 | 결과 |
|-----------|----------|------|
| **ADF 검정** | 정상성 판단 | {adf_str} |
| **시계열 분해** | 추세·계절·잔차 분리 | {decomp_str} |
| **ACF/PACF** | ARIMA 차수 추정 | p={arima_order[0]}, d={arima_order[1]}, q={arima_order[2]} |
| **SES** | 단순 지수 평활 | 훈련 완료 |
| **Holt** | 이중 지수 평활 | 훈련 완료 |
| **Holt-Winters** | 추세+계절성 | 훈련 완료 |
| **ARIMA** | 자기회귀 모델 | 차수 {arima_order} |
| **Naive 기준선** | 유용성 검증 | MASE 기준 비교 |
| **잔차 진단** | Ljung-Box | {lb_str} |
        """)

    # ── 탭 6: 다운로드 ──────────────────────────────────────────
    with tab6:
        st.subheader("💾 결과 다운로드")
        if best_model and best_model in future_results:
            fi = future_results[best_model]
            df_dl = pd.DataFrame({'date': fi['index'],
                                  f'forecast_{best_model}': np.round(fi['pred'], 4)})
            for nm, info in future_results.items():
                if nm != best_model:
                    df_dl[f'forecast_{nm}'] = np.round(info['pred'], 4)
            st.download_button("📥 미래 예측값 CSV",
                               data=df_dl.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f"forecast_{best_model}_{forecast_horizon}steps.csv",
                               mime='text/csv')

        if not metrics_df.empty:
            st.download_button("📥 모델 비교 CSV",
                               data=metrics_df.round(4).to_csv().encode('utf-8-sig'),
                               file_name="model_comparison.csv", mime='text/csv')

        st.download_button("📥 전처리된 데이터 CSV",
                           data=series.reset_index().to_csv(index=False).encode('utf-8-sig'),
                           file_name="cleaned_timeseries.csv", mime='text/csv')

        report_lines = [
            f"예측 신뢰 점수: {trust_total:.1f}/100  ({tl})",
            f"최적 모델: {best_model}",
            f"예측 기간: {forecast_horizon}스텝 ({freq_label})", "",
        ]
        for item, score in trust_scores.items():
            report_lines.append(f"{item}: {score:.0f}점  — {trust_reasons.get(item,'')}")
        st.download_button("📥 신뢰 리포트 TXT",
                           data="\n".join(report_lines).encode('utf-8'),
                           file_name="trust_report.txt", mime='text/plain')

    st.divider()
    st.caption("📈 시계열 예측 신뢰 분석기 | 단순 예측을 넘어 — 예측의 신뢰도를 판단합니다")


if __name__ == "__main__":
    main()
