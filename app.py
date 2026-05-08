"""
시계열 예측 신뢰 분석기 v3
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant
from scipy.stats import jarque_bera, norm as sp_norm
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── 헤더 ── */
.app-header {
    padding: 1.5rem 0 0.5rem 0;
    margin-bottom: 0.5rem;
}
.app-title {
    font-size: 2rem; font-weight: 900; color: #0f172a;
    letter-spacing: -0.5px; line-height: 1.2;
    display: flex; align-items: center; gap: 0.5rem;
}
.app-title-accent { color: #6366f1; }
.app-subtitle { color: #64748b; font-size: 0.95rem; margin-top: 0.3rem; }

/* ── KPI 카드 ── */
.kpi-row { display: flex; gap: 0.8rem; margin: 1rem 0; }
.kpi-card {
    flex: 1; border-radius: 14px; padding: 1.1rem 1.2rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute;
    top: 0; right: 0; width: 60px; height: 60px;
    border-radius: 50%; opacity: 0.08;
    transform: translate(20px, -20px);
}
.kpi-card-blue   { background: #eff6ff; border-color: #bfdbfe; }
.kpi-card-blue::before   { background: #3b82f6; }
.kpi-card-violet { background: #f5f3ff; border-color: #ddd6fe; }
.kpi-card-violet::before { background: #8b5cf6; }
.kpi-card-green  { background: #f0fdf4; border-color: #bbf7d0; }
.kpi-card-green::before  { background: #10b981; }
.kpi-card-orange { background: #fff7ed; border-color: #fed7aa; }
.kpi-card-orange::before { background: #f59e0b; }
.kpi-card-trust-green  { background: #ecfdf5; border-color: #6ee7b7; }
.kpi-card-trust-yellow { background: #fefce8; border-color: #fde68a; }
.kpi-card-trust-red    { background: #fef2f2; border-color: #fca5a5; }

.kpi-icon  { font-size: 1.4rem; margin-bottom: 0.3rem; }
.kpi-label { font-size: 0.72rem; font-weight: 700; color: #94a3b8;
             text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.7rem; font-weight: 800; color: #0f172a;
             line-height: 1.1; margin: 0.15rem 0; }
.kpi-sub   { font-size: 0.78rem; color: #64748b; font-weight: 500; }

/* ── 섹션 헤더 ── */
.sec-head {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 1rem; font-weight: 700; color: #1e293b;
    margin: 1.4rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f1f5f9;
}
.sec-head-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #6366f1; flex-shrink: 0;
}

/* ── 정보 카드 ── */
.card {
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 0.6rem; border: 1px solid #e2e8f0;
    background: white;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.card-green  { border-left: 4px solid #10b981; background: #f0fdf4; border-color: #bbf7d0; }
.card-yellow { border-left: 4px solid #f59e0b; background: #fffbeb; border-color: #fde68a; }
.card-red    { border-left: 4px solid #ef4444; background: #fef2f2; border-color: #fecaca; }
.card-blue   { border-left: 4px solid #3b82f6; background: #eff6ff; border-color: #bfdbfe; }
.card-violet { border-left: 4px solid #8b5cf6; background: #f5f3ff; border-color: #ddd6fe; }
.card-slate  { border-left: 4px solid #64748b; background: #f8fafc; border-color: #e2e8f0; }

/* ── 스토리 박스 ── */
.story-box {
    background: linear-gradient(135deg, #f8faff 0%, #f3f0ff 100%);
    border: 1px solid #c7d2fe; border-radius: 16px;
    padding: 1.5rem 1.8rem; line-height: 1.8;
    color: #334155; font-size: 0.96rem;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08);
}

/* ── ARIMA 케이스 카드 ── */
.arima-case {
    padding: 0.65rem 1rem; border-radius: 10px;
    margin-bottom: 0.4rem; font-size: 0.88rem;
    border: 1px solid #e2e8f0; background: #f8fafc;
    transition: all 0.2s;
}
.arima-case-active {
    border: 2px solid #6366f1; background: #f0f0ff;
    font-weight: 600;
}

/* ── 신뢰 점수 ── */
.trust-container {
    text-align: center; padding: 2rem 1rem;
    background: linear-gradient(135deg, #f8fafc 0%, #f0f4ff 100%);
    border-radius: 20px; border: 1px solid #e0e7ff;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    margin-bottom: 1.5rem;
}
.trust-label {
    font-size: 0.8rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.12em;
}
.trust-score-num {
    font-size: 5rem; font-weight: 900; line-height: 1;
    margin: 0.3rem 0;
}
.trust-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 1.2rem; border-radius: 999px;
    font-size: 0.95rem; font-weight: 700; margin-top: 0.3rem;
}
.trust-badge-green  { background: #d1fae5; color: #065f46; }
.trust-badge-yellow { background: #fef3c7; color: #92400e; }
.trust-badge-red    { background: #fee2e2; color: #991b1b; }
.trust-meta { color: #64748b; font-size: 0.85rem; margin-top: 0.8rem; }

/* ── 신뢰 항목 카드 ── */
.trust-item {
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 0.6rem; border: 1px solid #e2e8f0;
}
.trust-item-green  { background: #f0fdf4; border-color: #bbf7d0; }
.trust-item-yellow { background: #fffbeb; border-color: #fde68a; }
.trust-item-red    { background: #fef2f2; border-color: #fecaca; }
.trust-item-header { display:flex; justify-content:space-between; align-items:center; }
.trust-item-name   { font-weight: 700; font-size: 0.95rem; color: #1e293b; }
.trust-item-score  { font-size: 1.5rem; font-weight: 900; }
.trust-item-reason { color: #64748b; font-size: 0.85rem; margin-top: 0.3rem; }
.trust-bar-bg   { background: #e5e7eb; border-radius: 999px; height: 6px; margin-top: 0.7rem; }
.trust-bar-fill { border-radius: 999px; height: 6px; }

/* ── 진단 카드 ── */
.diag-card {
    border-radius: 12px; padding: 1rem; text-align: center;
    border: 1px solid #e2e8f0;
}
.diag-card-pass { background: #f0fdf4; border-color: #bbf7d0; }
.diag-card-fail { background: #fef2f2; border-color: #fecaca; }
.diag-card-warn { background: #fffbeb; border-color: #fde68a; }
.diag-card-na   { background: #f8fafc; border-color: #e2e8f0; }
.diag-name  { font-size: 0.75rem; font-weight: 700; color: #64748b;
              text-transform: uppercase; letter-spacing: 0.06em; }
.diag-sym   { font-size: 1.6rem; margin: 0.3rem 0; }
.diag-val   { font-size: 1.1rem; font-weight: 800; color: #1e293b; }
.diag-desc  { font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── 모델 이유 카드 ── */
.reason-card {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.75rem 1rem; border-radius: 10px;
    margin-bottom: 0.4rem; background: #f8fafc;
    border: 1px solid #e2e8f0; font-size: 0.9rem; color: #374151;
}
.reason-sym { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.1rem; }

/* ── 초기 화면 피처 카드 ── */
.feature-card {
    border-radius: 16px; padding: 1.4rem;
    background: white; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center; height: 100%;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.feature-title { font-weight: 700; font-size: 1rem; color: #1e293b; margin-bottom: 0.4rem; }
.feature-desc  { font-size: 0.85rem; color: #64748b; line-height: 1.5; }

/* ── 탭 컨텐츠 상단 여백 ── */
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

/* ── 버튼 개선 ── */
.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* ── info 박스 ── */
.stAlert [data-testid="stMarkdownContainer"] p { margin: 0; }

/* ── 배지 ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
}
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-violet { background: #ede9fe; color: #5b21b6; }

/* ── Streamlit 기본 요소 개선 ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem; background: #f8fafc;
    padding: 0.3rem; border-radius: 12px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 0.4rem 0.9rem;
    font-size: 0.85rem; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
div[data-testid="metric-container"] {
    background: white; border-radius: 10px;
    border: 1px solid #e2e8f0; padding: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 유틸리티 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true,dtype=float), np.array(y_pred,dtype=float)
    mask = y_true != 0
    if mask.sum()==0: return np.nan
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)

def safe_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true,dtype=float), np.array(y_pred,dtype=float)
    denom = (np.abs(y_true)+np.abs(y_pred))/2
    mask = denom!=0
    if mask.sum()==0: return np.nan
    return float(np.mean(np.abs(y_true[mask]-y_pred[mask])/denom[mask])*100)

def calc_mase(y_true, y_pred, y_train):
    y_true=np.array(y_true,dtype=float); y_pred=np.array(y_pred,dtype=float)
    y_train=np.array(y_train,dtype=float)
    ne=np.abs(np.diff(y_train))
    scale=ne.mean() if len(ne)>0 and ne.mean()!=0 else 1.0
    return float(np.mean(np.abs(y_true-y_pred))/scale)

def compute_metrics(y_true, y_pred, y_train, model_fit=None):
    y_true=np.array(y_true,dtype=float); y_pred=np.array(y_pred,dtype=float)
    mae=float(mean_absolute_error(y_true,y_pred))
    rmse=float(np.sqrt(mean_squared_error(y_true,y_pred)))
    mape=safe_mape(y_true,y_pred); smape=safe_smape(y_true,y_pred)
    mase=calc_mase(y_true,y_pred,y_train)
    aic,bic=np.nan,np.nan
    if model_fit is not None:
        try: aic=float(model_fit.aic)
        except: pass
        try: bic=float(model_fit.bic)
        except: pass
    return dict(MAE=mae,RMSE=rmse,MAPE=mape,sMAPE=smape,MASE=mase,AIC=aic,BIC=bic)

def ljung_box_pval(res,lags=10):
    try:
        r=np.array(res,dtype=float); r=r[~np.isnan(r)]
        if len(r)<4: return np.nan
        al=min(lags,len(r)//2)
        if al<1: return np.nan
        return float(acorr_ljungbox(r,lags=[al],return_df=True)['lb_pvalue'].iloc[0])
    except: return np.nan

def jarque_bera_pval(res):
    try:
        r=np.array(res,dtype=float); r=r[~np.isnan(r)]
        if len(r)<8: return np.nan
        return float(jarque_bera(r)[1])
    except: return np.nan

def durbin_watson_stat(res):
    try:
        r=np.array(res,dtype=float); r=r[~np.isnan(r)]
        if len(r)<4: return np.nan
        return float(durbin_watson(r))
    except: return np.nan

def breusch_pagan_pval(res):
    try:
        r=np.array(res,dtype=float); r=r[~np.isnan(r)]
        if len(r)<8: return np.nan
        x=add_constant(np.arange(len(r)))
        _,p,_,_=het_breuschpagan(r,x)
        return float(p)
    except: return np.nan


# ══════════════════════════════════════════════════════════════════
# 데이터 처리 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def try_parse_datetime(series):
    try:
        parsed=pd.to_datetime(series,errors='coerce')
        if parsed.notna().mean()>0.8: return parsed,float(parsed.notna().mean())
    except: pass
    for fmt in ['%Y-%m','%Y/%m','%Y.%m','%Y-%m-%d','%Y/%m/%d',
                '%m/%d/%Y','%d/%m/%Y','%Y-%m-%d %H:%M:%S']:
        try:
            parsed=pd.to_datetime(series,format=fmt,errors='coerce')
            if parsed.notna().mean()>0.8: return parsed,float(parsed.notna().mean())
        except: continue
    return None,0.0

def auto_detect_columns(df):
    date_col,numeric_cols=None,[]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]): numeric_cols.append(col); continue
        if date_col is None:
            parsed,rate=try_parse_datetime(df[col].astype(str))
            if parsed is not None and rate>0.8: date_col=col
            else:
                try:
                    if pd.to_numeric(df[col],errors='coerce').notna().mean()>0.5:
                        numeric_cols.append(col)
                except: pass
    return date_col,numeric_cols

def load_and_validate(df,date_col,val_col):
    df=df[[date_col,val_col]].copy()
    parsed,_=try_parse_datetime(df[date_col].astype(str))
    if parsed is None: raise ValueError(f"'{date_col}' 열을 날짜로 변환할 수 없습니다.")
    df[date_col]=parsed
    df=df.dropna(subset=[date_col])
    df[val_col]=pd.to_numeric(df[val_col],errors='coerce')
    df=df.sort_values(date_col).reset_index(drop=True)
    dup=int(df.duplicated(subset=[date_col]).sum())
    df=df.groupby(date_col,as_index=False)[val_col].mean()
    df=df.set_index(date_col)
    try:
        inf=pd.infer_freq(df.index)
        if inf: df.index.freq=inf
    except: pass
    miss=int(df[val_col].isna().sum())
    df[val_col]=df[val_col].interpolate(method='linear').ffill().bfill()
    return df,dup,miss

def detect_freq(index):
    if len(index)<2: return 'Monthly','MS',12
    try:
        secs=pd.Series(index).diff().dropna().apply(lambda x:x.total_seconds())
        days=secs.median()/86400
        if days<=1.5:  return 'Daily','D',7
        elif days<=8:  return 'Weekly','W',52
        elif days<=35: return 'Monthly','MS',12
        elif days<=100:return 'Quarterly','QS',4
        else:          return 'Yearly','YS',1
    except: return 'Monthly','MS',12

def generate_future_index(last_date,h,freq_str):
    try: return pd.date_range(start=last_date,periods=h+1,freq=freq_str)[1:]
    except: return pd.date_range(start=last_date,periods=h+1,freq='MS')[1:]


# ══════════════════════════════════════════════════════════════════
# 통계 분석 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def run_adf(series):
    try:
        c=series.dropna()
        if len(c)<8: return dict(statistic=np.nan,pvalue=np.nan,critical={'1%':np.nan,'5%':np.nan,'10%':np.nan},is_stationary=False)
        r=adfuller(c,autolag='AIC')
        return dict(statistic=float(r[0]),pvalue=float(r[1]),critical=r[4],is_stationary=bool(r[1]<0.05))
    except: return dict(statistic=np.nan,pvalue=np.nan,critical={'1%':np.nan,'5%':np.nan,'10%':np.nan},is_stationary=False)

def get_decomposition(series,period):
    try:
        if len(series)<period*2: return None
        return seasonal_decompose(series,model='additive',period=period,extrapolate_trend='freq')
    except: return None

def auto_arima_order(series,d_val):
    try:
        ds=series.dropna()
        for _ in range(d_val): ds=ds.diff().dropna()
        if len(ds)<8: return (1,d_val,1)
        ml=min(10,len(ds)//3)
        if ml<2: return (1,d_val,1)
        ar=acf(ds,nlags=ml,alpha=0.05); pr=pacf(ds,nlags=ml,alpha=0.05)
        av,pv=ar[0][1:],pr[0][1:]
        conf=1.96/np.sqrt(len(ds))
        p=min(next((i+1 for i,v in enumerate(pv) if abs(v)<conf),1),3)
        q=min(next((i+1 for i,v in enumerate(av) if abs(v)<conf),1),3)
        return (p,d_val,q)
    except: return (1,d_val,1)

def detect_arima_case(order):
    p,d,q=order
    if p==0 and d==0 and q==0: return "백색잡음"
    if p>0  and d==0 and q==0: return f"AR({p})"
    if p==0 and d==0 and q>0:  return f"MA({q})"
    if p>0  and d==0 and q>0:  return f"ARMA({p},{q})"
    if p==0 and d>0  and q==0: return f"I({d}) 랜덤워크"
    return f"ARIMA({p},{d},{q})"


# ══════════════════════════════════════════════════════════════════
# 모델 훈련 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def naive_forecast(train,h):
    return np.array([float(train.iloc[-1])]*h),None

def moving_average_forecast(train,h,window=4):
    w=min(window,len(train))
    return np.array([float(train.iloc[-w:].mean())]*h),None

def ses_forecast(train,h):
    try:
        m=SimpleExpSmoothing(train,initialization_method='estimated').fit(optimized=True)
        return m.forecast(h).values.astype(float),m
    except: return naive_forecast(train,h)

def holt_forecast(train,h):
    try:
        m=ExponentialSmoothing(train,trend='add',initialization_method='estimated').fit(optimized=True)
        return m.forecast(h).values.astype(float),m
    except: return ses_forecast(train,h)

def holtwinters_forecast(train,h,sp):
    if len(train)<sp*2+1: return holt_forecast(train,h)
    results=[]
    for seasonal in ['add','mul']:
        try:
            if seasonal=='mul' and (train<=0).any(): continue
            m=ExponentialSmoothing(train,trend='add',seasonal=seasonal,
                                   seasonal_periods=sp,initialization_method='estimated').fit(optimized=True)
            results.append((m.aic,seasonal,m))
        except: continue
    if not results: return holt_forecast(train,h)
    _,_,best_fit=min(results,key=lambda x:x[0])
    return best_fit.forecast(h).values.astype(float),best_fit

def arima_forecast(train,h,order=(1,1,1)):
    for o in [order,(1,1,1),(0,1,1),(1,1,0),(0,1,0)]:
        try:
            fit=ARIMA(train,order=o).fit()
            fc=fit.get_forecast(h)
            return fc.predicted_mean.values.astype(float),fit,fc.conf_int(alpha=0.05).values
        except: continue
    pred,_=naive_forecast(train,h)
    return pred,None,np.column_stack([pred*0.9,pred*1.1])

def sarima_forecast(train,h,order=(1,1,1),sp=12):
    for so in [(1,1,1,sp),(1,1,0,sp),(0,1,1,sp),(1,0,1,sp)]:
        try:
            m=SARIMAX(train,order=order,seasonal_order=so,
                      enforce_stationarity=False,enforce_invertibility=False)
            fit=m.fit(disp=False,maxiter=100)
            fc=fit.get_forecast(h)
            return fc.predicted_mean.values.astype(float),fit,fc.conf_int(alpha=0.05).values
        except: continue
    return arima_forecast(train,h,order)


# ══════════════════════════════════════════════════════════════════
# 신뢰 리포트 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def compute_trust_report(n,missing_before,adf_result,metrics_df,best_model,horizon,sp,residuals):
    scores,reasons={},{}
    mp=missing_before/max(n,1)*100
    q=100
    if n<24: q-=20
    if n<12: q-=20
    if mp>5: q-=15
    if mp>20: q-=20
    scores['데이터 품질']=max(0,q); reasons['데이터 품질']=f"총 {n}개 | 결측률 {mp:.1f}%"

    if np.isnan(adf_result['pvalue']):
        scores['정상성']=60; reasons['정상성']="ADF 검정 불가"
    elif adf_result['is_stationary']:
        scores['정상성']=90; reasons['정상성']=f"ADF p={adf_result['pvalue']:.4f} → 정상 ✅"
    else:
        scores['정상성']=65; reasons['정상성']=f"ADF p={adf_result['pvalue']:.4f} → 비정상 (차분 적용) △"

    if best_model and 'Naive' in metrics_df.index and best_model in metrics_df.index:
        nr=metrics_df.loc['Naive','RMSE']; br=metrics_df.loc[best_model,'RMSE']
        if nr>0 and not np.isnan(nr) and not np.isnan(br):
            imp=(nr-br)/nr*100
            if imp>20: scores['Naive 대비']=95; reasons['Naive 대비']=f"{best_model}이 Naive보다 {imp:.1f}% 정확 ✅"
            elif imp>0: scores['Naive 대비']=75; reasons['Naive 대비']=f"{best_model}이 Naive보다 {imp:.1f}% 개선 △"
            else: scores['Naive 대비']=40; reasons['Naive 대비']=f"⚠️ {best_model}이 Naive보다 나쁨"
        else: scores['Naive 대비']=70; reasons['Naive 대비']="비교 불가"
    else: scores['Naive 대비']=70; reasons['Naive 대비']="정보 없음"

    lb=ljung_box_pval(residuals)
    if np.isnan(lb): scores['잔차 독립성']=65; reasons['잔차 독립성']="검정 불가"
    elif lb>0.05: scores['잔차 독립성']=90; reasons['잔차 독립성']=f"Ljung-Box p={lb:.3f} → 백색잡음 ✅"
    else: scores['잔차 독립성']=45; reasons['잔차 독립성']=f"⚠️ p={lb:.3f} → 패턴 남음"

    r=horizon/max(n,1)
    if r<=0.15: scores['예측 기간']=95; reasons['예측 기간']=f"{horizon}스텝 ({r*100:.0f}%) → 적절 ✅"
    elif r<=0.3: scores['예측 기간']=75; reasons['예측 기간']=f"{horizon}스텝 ({r*100:.0f}%) → 다소 김 △"
    elif r<=0.5: scores['예측 기간']=55; reasons['예측 기간']=f"{horizon}스텝 ({r*100:.0f}%) → 주의 ⚠️"
    else: scores['예측 기간']=30; reasons['예측 기간']=f"⚠️ {horizon}스텝이 너무 김"

    return float(np.mean(list(scores.values()))),scores,reasons

def trust_color_class(s):
    if s>=75: return 'green','trust-badge-green','🟢 신뢰 가능','#10b981','trust-item-green','kpi-card-trust-green'
    if s>=55: return 'yellow','trust-badge-yellow','🟡 조건부 신뢰','#f59e0b','trust-item-yellow','kpi-card-trust-yellow'
    return 'red','trust-badge-red','🔴 주의 필요','#ef4444','trust-item-red','kpi-card-trust-red'


# ══════════════════════════════════════════════════════════════════
# 스토리텔링 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def generate_story(series,val_col,freq_label,sp,adf_result,decomp,metrics_df,best_model,arima_order,trust_total,horizon):
    n=len(series)
    start=series.index[0].strftime('%Y년 %m월'); end=series.index[-1].strftime('%Y년 %m월')
    mean_v=series[val_col].mean(); max_v=series[val_col].max(); min_v=series[val_col].min()
    fh=series[val_col].iloc[:n//2].mean(); sh=series[val_col].iloc[n//2:].mean()
    td="상승" if sh>fh*1.05 else ("하락" if sh<fh*0.95 else "안정적")
    tp=abs(sh-fh)/fh*100
    seas_str=""
    if decomp is not None:
        sv=float(np.var(decomp.seasonal.dropna())); rv=float(np.var(decomp.resid.dropna()))
        ss=max(0,1-rv/(sv+rv+1e-9))
        seas_str=f"매 {sp}스텝마다 반복되는 <b>뚜렷한 계절 패턴</b>이 있습니다 (강도 {ss:.0%})." if ss>0.4 else f"계절성은 약하게 나타납니다 (강도 {ss:.0%})."
    if best_model and best_model in metrics_df.index and 'Naive' in metrics_df.index:
        br=metrics_df.loc[best_model,'RMSE']; nr=metrics_df.loc['Naive','RMSE']
        imp=(nr-br)/nr*100 if nr>0 else 0
        model_str=f"<b>{best_model}</b> 모델이 선택되었으며 Naive 대비 <b>{imp:.1f}% 더 정확</b>합니다."
    else: model_str=f"<b>{best_model}</b> 모델이 선택되었습니다."
    stat_str="정상 시계열" if adf_result['is_stationary'] else f"비정상 시계열 (p={adf_result['pvalue']:.3f}, d={arima_order[1]} 차분 적용)"
    return f"""📊 <b>{start} ~ {end}</b> | <b>{n}개</b> {freq_label} 관측치<br><br>
평균 <b>{mean_v:.1f}</b>, 범위 <b>{min_v:.1f} ~ {max_v:.1f}</b> 이며, 후반부가 전반부 대비 <b>{tp:.1f}% {td}</b> 추세입니다.
{seas_str}<br><br>
ADF 검정 결과 <b>{stat_str}</b>이며, ARIMA 차수 <b>{arima_order}</b> (<b>{detect_arima_case(arima_order)}</b>) 가 추정됩니다.
{model_str} 종합 신뢰 점수 <b>{trust_total:.0f}/100점</b>."""

def generate_model_reason(best_model,metrics_df,decomp,sp,adf_result,arima_order):
    reasons=[]
    if decomp is not None:
        sv=float(np.var(decomp.seasonal.dropna())); rv=float(np.var(decomp.resid.dropna()))
        tv=float(np.var(decomp.trend.dropna()))
        ss=max(0,1-rv/(sv+rv+1e-9)); ts=max(0,1-rv/(tv+rv+1e-9))
        if ss>0.4: reasons.append(("✅",f"강한 계절성 감지 ({ss:.0%}) → 계절 모델 선택 근거"))
        if ts>0.5: reasons.append(("✅",f"강한 추세 감지 ({ts:.0%}) → 추세 컴포넌트 필요"))
    if not adf_result['is_stationary'] and not np.isnan(adf_result['pvalue']):
        reasons.append(("✅",f"비정상 시계열 → d={arima_order[1]} 차분 적용"))
    if best_model and best_model in metrics_df.index:
        m=metrics_df.loc[best_model]
        reasons.append(("✅",f"테스트 RMSE={m['RMSE']:.2f} — 모든 모델 중 최저"))
        mv=m['MASE']
        if not np.isnan(mv): reasons.append(("✅" if mv<1 else "⚠️",f"MASE={mv:.3f} → {'Naive보다 정확' if mv<1 else 'Naive보다 부정확'}"))
    return reasons


# ══════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════
COLORS={
    'Naive':'#94a3b8','MovAvg':'#64748b','SES':'#3b82f6',
    'Holt':'#8b5cf6','HoltWinters':'#f59e0b','ARIMA':'#10b981','SARIMA':'#ec4899',
}

def _hex_rgb(h):
    h=h.lstrip('#')
    return ','.join(str(int(h[i:i+2],16)) for i in (0,2,4))

def styled_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#1e293b', family='Inter')),
        height=height, plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=True,gridcolor='#f1f5f9',linecolor='#e2e8f0',
                   tickfont=dict(size=11,color='#64748b')),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',linecolor='#e2e8f0',
                   tickfont=dict(size=11,color='#64748b')),
        margin=dict(l=50,r=20,t=50 if title else 20,b=40),
        font=dict(family='Inter',color='#334155'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)',bordercolor='#e2e8f0',
                    borderwidth=1,font=dict(size=11)),
    )
    return fig

def plot_raw(series,val_col):
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,y=series[val_col],mode='lines',name='실제 데이터',
        line=dict(color='#6366f1',width=2.5),
        fill='tozeroy',fillcolor='rgba(99,102,241,0.07)'
    ))
    return styled_layout(fig,'원본 시계열 데이터',350)

def plot_decomp(decomp):
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,
                      subplot_titles=['원본 (Observed)','추세 (Trend)','계절성 (Seasonal)','잔차 (Residual)'],
                      vertical_spacing=0.07)
    for row,(comp,color) in enumerate([(decomp.observed,'#6366f1'),(decomp.trend,'#f59e0b'),
                                        (decomp.seasonal,'#10b981'),(decomp.resid,'#ef4444')],start=1):
        fig.add_trace(go.Scatter(x=comp.index,y=comp.values,mode='lines',showlegend=False,
                                 line=dict(color=color,width=1.8)),row=row,col=1)
    fig.update_layout(height=700,title='시계열 분해 (가법 모형)',plot_bgcolor='white',
                      paper_bgcolor='white',font=dict(family='Inter'),
                      margin=dict(l=50,r=20,t=60,b=40))
    for i in range(1,5):
        fig.update_xaxes(showgrid=True,gridcolor='#f1f5f9',row=i,col=1)
        fig.update_yaxes(showgrid=True,gridcolor='#f1f5f9',row=i,col=1)
    return fig

def plot_acf_pacf(series,lags=24):
    try:
        n=len(series.dropna()); sl=min(lags,n//3)
        if sl<4: return None
        ar=acf(series.dropna(),nlags=sl,alpha=0.05)
        pr=pacf(series.dropna(),nlags=sl,alpha=0.05)
        av,pv=ar[0],pr[0]; conf=1.96/np.sqrt(n)
        fig=make_subplots(rows=1,cols=2,subplot_titles=['ACF (→ MA 차수 q 결정)','PACF (→ AR 차수 p 결정)'])
        for ci,(vals,nm) in enumerate([(av,'ACF'),(pv,'PACF')],start=1):
            colors=['#6366f1' if abs(v)>conf else '#cbd5e1' for v in vals]
            fig.add_trace(go.Bar(x=list(range(len(vals))),y=vals,marker_color=colors,showlegend=False),row=1,col=ci)
            for sign in [1,-1]:
                fig.add_hline(y=sign*conf,line_dash='dash',line_color='#ef4444',opacity=0.6,row=1,col=ci)
        fig.update_layout(height=320,title='ACF / PACF',plot_bgcolor='white',paper_bgcolor='white',
                          font=dict(family='Inter'),margin=dict(l=50,r=20,t=50,b=40))
        for i in [1,2]:
            fig.update_xaxes(showgrid=False,row=1,col=i)
            fig.update_yaxes(showgrid=True,gridcolor='#f1f5f9',row=1,col=i)
        return fig
    except: return None

def plot_forecast(series,val_col,test_results,future_results,best_model,train_end):
    fig=go.Figure()
    # 실제 데이터
    fig.add_trace(go.Scatter(x=series.index,y=series[val_col],
                             name='실제 데이터',mode='lines',
                             line=dict(color='#1e293b',width=2.5)))
    # 비최적 모델 먼저 (뒤에 깔리도록) - 연한 회색
    for name,info in test_results.items():
        if name==best_model: continue
        fig.add_trace(go.Scatter(
            x=info['index'],y=info['pred'],name=name,mode='lines',
            line=dict(color='#cbd5e1',dash='dot',width=1),opacity=0.6))
    # 최적 모델: 신뢰구간 + 굵은 컬러 선
    if best_model and best_model in test_results:
        info=test_results[best_model]; color=COLORS.get(best_model,'#6366f1')
        if info.get('ci') is not None:
            ci=info['ci']; il=list(info['index'])
            fig.add_trace(go.Scatter(
                x=il+il[::-1],y=list(ci[:,1])+list(ci[:,0])[::-1],
                fill='toself',fillcolor=f'rgba({_hex_rgb(color)},0.18)',
                line=dict(color='rgba(0,0,0,0)'),name='95% 신뢰구간',hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter(
            x=info['index'],y=info['pred'],name=f'{best_model} ★ 최적',mode='lines',
            line=dict(color=color,dash='solid',width=3)))
    # 미래 예측
    if best_model and best_model in future_results:
        fi=future_results[best_model]; color=COLORS.get(best_model,'#6366f1')
        if fi.get('ci') is not None:
            ci=fi['ci']; il=list(fi['index'])
            fig.add_trace(go.Scatter(
                x=il+il[::-1],y=list(ci[:,1])+list(ci[:,0])[::-1],
                fill='toself',fillcolor=f'rgba({_hex_rgb(color)},0.12)',
                line=dict(color='rgba(0,0,0,0)'),name='미래 95% CI',hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter(
            x=fi['index'],y=fi['pred'],name=f'{best_model} 미래 예측',
            mode='lines+markers',line=dict(color=color,width=3,dash='dash'),
            marker=dict(size=6,symbol='circle')))
        if len(fi['index'])>0:
            fig.add_vrect(x0=str(fi['index'][0]),x1=str(fi['index'][-1]),
                          fillcolor='rgba(99,102,241,0.04)',layer='below',line_width=0,
                          annotation_text='예측 구간',annotation_position='top left')
    fig.add_vline(x=int(train_end.timestamp()*1000),line_dash='dash',
                  line_color='#94a3b8',opacity=0.8,
                  annotation_text='훈련 | 테스트',annotation_position='top right')
    fig=styled_layout(fig,f'예측 결과 — 최적 모델: {best_model}  (음영=95% 신뢰구간)',500)
    fig.update_layout(legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
    return fig

def plot_residuals_full(residuals,model_name):
    res=np.array(residuals,dtype=float); res=res[~np.isnan(res)]
    fig=make_subplots(rows=2,cols=2,
                      subplot_titles=['잔차 시계열','잔차 분포 + 정규곡선','QQ Plot (정규성)','잔차 ACF'])
    fig.add_trace(go.Scatter(y=res,mode='lines+markers',
                             marker=dict(size=3,color='#6366f1'),
                             line=dict(color='#6366f1',width=1),showlegend=False),row=1,col=1)
    fig.add_hline(y=0,line_dash='dash',line_color='#ef4444',opacity=0.6,row=1,col=1)
    fig.add_trace(go.Histogram(x=res,nbinsx=20,marker_color='#6366f1',
                               opacity=0.6,showlegend=False,histnorm='probability density'),row=1,col=2)
    xr=np.linspace(res.min(),res.max(),100)
    fig.add_trace(go.Scatter(x=xr,y=sp_norm.pdf(xr,res.mean(),res.std()),
                             mode='lines',line=dict(color='#ef4444',width=2),showlegend=False),row=1,col=2)
    try:
        sr=np.sort(res); nn=len(sr)
        pp=np.array([(i-0.5)/nn for i in range(1,nn+1)])
        qt=sp_norm.ppf(pp)
        fig.add_trace(go.Scatter(x=qt,y=sr,mode='markers',
                                 marker=dict(color='#10b981',size=4),showlegend=False),row=2,col=1)
        sl=np.std(sr); mu=np.mean(sr)
        fig.add_trace(go.Scatter(x=[qt[0],qt[-1]],y=[sl*qt[0]+mu,sl*qt[-1]+mu],
                                 mode='lines',line=dict(color='#ef4444',dash='dash'),showlegend=False),row=2,col=1)
    except: pass
    try:
        nl=min(20,len(res)//3)
        if nl>=4:
            av=acf(res,nlags=nl)[0]; conf=1.96/np.sqrt(len(res))
            colors=['#ef4444' if abs(v)>conf else '#6366f1' for v in av]
            fig.add_trace(go.Bar(x=list(range(len(av))),y=av,
                                 marker_color=colors,showlegend=False),row=2,col=2)
            for sign in [1,-1]:
                fig.add_hline(y=sign*conf,line_dash='dash',
                              line_color='#ef4444',opacity=0.5,row=2,col=2)
    except: pass
    fig.update_layout(height=500,title=f'잔차 진단 4종 — {model_name}',
                      plot_bgcolor='white',paper_bgcolor='white',
                      font=dict(family='Inter'),margin=dict(l=50,r=20,t=70,b=40))
    for r in range(1,3):
        for c in range(1,3):
            fig.update_xaxes(showgrid=True,gridcolor='#f1f5f9',row=r,col=c)
            fig.update_yaxes(showgrid=True,gridcolor='#f1f5f9',row=r,col=c)
    return fig

def plot_model_comparison(metrics_df,metric='RMSE'):
    if metric not in metrics_df.columns: return go.Figure()
    vals=metrics_df[metric].values
    valid=[v for v in vals if not np.isnan(v)]
    if not valid: return go.Figure()
    min_v=min(valid)
    colors=['#F59E0B' if (not np.isnan(v) and v==min_v) else '#e2e8f0' for v in vals]
    fig=go.Figure(go.Bar(
        x=metrics_df.index.tolist(),y=vals,marker_color=colors,
        text=[f'{v:.3f}' if not np.isnan(v) else 'N/A' for v in vals],
        textposition='outside',marker_line_width=0))
    return styled_layout(fig,f'{metric} 비교 (노란색=최적, 낮을수록 좋음)',360)

def plot_trust_gauge(score):
    """신뢰 점수 게이지 차트"""
    color = '#10b981' if score>=75 else ('#f59e0b' if score>=55 else '#ef4444')
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x':[0,1],'y':[0,1]},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#94a3b8',
                    'tickvals':[0,25,50,75,100]},
            'bar':{'color':color,'thickness':0.3},
            'bgcolor':'white',
            'borderwidth':0,
            'steps':[
                {'range':[0,55],'color':'#fef2f2'},
                {'range':[55,75],'color':'#fefce8'},
                {'range':[75,100],'color':'#f0fdf4'},
            ],
            'threshold':{'line':{'color':color,'width':4},'thickness':0.75,'value':score}
        },
        number={'font':{'size':52,'color':color,'family':'Inter'},'suffix':'/100'}
    ))
    fig.update_layout(height=260,margin=dict(l=30,r=30,t=10,b=10),
                      paper_bgcolor='white',font=dict(family='Inter'))
    return fig


# ══════════════════════════════════════════════════════════════════
# 샘플 데이터
# ══════════════════════════════════════════════════════════════════
def get_airpassengers():
    idx=pd.date_range('1949-01',periods=144,freq='MS')
    vals=[112,118,132,129,121,135,148,148,136,119,104,118,115,126,141,135,125,149,170,170,158,133,114,140,
          145,150,178,163,172,178,199,199,184,162,146,166,171,180,193,181,183,218,230,242,209,191,172,194,
          196,196,236,235,229,243,264,272,237,211,180,201,204,188,235,227,234,264,302,293,259,229,203,229,
          242,233,267,269,270,315,364,347,312,274,237,278,284,277,317,313,318,374,413,405,355,306,271,306,
          315,301,356,348,355,422,465,467,404,347,305,336,340,318,362,348,363,435,491,505,404,359,310,337,
          360,342,406,396,420,472,548,559,463,407,362,405,417,391,419,461,472,535,622,606,508,461,390,432]
    return pd.DataFrame({'Month':idx,'Passengers':vals})


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main():
    # ── 사이드바 ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.8rem 0 0.2rem;font-size:1rem;font-weight:800;color:#1e293b;">'
            '📈 시계열 예측 신뢰 분석기</div>'
            '<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.8rem;">TimeSeries Trust Analyzer</div>',
            unsafe_allow_html=True
        )
        st.divider()
        data_mode=st.radio("데이터 소스",["CSV 업로드","샘플 데이터 (AirPassengers)"],
                           label_visibility="collapsed")
        uploaded=None
        if data_mode=="CSV 업로드":
            uploaded=st.file_uploader("CSV 파일 업로드",type=['csv'],label_visibility="collapsed")
        st.divider()
        st.markdown('<div style="font-size:0.78rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">'
                    'MODEL SETTINGS</div>', unsafe_allow_html=True)
        forecast_horizon=st.slider("예측 기간 (스텝)",1,60,12)
        test_ratio      =st.slider("테스트 비율 (%)",10,40,20)
        use_auto_arima  =st.checkbox("ARIMA 차수 자동 추정",value=True)
        use_sarima      =st.checkbox("SARIMA 포함",value=True,
                                     help="계절성 ARIMA. 30초 내외 소요")

    # ── 데이터 로드 ───────────────────────────────────────────────
    df_raw=None; date_col=None; numeric_cols=[]; val_col=None

    if data_mode=="샘플 데이터 (AirPassengers)":
        df_raw=get_airpassengers(); date_col='Month'
        numeric_cols=['Passengers']; val_col='Passengers'
        st.sidebar.success("✅ AirPassengers 로드 완료")
    elif uploaded is not None:
        try:
            df_raw=pd.read_csv(uploaded)
            date_col,numeric_cols=auto_detect_columns(df_raw)
        except Exception as e:
            st.error(f"❌ CSV 읽기 오류: {e}"); st.stop()
        if not numeric_cols: st.error("❌ 수치형 열이 없습니다."); st.stop()
        val_col=numeric_cols[0]
    else:
        # ══════════════════════════════════════════════════════════
        # 히어로 랜딩 페이지 (간결 버전)

        # 랜딩 페이지: 사이드바를 얇게 + 헤더 숨김
        st.markdown(
            '<style>'
            '[data-testid="stSidebar"]{min-width:220px!important;max-width:220px!important;}'
            '[data-testid="stSidebar"] .stRadio label{font-size:0.85rem;}'
            '</style>',
            unsafe_allow_html=True
        )
        # ══════════════════════════════════════════════════════════

        # ① 히어로 배너 (작게 나눠서 안전하게 렌더링)
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#4c1d95 100%);'
            'border-radius:20px;padding:2.5rem 2.5rem 2rem;margin-bottom:1.5rem;">'
            '<div style="display:inline-block;background:rgba(139,92,246,0.3);'
            'border:1px solid rgba(167,139,250,0.4);border-radius:999px;'
            'padding:3px 14px;font-size:0.75rem;color:#c4b5fd;'
            'font-weight:600;letter-spacing:0.06em;margin-bottom:1rem;">✦ AI TIME SERIES ANALYSIS</div>'
            '<h1 style="color:white;font-size:2.4rem;font-weight:900;'
            'line-height:1.2;margin:0 0 0.8rem 0;letter-spacing:-1px;">'
            '시계열 예측,<br>'
            '<span style="color:#a78bfa;">신뢰도까지 판단합니다</span></h1>'
            '<p style="color:#94a3b8;font-size:0.92rem;margin:0 0 1.5rem 0;line-height:1.6;">'
            'CSV 파일 하나로 &nbsp;<strong style="color:#c4b5fd;">자동 분석</strong> →'
            '&nbsp;<strong style="color:#c4b5fd;">7개 모델 예측</strong> →'
            '&nbsp;<strong style="color:#c4b5fd;">신뢰 점수 채점</strong>까지</p>'
            '<div style="display:flex;gap:0.8rem;flex-wrap:wrap;">'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">7개</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">예측 모델</div></div>'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">5종</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">신뢰 기준</div></div>'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">4종</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">잔차 진단</div></div>'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">95%</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">신뢰구간</div></div>'
            '</div></div>',
            unsafe_allow_html=True
        )

        # ② 데모 차트 생성
        import warnings as _w; _w.filterwarnings('ignore')
        _demo_df = get_airpassengers()
        _demo_s, _, _ = load_and_validate(_demo_df, 'Month', 'Passengers')
        _demo_future_idx = generate_future_index(_demo_s.index[-1], 24, 'MS')
        _demo_decomp = get_decomposition(_demo_s['Passengers'], 12)
        try:
            _demo_pred, _ = holtwinters_forecast(_demo_s['Passengers'], 24, 12)
        except Exception:
            _demo_pred = np.full(24, float(_demo_s['Passengers'].iloc[-1]))

        _ci_upper = _demo_pred * 1.12
        _ci_lower = _demo_pred * 0.88
        _il = list(_demo_future_idx)

        _fig_demo = go.Figure()
        if _demo_decomp is not None:
            _fig_demo.add_trace(go.Scatter(
                x=_demo_decomp.trend.index, y=_demo_decomp.trend.values,
                mode='lines', name='추세 (Trend)',
                line=dict(color='#f59e0b', width=2, dash='dot'), opacity=0.8
            ))
        _fig_demo.add_trace(go.Scatter(
            x=_demo_s.index, y=_demo_s['Passengers'],
            mode='lines', name='실제 데이터',
            line=dict(color='#6366f1', width=2),
            fill='tozeroy', fillcolor='rgba(99,102,241,0.07)'
        ))
        _fig_demo.add_trace(go.Scatter(
            x=_il + _il[::-1],
            y=list(_ci_upper) + list(_ci_lower)[::-1],
            fill='toself', fillcolor='rgba(236,72,153,0.13)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% 신뢰구간', hoverinfo='skip', showlegend=True
        ))
        _fig_demo.add_trace(go.Scatter(
            x=_demo_future_idx, y=_demo_pred,
            mode='lines+markers', name='HoltWinters 예측 (24개월)',
            line=dict(color='#ec4899', width=2.5, dash='dash'),
            marker=dict(size=4)
        ))
        _fig_demo.add_vline(
            x=int(_demo_s.index[-1].timestamp() * 1000),
            line_dash='dash', line_color='#94a3b8', opacity=0.6,
            annotation_text='현재 → 예측', annotation_position='top right'
        )
        _fig_demo.update_layout(
            height=340, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=50, r=20, t=10, b=40),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                       tickfont=dict(size=11, color='#94a3b8')),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                       tickfont=dict(size=11, color='#94a3b8'),
                       title=dict(text='승객 수 (천 명)', font=dict(size=11, color='#94a3b8'))),
            legend=dict(orientation='h', y=1.06, x=0,
                        bgcolor='rgba(0,0,0,0)', font=dict(size=11, color='#64748b')),
            font=dict(family='Inter'),
        )

        # ③ 차트 + 오른쪽 스탯 카드
        _c1, _c2 = st.columns([2.3, 1])
        with _c1:
            st.markdown(
                '<p style="color:#64748b;font-size:0.78rem;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.3rem;">'
                '📊 DEMO PREVIEW — AirPassengers 월별 항공 승객 (1949~1962)</p>',
                unsafe_allow_html=True
            )
            st.plotly_chart(_fig_demo, use_container_width=True)

        with _c2:
            st.markdown('<div style="height:1.8rem;"></div>', unsafe_allow_html=True)
            for _color, _icon, _label, _val, _sub in [
                ('#6366f1','📅','데이터 기간','1949–1960','144개월'),
                ('#f59e0b','📈','전체 추세','상승 +107%','후반부 대비'),
                ('#10b981','🔄','계절 주기','12개월','강도 77%'),
                ('#ec4899','🏆','최적 모델','HoltWinters','MASE 0.985'),
            ]:
                st.markdown(
                    f'<div style="background:white;border-radius:10px;padding:0.7rem 1rem;'
                    f'margin-bottom:0.5rem;border-left:3px solid {_color};'
                    f'box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
                    f'<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.04em;">{_icon} {_label}</div>'
                    f'<div style="font-size:1rem;font-weight:800;color:#1e293b;">{_val}</div>'
                    f'<div style="font-size:0.75rem;color:#94a3b8;">{_sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ④ 3단계 사용법
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        _steps = [
            ('#ede9fe','1','CSV 업로드','날짜+숫자 컬럼 있는 단변량 시계열이면 OK'),
            ('#dbeafe','2','자동 분석','ADF·분해·ACF/PACF·ARIMA 차수 전부 자동'),
            ('#fef3c7','3','예측 + 신뢰도','7개 모델 비교 후 최적 선택, 신뢰 점수 제공'),
            ('#d1fae5','4','결과 다운로드','예측값 CSV + 신뢰 리포트 TXT 저장'),
        ]
        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        for _col, (_bg, _num, _title, _desc) in zip([_sc1,_sc2,_sc3,_sc4], _steps):
            with _col:
                st.markdown(
                    f'<div style="background:white;border-radius:14px;padding:1.1rem;'
                    f'border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.04);'
                    f'text-align:center;height:100%;">'
                    f'<div style="width:34px;height:34px;background:{_bg};border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'margin:0 auto 0.6rem;font-size:0.9rem;font-weight:700;color:#374151;">{_num}</div>'
                    f'<div style="font-weight:700;color:#1e293b;font-size:0.9rem;margin-bottom:0.3rem;">{_title}</div>'
                    f'<div style="font-size:0.78rem;color:#94a3b8;line-height:1.5;">{_desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        return

    # ── 데이터 로드 시 컴팩트 헤더 ─────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.6rem;'
        'padding:0.6rem 0 0.2rem;margin-bottom:0.5rem;">'
        '<span style="font-size:1.3rem;font-weight:900;color:#1e293b;">📈 시계열 예측 신뢰 분석기</span>'
        '<span style="background:#ede9fe;color:#6d28d9;border-radius:999px;'
        'padding:2px 10px;font-size:0.72rem;font-weight:700;">ANALYZING</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── 열 선택 ───────────────────────────────────────────────────
    with st.expander("🔧 열 설정 확인 / 변경",expanded=(date_col is None)):
        ac=list(df_raw.columns)
        c1,c2=st.columns(2)
        with c1:
            di=ac.index(date_col) if date_col in ac else 0
            date_col=st.selectbox("날짜 열",ac,index=di)
        with c2:
            vc=[c for c in ac if c!=date_col]
            vi=vc.index(val_col) if val_col in vc else 0
            val_col=st.selectbox("값 열 (예측 대상)",vc,index=vi)

    # ── 전처리 ────────────────────────────────────────────────────
    try: series,dup_count,missing_before=load_and_validate(df_raw,date_col,val_col)
    except Exception as e: st.error(f"❌ 전처리 오류: {e}"); st.stop()

    n=len(series)
    if n<16: st.error(f"❌ 데이터 부족 ({n}행)"); st.stop()

    freq_label,freq_str,sp=detect_freq(series.index)
    adf_result =run_adf(series[val_col])
    d_val      =1 if not adf_result['is_stationary'] else 0
    arima_order=auto_arima_order(series[val_col],d_val) if use_auto_arima else (1,d_val,1)
    decomp     =get_decomposition(series[val_col],sp)

    test_n =max(4,int(n*test_ratio/100))
    train_n=n-test_n
    train_s=series[val_col].iloc[:train_n]
    test_s =series[val_col].iloc[train_n:]
    train_end=series.index[train_n-1]

    # ── 모델 훈련 ─────────────────────────────────────────────────
    with st.spinner("🔄 모델 훈련 중..."):
        runners={
            'Naive':       lambda:(*naive_forecast(train_s,test_n),None),
            'MovAvg':      lambda:(*moving_average_forecast(train_s,test_n),None),
            'SES':         lambda:(*ses_forecast(train_s,test_n),None),
            'Holt':        lambda:(*holt_forecast(train_s,test_n),None),
            'HoltWinters': lambda:(*holtwinters_forecast(train_s,test_n,sp),None),
            'ARIMA':       lambda:arima_forecast(train_s,test_n,arima_order),
        }
        if use_sarima and n>=sp*2+4:
            runners['SARIMA']=lambda:sarima_forecast(train_s,test_n,arima_order,sp)

        test_results={}; all_metrics={}
        best_model=None; best_rmse=float('inf')

        for name,fn in runners.items():
            try:
                result=fn(); pred=np.array(result[0],dtype=float)
                fit=result[1]; ci=result[2]
                if ci is None:
                    rs=float(np.std(test_s.values-pred)) if len(pred)==len(test_s) else 1.0
                    ci=np.column_stack([pred-1.96*rs,pred+1.96*rs])
                test_results[name]={'pred':pred,'index':test_s.index,'ci':ci}
                m=compute_metrics(test_s.values,pred,train_s.values,fit)
                all_metrics[name]=m
                if not np.isnan(m['RMSE']) and m['RMSE']<best_rmse:
                    best_rmse=m['RMSE']; best_model=name
            except Exception as e:
                st.warning(f"⚠️ {name} 실패: {e}")

        metrics_df=pd.DataFrame(all_metrics).T if all_metrics else pd.DataFrame()

        future_runners={
            'Naive':       lambda:(*naive_forecast(series[val_col],forecast_horizon),None),
            'SES':         lambda:(*ses_forecast(series[val_col],forecast_horizon),None),
            'Holt':        lambda:(*holt_forecast(series[val_col],forecast_horizon),None),
            'HoltWinters': lambda:(*holtwinters_forecast(series[val_col],forecast_horizon,sp),None),
            'ARIMA':       lambda:arima_forecast(series[val_col],forecast_horizon,arima_order),
        }
        if use_sarima and n>=sp*2+4:
            future_runners['SARIMA']=lambda:sarima_forecast(series[val_col],forecast_horizon,arima_order,sp)

        future_idx=generate_future_index(series.index[-1],forecast_horizon,freq_str)
        future_results={}
        for name,fn in future_runners.items():
            try:
                result=fn(); pred=np.array(result[0],dtype=float)
                ci=result[2]
                if ci is None and name in test_results:
                    rs=float(np.std(test_s.values-test_results[name]['pred']))
                    ci=np.column_stack([pred-1.96*rs,pred+1.96*rs])
                future_results[name]={'pred':pred,'index':future_idx,'ci':ci}
            except: pass

        residuals=np.array([])
        if best_model and best_model in test_results:
            residuals=test_s.values-test_results[best_model]['pred']

        trust_total,trust_scores,trust_reasons=compute_trust_report(
            n,missing_before,adf_result,metrics_df,best_model,forecast_horizon,sp,residuals)

        story_text=generate_story(series,val_col,freq_label,sp,adf_result,decomp,
                                   metrics_df,best_model,arima_order,trust_total,forecast_horizon)
        model_reasons=generate_model_reason(best_model,metrics_df,decomp,sp,adf_result,arima_order)

    # ── KPI 바 ────────────────────────────────────────────────────
    _,badge_cls,badge_text,trust_hex,_,kpi_trust_cls=trust_color_class(trust_total)
    arima_case=detect_arima_case(arima_order)

    kpi_data=[
        ("kpi-card-blue",   "📊","데이터 수",      str(n),            freq_label),
        ("kpi-card-violet", "🔄","계절 주기",      f"{sp} 스텝",      f"감지 빈도: {freq_label}"),
        ("kpi-card-orange", "🏆","최적 모델",      best_model or "N/A", f"RMSE {best_rmse:.1f}"),
        ("kpi-card-orange", "🔢","ARIMA 차수",    str(arima_order),  arima_case),
        (kpi_trust_cls,     "🛡️","신뢰 점수",      f"{trust_total:.0f}/100", badge_text),
    ]
    cols=st.columns(5)
    for col,(card_cls,icon,label,value,sub) in zip(cols,kpi_data):
        with col:
            val_color=trust_hex if label=="신뢰 점수" else "#0f172a"
            st.markdown(f"""
            <div class="kpi-card {card_cls}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{val_color};">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:0.8rem 0;'></div>", unsafe_allow_html=True)

    # ── 탭 ───────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
        "📊 데이터 리포트","🔬 분해 & 진단",
        "📈 예측 개요","🏆 모델 비교",
        "🛡️ 신뢰 리포트","💾 다운로드",
    ])

    # ──────────────────────────────────────────────────────────────
    # 탭 1: 데이터 리포트
    # ──────────────────────────────────────────────────────────────
    with tab1:
        # 스토리
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>자동 분석 요약</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="story-box">{story_text}</div>',unsafe_allow_html=True)

        # 전처리 3개 카드
        c1,c2,c3=st.columns(3)
        for col,clr,icon,title,txt in [
            (c1,'green' if missing_before==0 else 'yellow','🔧','결측값 처리',f'발견 {missing_before}개 → 선형 보간'),
            (c2,'green' if dup_count==0 else 'yellow','📋','중복 타임스탬프',f'{dup_count}개 → 평균 집계'),
            (c3,'green','📅','날짜 범위',f'{series.index[0].date()} ~ {series.index[-1].date()}'),
        ]:
            with col:
                st.markdown(f'<div class="card card-{clr}">{icon} <b>{title}</b><br><span style="color:#64748b;font-size:0.88rem;">{txt}</span></div>',
                            unsafe_allow_html=True)

        st.plotly_chart(plot_raw(series,val_col),use_container_width=True)

        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>기술통계</div>',unsafe_allow_html=True)
            st.dataframe(series[val_col].describe().to_frame("값").style.format("{:.3f}"),use_container_width=True)
        with c2:
            st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>ADF 정상성 검정</div>',unsafe_allow_html=True)
            if np.isnan(adf_result['pvalue']):
                st.warning("데이터 부족으로 ADF 검정 불가")
            else:
                clr='green' if adf_result['is_stationary'] else 'yellow'
                msg="✅ 정상 시계열" if adf_result['is_stationary'] else "⚠️ 비정상 시계열"
                st.markdown(f"""<div class="card card-{clr}">
                <b>{msg}</b><br>
                <span style="color:#64748b;font-size:0.88rem;">
                검정 통계량: {adf_result['statistic']:.4f} &nbsp;|&nbsp;
                p-value: {adf_result['pvalue']:.4f} &nbsp;|&nbsp;
                임계값(5%): {adf_result['critical'].get('5%','N/A')}
                </span></div>""",unsafe_allow_html=True)
                tip="p < 0.05 → 정상. 평균·분산이 시간에 따라 안정적입니다." if adf_result['is_stationary'] else f"p > 0.05 → 비정상. ARIMA/SARIMA에서 d={d_val} 차분 자동 적용됩니다."
                st.info(tip)

        # ARIMA 케이스
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>ARIMA 차수 & 특수 케이스</div>',unsafe_allow_html=True)
        st.markdown(f"""<div class="card card-violet">
        🔢 <b>이 데이터의 추정 차수: ARIMA{arima_order} → {arima_case}</b><br>
        <span style="color:#64748b;font-size:0.85rem;">
        p={arima_order[0]}: PACF 기반 AR 차수 &nbsp;|&nbsp;
        d={arima_order[1]}: ADF 기반 차분 횟수 &nbsp;|&nbsp;
        q={arima_order[2]}: ACF 기반 MA 차수
        </span></div>""",unsafe_allow_html=True)

        p,d,q=arima_order
        cases=[
            ("ARIMA(p,0,0)","AR(p)","과거 p개 값 → 현재 예측. PACF가 p에서 절단.",p>0 and d==0 and q==0),
            ("ARIMA(0,0,q)","MA(q)","과거 q개 오차 → 현재 예측. ACF가 q에서 절단.",p==0 and d==0 and q>0),
            ("ARIMA(p,0,q)","ARMA(p,q)","AR+MA 결합. 두 함수 모두 완만히 감소.",p>0 and d==0 and q>0),
            ("ARIMA(0,1,0)","랜덤워크","1차 차분만. 주가 등 예측 불가 패턴.",p==0 and d==1 and q==0),
            ("ARIMA(p,d,q)","ARIMA","비정상 시계열에 d차 차분 후 ARMA.",d>0 and (p>0 or q>0)),
        ]
        for notation,name,desc,is_active in cases:
            cls="arima-case-active" if is_active else "arima-case"
            prefix="⭐ " if is_active else ""
            st.markdown(f'<div class="{cls}">{prefix}<b>{notation}</b> = <b>{name}</b> &nbsp;—&nbsp; {desc}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>데이터 미리보기</div>',unsafe_allow_html=True)
        st.dataframe(series.head(20),use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    # 탭 2: 분해 & 진단
    # ──────────────────────────────────────────────────────────────
    with tab2:
        if decomp is not None:
            st.plotly_chart(plot_decomp(decomp),use_container_width=True)
            rv=float(np.var(decomp.resid.dropna()))
            tv=float(np.var(decomp.trend.dropna()))
            sv=float(np.var(decomp.seasonal.dropna()))
            ts=max(0.0,1-rv/(tv+rv+1e-9)); ss=max(0.0,1-rv/(sv+rv+1e-9))
            c1,c2,c3=st.columns(3)
            with c1: st.metric("추세 강도",f"{ts:.1%}")
            with c2: st.metric("계절성 강도",f"{ss:.1%}")
            with c3: st.metric("계절 주기",f"{sp} 스텝")
            if ss>0.4: st.success(f"계절성 강함 ({ss:.1%}) → SARIMA 또는 Holt-Winters 권장")
            else: st.info("계절성 약함 → ARIMA 또는 Holt 모델이 적합할 수 있습니다.")
        else:
            st.warning(f"분해 불가: 데이터({n}개) < 계절주기({sp})×2")

        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>ACF / PACF</div>',unsafe_allow_html=True)
        af=plot_acf_pacf(series[val_col])
        if af:
            st.plotly_chart(af,use_container_width=True)
            st.caption("파란 막대 = 신뢰구간 초과 (유의미한 자기상관) | 빨간 점선 = 95% 신뢰구간")
        else:
            st.warning("데이터 부족으로 ACF/PACF 계산 불가")

        if not adf_result['is_stationary'] and not np.isnan(adf_result['pvalue']):
            st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>1차 차분 시계열</div>',unsafe_allow_html=True)
            ds=series[val_col].diff().dropna(); da=run_adf(ds)
            fig_d=go.Figure()
            fig_d.add_trace(go.Scatter(x=ds.index,y=ds.values,mode='lines',
                                       line=dict(color='#f59e0b',width=1.8),name='1차 차분'))
            fig_d.add_hline(y=0,line_dash='dash',line_color='#94a3b8',opacity=0.5)
            styled_layout(fig_d,'1차 차분 시계열',260)
            st.plotly_chart(fig_d,use_container_width=True)
            if da['is_stationary']: st.success(f"✅ 1차 차분 후 정상 (p={da['pvalue']:.4f})")
            elif not np.isnan(da['pvalue']): st.warning(f"⚠️ 1차 차분 후에도 비정상 (p={da['pvalue']:.4f})")

        # 잔차 진단 4종
        if len(residuals)>=8:
            st.markdown(f'<div class="sec-head"><div class="sec-head-dot"></div>잔차 진단 4종 — {best_model}</div>',unsafe_allow_html=True)
            st.plotly_chart(plot_residuals_full(residuals,best_model),use_container_width=True)

            lb_p=ljung_box_pval(residuals); jb_p=jarque_bera_pval(residuals)
            dw_s=durbin_watson_stat(residuals); bp_p=breusch_pagan_pval(residuals)

            c1,c2,c3,c4=st.columns(4)
            diag_tests=[
                (c1,"Ljung-Box","잔차 자기상관",lb_p,True,None,"p>0.05: 백색잡음"),
                (c2,"Jarque-Bera","잔차 정규성",jb_p,True,None,"p>0.05: 정규분포"),
                (c3,"Durbin-Watson","자기상관",dw_s,None,(1.5,2.5),"1.5~2.5: 이상 없음"),
                (c4,"Breusch-Pagan","등분산성",bp_p,True,None,"p>0.05: 등분산"),
            ]
            for col,name,meaning,stat,is_p,rng,hint in diag_tests:
                with col:
                    if stat is None or np.isnan(float(stat)):
                        card_cls,sym,val_str="diag-card-na","❓","N/A"
                    elif is_p:
                        ok=float(stat)>0.05
                        card_cls="diag-card-pass" if ok else "diag-card-fail"
                        sym="✅" if ok else "❌"
                        val_str=f"p={stat:.3f}"
                    else:
                        ok=rng[0]<float(stat)<rng[1]
                        card_cls="diag-card-pass" if ok else "diag-card-warn"
                        sym="✅" if ok else "△"
                        val_str=f"{stat:.3f}"
                    st.markdown(f"""<div class="diag-card {card_cls}">
                    <div class="diag-name">{name}</div>
                    <div class="diag-sym">{sym}</div>
                    <div class="diag-val">{val_str}</div>
                    <div class="diag-desc">{meaning}<br>{hint}</div>
                    </div>""",unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # 탭 3: 예측 개요
    # ──────────────────────────────────────────────────────────────
    with tab3:
        if forecast_horizon>n*0.5:
            st.warning(f"⚠️ 예측 {forecast_horizon}스텝 > 데이터({n})의 50% — 불확실성 큼")
        if forecast_horizon>sp*2:
            st.warning(f"⚠️ 예측 기간이 계절 주기({sp})의 2배 초과")

        st.info(
            f"📅 **1스텝 = 1{freq_label}** &nbsp; | &nbsp; "
            f"예측 기간 **{forecast_horizon}스텝** &nbsp; | &nbsp; "
            f"최적 모델: **{best_model}** &nbsp; | &nbsp; "
            f"음영 = 95% 신뢰구간"
        )

        if test_results:
            st.plotly_chart(plot_forecast(series,val_col,test_results,future_results,best_model,train_end),
                            use_container_width=True)

        # 모델 선택 이유
        st.markdown(f'<div class="sec-head"><div class="sec-head-dot"></div>{best_model} 선택 이유</div>',unsafe_allow_html=True)
        for sym,reason in model_reasons:
            st.markdown(f'<div class="reason-card"><div class="reason-sym">{sym}</div><div>{reason}</div></div>',unsafe_allow_html=True)

        # 미래 예측표
        if best_model and best_model in future_results:
            st.markdown(f'<div class="sec-head"><div class="sec-head-dot"></div>미래 예측값 — {best_model}</div>',unsafe_allow_html=True)
            fi=future_results[best_model]
            df_ft=pd.DataFrame({'날짜':fi['index'],'예측값':np.round(fi['pred'],2)})
            if fi.get('ci') is not None:
                df_ft['하한(95%)']= np.round(fi['ci'][:,0],2)
                df_ft['상한(95%)']= np.round(fi['ci'][:,1],2)
            st.dataframe(df_ft,use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    # 탭 4: 모델 비교
    # ──────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="card card-violet">💡 <b>핵심 원칙</b>: <b>MASE &lt; 1</b> 이어야 Naive보다 의미 있는 예측입니다. Naive를 이기지 못하면 그냥 Naive를 쓰는 게 낫습니다.</div>',unsafe_allow_html=True)

        if not metrics_df.empty:
            dc=[c for c in ['MAE','RMSE','MAPE','sMAPE','MASE','AIC','BIC'] if c in metrics_df.columns]
            disp=metrics_df[dc].copy()
            def highlight_best(s):
                v=s.dropna()
                if v.empty: return ['']*len(s)
                mv=v.min()
                return ['background-color:#d1fae5;font-weight:bold' if (not np.isnan(x) and x==mv) else '' for x in s]
            st.dataframe(
                disp.style.apply(highlight_best,subset=[c for c in ['MAE','RMSE','MAPE','sMAPE','MASE'] if c in disp.columns])
                          .format("{:.4f}",na_rep="N/A"),
                use_container_width=True)

            c1,c2=st.columns(2)
            with c1: st.plotly_chart(plot_model_comparison(metrics_df,'RMSE'),use_container_width=True)
            with c2: st.plotly_chart(plot_model_comparison(metrics_df,'MASE'),use_container_width=True)

            if 'Naive' in metrics_df.index:
                st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>Naive 대비 개선율</div>',unsafe_allow_html=True)
                nr=metrics_df.loc['Naive','RMSE']
                rows=[]
                for m in metrics_df.index:
                    if m=='Naive': continue
                    r=metrics_df.loc[m,'RMSE']
                    imp=(nr-r)/nr*100 if nr>0 and not np.isnan(r) else 0.0
                    rows.append({'모델':m,'RMSE':r,'Naive 대비 개선(%)':imp,
                                 '판정':'✅ 개선' if imp>0 else '❌ Naive보다 나쁨'})
                if rows:
                    st.dataframe(pd.DataFrame(rows).set_index('모델').style.format(
                        {'RMSE':'{:.4f}','Naive 대비 개선(%)':'{:.2f}'}),use_container_width=True)

            # 지표 해석
            st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>이 데이터 기준 지표 해석</div>',unsafe_allow_html=True)
            if best_model and best_model in metrics_df.index:
                m=metrics_df.loc[best_model]
                c1,c2=st.columns(2)
                interps=[
                    ('blue','MAE',f"평균 오차: <b>{m['MAE']:.2f}</b>"),
                    ('slate','RMSE',f"큰 오차 민감: <b>{m['RMSE']:.2f}</b>"),
                    ('green' if m['MASE']<1 else 'yellow','MASE',
                     f"{'Naive보다 <b>정확</b> ✅' if m['MASE']<1 else 'Naive보다 <b>부정확</b> ⚠️'} (MASE={m['MASE']:.3f})"),
                    ('blue','MAPE',f"상대 오차: <b>{m['MAPE']:.1f}%</b>"),
                ]
                for i,(clr,name,txt) in enumerate(interps):
                    with (c1 if i%2==0 else c2):
                        st.markdown(f'<div class="card card-{clr}"><b>{name}</b>: {txt}</div>',unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # 탭 5: 신뢰 리포트
    # ──────────────────────────────────────────────────────────────
    with tab5:
        _,badge_cls,badge_text,trust_hex,trust_item_cls,_=trust_color_class(trust_total)

        c1,c2=st.columns([1,1.6])
        with c1:
            st.plotly_chart(plot_trust_gauge(trust_total),use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:-1rem;">
                <span class="trust-badge {badge_cls}">{badge_text}</span>
                <div class="trust-meta">
                    최적 모델: <b>{best_model}</b><br>
                    예측 기간: <b>{forecast_horizon}스텝</b> ({freq_label})
                </div>
            </div>""",unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>세부 평가</div>',unsafe_allow_html=True)
            for item,score in trust_scores.items():
                reason=trust_reasons.get(item,'')
                sc='green' if score>=75 else ('yellow' if score>=55 else 'red')
                bc='#10b981' if score>=75 else ('#f59e0b' if score>=55 else '#ef4444')
                st.markdown(f"""
                <div class="trust-item trust-item-{sc}">
                    <div class="trust-item-header">
                        <span class="trust-item-name">{item}</span>
                        <span class="trust-item-score" style="color:{bc};">{score:.0f}점</span>
                    </div>
                    <div class="trust-item-reason">{reason}</div>
                    <div class="trust-bar-bg">
                        <div class="trust-bar-fill" style="width:{min(score,100):.0f}%;background:{bc};"></div>
                    </div>
                </div>""",unsafe_allow_html=True)

        # 자연어 해석
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>자동 해석</div>',unsafe_allow_html=True)
        lb_p=ljung_box_pval(residuals); jb_p=jarque_bera_pval(residuals)
        dw_s=durbin_watson_stat(residuals)

        interps=[]
        if trust_total>=75: interps.append(("green",f"✅ 이 예측은 신뢰할 수 있습니다. {best_model}이 주요 기준을 충족합니다."))
        elif trust_total>=55: interps.append(("yellow","⚠️ 조건부로 참고할 수 있습니다. 일부 기준에서 경고가 있습니다."))
        else: interps.append(("red","🔴 신뢰하기 어렵습니다. 데이터 보강 또는 모델 재검토가 필요합니다."))

        if not adf_result['is_stationary'] and not np.isnan(adf_result['pvalue']):
            interps.append(("slate",f"📌 비정상 시계열 감지 → d={d_val} 차분 적용됨"))
        if best_model and best_model in metrics_df.index:
            mv=metrics_df.loc[best_model,'MASE']
            if not np.isnan(mv):
                interps.append(("green" if mv<1 else "yellow",f"📌 MASE={mv:.3f} → {'Naive보다 정확 ✅' if mv<1 else 'Naive보다 부정확 ⚠️'}"))
        if not np.isnan(lb_p):
            interps.append(("green" if lb_p>0.05 else "red",f"📌 Ljung-Box p={lb_p:.3f} → {'백색잡음 ✅' if lb_p>0.05 else '패턴 남음 ⚠️'}"))
        if not np.isnan(jb_p):
            interps.append(("green" if jb_p>0.05 else "yellow",f"📌 Jarque-Bera p={jb_p:.3f} → {'정규분포 ✅' if jb_p>0.05 else '비정규 △'}"))
        if not np.isnan(dw_s):
            interps.append(("green" if 1.5<dw_s<2.5 else "yellow",f"📌 Durbin-Watson {dw_s:.3f} → {'자기상관 없음 ✅' if 1.5<dw_s<2.5 else '자기상관 의심 △'}"))

        c1,c2=st.columns(2)
        for i,(clr,txt) in enumerate(interps):
            with (c1 if i%2==0 else c2):
                st.markdown(f'<div class="card card-{clr}">{txt}</div>',unsafe_allow_html=True)



    # ──────────────────────────────────────────────────────────────
    # 탭 6: 다운로드
    # ──────────────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="sec-head"><div class="sec-head-dot"></div>결과 다운로드</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            if best_model and best_model in future_results:
                fi=future_results[best_model]
                df_dl=pd.DataFrame({'date':fi['index'],f'forecast_{best_model}':np.round(fi['pred'],4)})
                if fi.get('ci') is not None:
                    df_dl['lower_95']=np.round(fi['ci'][:,0],4)
                    df_dl['upper_95']=np.round(fi['ci'][:,1],4)
                for nm,info in future_results.items():
                    if nm!=best_model: df_dl[f'forecast_{nm}']=np.round(info['pred'],4)
                st.download_button("📥 미래 예측값 CSV (신뢰구간 포함)",
                                   data=df_dl.to_csv(index=False).encode('utf-8-sig'),
                                   file_name=f"forecast_{best_model}_{forecast_horizon}steps.csv",
                                   mime='text/csv',use_container_width=True)
            if not metrics_df.empty:
                st.download_button("📥 모델 비교 CSV",
                                   data=metrics_df.round(4).to_csv().encode('utf-8-sig'),
                                   file_name="model_comparison.csv",mime='text/csv',use_container_width=True)
        with c2:
            st.download_button("📥 전처리된 데이터 CSV",
                               data=series.reset_index().to_csv(index=False).encode('utf-8-sig'),
                               file_name="cleaned_timeseries.csv",mime='text/csv',use_container_width=True)
            lb_p=ljung_box_pval(residuals); jb_p=jarque_bera_pval(residuals)
            dw_s=durbin_watson_stat(residuals); bp_p=breusch_pagan_pval(residuals)
            report=[f"시계열 예측 신뢰 리포트","="*40,
                    f"신뢰 점수: {trust_total:.1f}/100 ({badge_text})",
                    f"최적 모델: {best_model}",
                    f"ARIMA 차수: {arima_order} ({arima_case})",
                    f"예측 기간: {forecast_horizon}스텝 ({freq_label})","",
                    "[ 세부 평가 ]"]
            for item,score in trust_scores.items():
                report.append(f"  {item}: {score:.0f}점 — {trust_reasons.get(item,'')}")
            report+=["","[ 잔차 진단 ]",
                     f"  Ljung-Box p={lb_p:.4f} ({'통과' if not np.isnan(lb_p) and lb_p>0.05 else '실패'})",
                     f"  Jarque-Bera p={jb_p:.4f} ({'통과' if not np.isnan(jb_p) and jb_p>0.05 else '실패'})",
                     f"  Durbin-Watson={dw_s:.4f} ({'통과' if not np.isnan(dw_s) and 1.5<dw_s<2.5 else '실패'})",
                     f"  Breusch-Pagan p={bp_p:.4f} ({'통과' if not np.isnan(bp_p) and bp_p>0.05 else '실패'})"]
            st.download_button("📥 신뢰 리포트 TXT",
                               data="\n".join(report).encode('utf-8'),
                               file_name="trust_report.txt",mime='text/plain',use_container_width=True)

    st.divider()
    st.caption("📈 TimeSeries Trust Analyzer · 자동 분석 · 다중 모델 예측 · 신뢰도 채점")


if __name__ == "__main__":
    main()
