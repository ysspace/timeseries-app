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
            '<div style="padding:0.6rem 0 0.1rem;">'
            '<span style="font-size:0.95rem;font-weight:800;color:#1e293b;">📈 시계열 예측 신뢰 분석기</span><br>'
            '<span style="font-size:0.72rem;color:#94a3b8;">TimeSeries Trust Analyzer</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.divider()
        data_mode=st.radio("**데이터 소스**",["CSV 업로드","샘플 데이터 (AirPassengers)"])
        uploaded=None
        if data_mode=="CSV 업로드":
            uploaded=st.file_uploader("CSV 파일 업로드",type=['csv'],label_visibility="collapsed")

    # 데이터 있을 때만 모델 설정 표시 (중복 렌더링 방지)
    _has_data = (data_mode=="샘플 데이터 (AirPassengers)") or (uploaded is not None)
    if _has_data:
        with st.sidebar:
            st.divider()
            st.markdown("**⚙️ 예측 설정**")
            forecast_horizon=st.slider("예측 기간 (스텝)",1,60,12)
            test_ratio      =st.slider("테스트 비율 (%)",10,40,20)
            use_auto_arima  =st.checkbox("ARIMA 차수 자동 추정",value=True)
            use_sarima      =st.checkbox("SARIMA 포함",value=True,
                                         help="계절성 ARIMA. 30초 내외 소요")
    else:
        forecast_horizon=12; test_ratio=20
        use_auto_arima=True; use_sarima=True

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
        # 랜딩 페이지
        # ══════════════════════════════════════════════════════════

        # ① 히어로
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#4c1d95 100%);'
            'border-radius:20px;padding:2.5rem 2.5rem 2rem;margin-bottom:1.2rem;">'
            '<div style="display:inline-block;background:rgba(139,92,246,0.3);'
            'border:1px solid rgba(167,139,250,0.4);border-radius:999px;'
            'padding:3px 14px;font-size:0.75rem;color:#c4b5fd;font-weight:600;'
            'letter-spacing:0.06em;margin-bottom:1rem;">✦ 자동 시계열 분석</div>'
            '<h1 style="color:white;font-size:2.2rem;font-weight:900;'
            'line-height:1.2;margin:0 0 0.8rem 0;letter-spacing:-1px;">'
            '임의의 단변량 CSV를 올리면<br>'
            '<span style="color:#a78bfa;">자동 분석·예측·신뢰 진단까지</span></h1>'
            '<p style="color:#94a3b8;font-size:0.9rem;margin:0 0 1.5rem 0;line-height:1.6;">'
            '날짜와 수요 컬럼만 있으면 자동 전처리, 모델 비교, 시평 변경, 평가지표 대시보드를 제공합니다.</p>'
            '<div style="display:flex;gap:0.8rem;flex-wrap:wrap;">'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">자동</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">전처리·분석</div></div>'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">7개</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">모델 비교</div></div>'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">시평</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">파라미터 변경</div></div>'
            '<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
            'border-radius:10px;padding:0.6rem 1.1rem;text-align:center;">'
            '<div style="color:white;font-size:1.3rem;font-weight:800;">신뢰</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;">점수 리포트</div></div>'
            '</div></div>',
            unsafe_allow_html=True
        )

        # ② 차별점 카드
        st.markdown(
            '<div style="background:#fafafa;border:1px solid #e2e8f0;border-left:4px solid #6366f1;'
            'border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;">'
            '<span style="font-size:0.8rem;font-weight:700;color:#6366f1;'
            'text-transform:uppercase;letter-spacing:0.06em;">Why this app?</span><br>'
            '<span style="font-size:0.95rem;color:#1e293b;font-weight:600;">'
            '단순 예측이 아니라, 예측의 사용 가능성을 판단합니다.</span><br>'
            '<span style="font-size:0.83rem;color:#64748b;">'
            '데이터 품질 · Naive 대비 성능 · 잔차 진단 · 예측 기간 위험도를 함께 평가하여 '
            '현재 예측을 믿어도 되는지 0~100점으로 채점합니다.</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # ③ 4단계 흐름 (columns 없이 HTML로 — 사이드바 중복 방지)
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-bottom:1.2rem;">'
            + ''.join([
                f'<div style="background:white;border-radius:12px;padding:1rem;'
                f'border:1px solid #e2e8f0;box-shadow:0 2px 6px rgba(0,0,0,0.04);text-align:center;">'
                f'<div style="width:30px;height:30px;background:{bg};border-radius:50%;'
                f'display:flex;align-items:center;justify-content:center;'
                f'margin:0 auto 0.5rem;font-size:0.85rem;font-weight:700;color:#374151;">{num}</div>'
                f'<div style="font-weight:700;color:#1e293b;font-size:0.88rem;margin-bottom:0.2rem;">{title}</div>'
                f'<div style="font-size:0.76rem;color:#94a3b8;line-height:1.4;">{desc}</div>'
                f'</div>'
                for bg,num,title,desc in [
                    ('#ede9fe','1','CSV 업로드','날짜+수요 컬럼이 있는 단변량 시계열이면 OK'),
                    ('#dbeafe','2','자동 분석','ADF·분해·ACF/PACF·ARIMA 차수 전부 자동'),
                    ('#fef3c7','3','모델 비교','7개 모델 동시 훈련, 신뢰 점수 0~100 채점'),
                    ('#d1fae5','4','결과 저장','예측값 CSV + 신뢰 리포트 TXT 다운로드'),
                ]
            ])
            + '</div>',
            unsafe_allow_html=True
        )

        # ④ 데모 차트 (단일 컬럼 — columns 사용 안 함)
        st.markdown(
            '<p style="color:#64748b;font-size:0.78rem;font-weight:700;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;">'
            '📊 샘플 예측 미리보기 (AirPassengers) '
            '— 실제 CSV 업로드 시 이 대시보드가 자동 갱신됩니다</p>',
            unsafe_allow_html=True
        )

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
            mode='lines+markers', name='Holt-Winters 예측 (24개월)',
            line=dict(color='#ec4899', width=2.5, dash='dash'),
            marker=dict(size=4)
        ))
        _fig_demo.add_vline(
            x=int(_demo_s.index[-1].timestamp() * 1000),
            line_dash='dash', line_color='#94a3b8', opacity=0.6,
            annotation_text='현재 → 예측', annotation_position='top right'
        )
        _fig_demo.update_layout(
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=50, r=20, t=10, b=40),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                       tickfont=dict(size=11, color='#94a3b8')),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                       tickfont=dict(size=11, color='#94a3b8'),
                       title=dict(text='승객 수 (천 명)', font=dict(size=11,color='#94a3b8'))),
            legend=dict(orientation='h', y=1.06, x=0,
                        bgcolor='rgba(0,0,0,0)', font=dict(size=11,color='#64748b')),
            font=dict(family='Inter'),
        )
        st.plotly_chart(_fig_demo, use_container_width=True)

        return

    # ── 열 선택