import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.stats import gumbel_r
from scipy.optimize import minimize, brentq
import math

# ============================================================
# 1. Gumbel 분석 및 플로팅 함수 모음
# ============================================================
# (이전과 동일한 모든 분석 함수들... 내용은 생략하고 함수 이름만 표기)
def load_dataset_from_text(text: str):
    buf = StringIO(text.strip())
    T = np.array(list(map(float, buf.readline().split())))
    X = np.loadtxt(buf)
    return T, X

def validate_dataset(T, X):
    if T.ndim != 1: return False, "T must be 1D"
    if X.ndim != 2: return False, "X must be 2D (n_samples, n_times)"
    if X.shape[1] != len(T): return False, "X columns must match len(T)"
    if np.any(~np.isfinite(X)): return False, "X contains NaN/Inf"
    return True, ""

def fit_gumbel_per_time(T, X):
    k = len(T)
    mu, beta = np.zeros(k), np.zeros(k)
    for j in range(k):
        loc, scale = gumbel_r.fit(X[:, j])
        mu[j], beta[j] = loc, scale
    return mu, beta

def smooth_mu_power(T, mu_hat):
    z, y = np.log(T), np.log(np.maximum(mu_hat, 1e-12))
    b, loga = np.polyfit(z, y, 1)
    a = np.exp(loga)
    def mu_func(t): return a * (t ** b)
    params = {"a": float(a), "b": float(b)}
    return mu_func, params

def smooth_beta_proportional_mu(T, mu_hat, beta_hat, mu_func):
    ratios = beta_hat / np.maximum(mu_hat, 1e-12)
    C = np.mean(ratios)
    def beta_func(t): return np.maximum(C * mu_func(t), 1e-6)
    params = {"proportional_C": float(C)}
    return beta_func, params

def smooth_beta_polylog(T, beta_hat, deg=1):
    z = np.log(T)
    coeff = np.polyfit(z, beta_hat, deg)
    def beta_func(t): return np.maximum(np.polyval(coeff, np.log(t)), 1e-6)
    params = {"coeff_beta_poly": coeff.tolist()}
    return beta_func, params

def build_smoothed_mu_beta(T, mu_hat, beta_hat, beta_smoothing_method="poly_log"):
    mu_func, mu_params = smooth_mu_power(T, mu_hat)
    if beta_smoothing_method == "proportional_mu":
        beta_func, beta_params = smooth_beta_proportional_mu(T, mu_hat, beta_hat, mu_func)
    elif beta_smoothing_method == "poly_log":
        beta_func, beta_params = smooth_beta_polylog(T, beta_hat, deg=1)
    else:
        raise ValueError(f"Unknown beta_smoothing_method: {beta_smoothing_method}")
    params = {"mu": mu_params, "beta": beta_params}
    return mu_func, beta_func, params

def qq_r2_regression(sample, loc, scale):
    s = np.asarray(sample); s = s[~np.isnan(s)]; n = len(s)
    if n < 2: return 0.0
    y = np.sort(s); p = (np.arange(1, n + 1) - 0.5) / n
    x = gumbel_r.ppf(p, loc=loc, scale=scale) 
    b, a = np.polyfit(x, y, 1); yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

def calc_r2_trend(T, X, mu_vec, beta_vec):
    return np.array([qq_r2_regression(X[:, j], mu_vec[j], beta_vec[j]) for j in range(len(T))])

def solve_time_for_q99(mu_func, beta_func, H, p=0.99, t_min=1.0, t_max=1e7):
    def f(t):
        mu, beta = mu_func(t), beta_func(t)
        if beta <= 0: return -1e9
        return gumbel_r.ppf(p, loc=mu, scale=beta) - H
    try: return float(brentq(f, t_min, t_max))
    except (ValueError, RuntimeError): return np.nan

def failure_prob_from_gumbel_model(mu_func, beta_func, H_FAIL, t_grid):
    F = [1.0 - np.exp(-np.exp(-(H_FAIL - mu_func(t)) / max(beta_func(t), 1e-12))) for t in t_grid]
    return np.clip(F, 0.0, 1.0)

def system_failure_prob(F_unit_grid, n_units):
    F_sys = 1.0 - (1.0 - np.asarray(F_unit_grid))**n_units
    return np.clip(F_sys, 0.0, 1.0)

def fit_weibull_from_cdf_curve(t_grid, F_grid, fixed_m=None, eps=1e-9):
    """
    [수정됨] 안정성을 높이고, 적합도(R²)를 함께 반환합니다.
    """
    t, F = np.asarray(t_grid), np.asarray(F_grid)
    mask = t > 0; t, F = t[mask], F[mask]; F = np.clip(F, eps, 1.0 - eps)
    w = 1.0 / np.clip(F, 1e-3, 1.0)

    if fixed_m is not None:
        m = fixed_m
        def obj(log_eta_scalar):
            eta = np.exp(log_eta_scalar)
            Fw = 1.0 - np.exp(- (t / eta) ** m)
            return np.sum(w * (Fw - F) ** 2)
        res = minimize(obj, np.log(np.median(t)), method="Nelder-Mead")
        eta = np.exp(res.x[0]) if isinstance(res.x, (list, np.ndarray)) else np.exp(res.x)

    else:
        def obj(theta):
            m, eta = np.exp(theta)
            Fw = 1.0 - np.exp(- (t / eta) ** m)
            return np.sum(w * (Fw - F) ** 2)
        init = [np.log(2.0), np.log(np.median(t))]
        res = minimize(obj, init, method="Nelder-Mead")
        m, eta = np.exp(res.x)

    if isinstance(eta, (list, np.ndarray)): eta = eta[0]
    if isinstance(m, (list, np.ndarray)): m = m[0]
    
    # --- [추가] R² 계산 로직 ---
    Fw_final = 1.0 - np.exp(- (t / eta) ** m)
    ss_res = np.sum((Fw_final - F)**2)
    ss_tot = np.sum((F - np.mean(F))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
    return {"shape_m": float(m), "scale_eta": float(eta), "r2": r2}

def fit_weibull_joint_cdf(t_grid, F_grid_list, joint_fixed_m=None, eps=1e-9):
    """
    [수정됨] m 고정 시, 변수 개수가 1개일 때와 여러 개일 때를 구분하여 처리합니다.
    """
    N = len(F_grid_list); t = np.asarray(t_grid)
    mask = t > 0; t = t[mask]
    w_list, F_target_list = [], []
    for F in F_grid_list:
        F_masked = np.asarray(F)[mask]; F_clipped = np.clip(F_masked, eps, 1.0 - eps)
        w = 1.0 / np.clip(F_clipped, 1e-3, 1.0)
        F_target_list.append(F_clipped); w_list.append(w)
        
    if joint_fixed_m is not None:
        # --- m 고정, eta들만 최적화 ---
        m_joint = joint_fixed_m
        def obj(log_etas):
            etas = np.exp(log_etas); total_loss = 0.0
            for i in range(N):
                # eta가 스칼라 값일 경우를 대비
                current_eta = etas if N == 1 else etas[i]
                F_w = 1.0 - np.exp(- (t / current_eta) ** m_joint)
                total_loss += np.sum(w_list[i] * (F_w - F_target_list[i]) ** 2)
            return total_loss
        
        init = [np.log(np.median(t)) for _ in range(N)]
        res = minimize(obj, init, method="Nelder-Mead")
        # res.x가 단일 값일 때와 배열일 때 모두 처리
        etas_joint = [float(e) for e in np.exp(np.atleast_1d(res.x))]

    else:
        # --- m과 eta들 동시 최적화 ---
        def obj(theta):
            m = np.exp(theta[0]); etas = np.exp(theta[1:]); total_loss = 0.0
            for i in range(N):
                F_w = 1.0 - np.exp(- (t / etas[i]) ** m)
                total_loss += np.sum(w_list[i] * (F_w - F_target_list[i]) ** 2)
            return total_loss
        init = [np.log(2.0)] + [np.log(np.median(t)) for _ in range(N)]
        res = minimize(obj, init, method="Nelder-Mead")
        m_joint = float(np.exp(res.x[0]))
        etas_joint = [float(e) for e in np.exp(res.x[1:])]

    # 각 곡선별 적합도(R²) 계산
    r2_list = []
    for i in range(N):
        F_w = 1.0 - np.exp(- (t / etas_joint[i]) ** m_joint)
        ss_res = np.sum((F_w - F_target_list[i])**2)
        ss_tot = np.sum((F_target_list[i] - np.mean(F_target_list[i]))**2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        r2_list.append(r2)
        
    return m_joint, etas_joint, r2_list

def plot_forecast_q99_with_boxplot(T, X, forecast_time, forecast_q99, mu_func, beta_func, p_tail, H_FAIL, tB1, t_max):
    q_model = lambda tt: gumbel_r.ppf(p_tail, loc=mu_func(tt), scale=beta_func(tt))
    q99_emp = np.quantile(X, p_tail, axis=0); q99_model_at_T = np.array([q_model(tt) for tt in T])
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([X[:, j] for j in range(len(T))], positions=T, widths=np.minimum(0.08 * np.diff(np.r_[T, T[-1] + (T[-1] - T[-2])]), 40), showfliers=True, patch_artist=True, zorder=1)
    for box in bp['boxes']: box.set(facecolor='#DCEBFF', edgecolor='#4A6FA5', alpha=0.8)
    for med in bp['medians']: med.set(color='blue', linewidth=2)
    ax.plot(forecast_time, forecast_q99, "r-", lw=2, label=f"Model q{int(p_tail*100)} (forecast)", zorder=5)
    ax.plot(T, q99_model_at_T, "ro", ms=6, label="Model q99 @ observed T", zorder=6)
    ax.plot(T, q99_emp, "kD", ms=6, label="Empirical q99 @ observed T", zorder=7)
    if H_FAIL is not None: ax.axhline(H_FAIL, color="gray", ls="--", lw=1.5, label=f"H_FAIL={H_FAIL}", zorder=3)
    if tB1 is not None and np.isfinite(tB1) and tB1 <= t_max: ax.axvline(tB1, color="b", ls="--", lw=1.5, label=f"B1={tB1:.1f}h", zorder=3)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Max pit depth")
    ax.set_title(f"Forecast to {t_max}h with Pit Depth Distribution", fontsize=14)
    ax.grid(True, alpha=0.4); ax.legend(); ax.set_xlim(0, t_max); ax.set_ylim(bottom=0)
    fig.tight_layout(); return fig

def plot_weibull_pdf(t_grid, m, eta, title_prefix="Unit"):
    pdf = (m / eta) * (t_grid / eta) ** (m - 1) * np.exp(- (t_grid / eta) ** m)
    fig, ax = plt.subplots(figsize=(8, 5))
    color = "r-" if title_prefix == "Unit" else "g-"
    ax.plot(t_grid, pdf, color, lw=2, label=f"{title_prefix} Weibull PDF (m={m:.3f}, η={eta:.1f}h)")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Probability density f(t)")
    ax.set_title(f"{title_prefix} Failure Rate Distribution", fontsize=12)
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout(); return fig

# ===== [추가] 무고장 시험 설계 계산 함수 =====
def calculate_test_time(CL, R, B_life_yr, m, AF, n):
    try:
        # B1 수명 (시간 단위)
        B_life_hr = B_life_yr * 365 * 24
        # 목표 신뢰도(R)를 만족하는 척도모수 eta_field 계산
        eta_field = B_life_hr / ((-math.log(R))**(1/m))
        # 가속 시험에서의 척도모수 eta_test
        eta_test = eta_field / AF
        # 무고장 시험 시간 계산
        chi_sq_factor = -math.log(1 - CL)
        test_time = eta_test * ((chi_sq_factor / n)**(1/m))
        return test_time
    except (ValueError, ZeroDivisionError):
        return np.nan
# ===============================================

# ============================================================
# 2. Streamlit UI 및 상태 관리
# ============================================================
st.set_page_config(layout="wide")
st.title("📊 Gumbel & Weibull Reliability Analysis App")

if 'runs' not in st.session_state:
    st.session_state.runs = []

# --- 사이드바 ---
# (기존 사이드바 코드는 변경 없음, 생략)
with st.sidebar:
    st.header("Run 이름 지정 (선택 사항)")
    run_name_input = st.text_input("이번 분석의 이름을 입력하세요.", placeholder="예: 초기 샘플 분석")
    st.markdown("---")
    st.header("1. 데이터 입력")
    sample_data = "1 264 336 600 768 992\n0.441 42.778 45.272 42.939 54.148 60.564\n2.006 39.362 42.939 54.26 80.216 60.671\n4.238 41.159 51.088 73.058 64.236 77.097\n4.908 47.881 71.839 58.346 43.162 49.545\n4.833 41.809 55.961 63.304 74.216 55.582\n4.708 41.328 48.43 42.97 55.824 45.73"
    data_txt = st.text_area("새로운 데이터를 붙여넣으세요.", value=sample_data, height=200)
    st.header("2. 분석 옵션")
    h_fail = st.number_input("Fail 기준 깊이 (H_FAIL)", value=289.0, format="%.1f")
    p_tail = st.slider("분석 분위수 (P_tail)", 0.90, 0.999, 0.99, 0.001, format="%.3f")
    max_time = st.number_input("최대 예측 시간 (Max Time)", min_value=1000, max_value=200000, value=5000, step=500)
    beta_model_option = st.selectbox('Beta Smoothing 모델 선택', ('poly_log', 'proportional_mu'), index=0, help="`poly_log`가 일반적인 모델이며, `proportional_mu`는 Beta(t)=C*mu(t) 모델입니다.")
    st.markdown("---")
    st.subheader("Weibull 적합 옵션")
    fix_m_checkbox = st.checkbox("Weibull 'm' 값 고정")
    fixed_m_value = None
    if fix_m_checkbox:
        fixed_m_value = st.number_input("고정할 m 값 (Fixed m value)", min_value=0.1, value=2.0, step=0.1, format="%.2f")
    st.markdown("---")
    st.subheader("제품 (System) 신뢰성 분석")
    n_units = st.number_input("FFZ 개수 (Unit n)", min_value=1, value=100, step=10, help="열교환기에 포함된 전체 유닛의 수입니다.")
    col_run, col_reset = st.columns(2)
    run_button = col_run.button("➕ 새로운 Run 추가", use_container_width=True)
    reset_button = col_reset.button("🗑️ 모든 기록 삭제", use_container_width=True)
    if reset_button:
        st.session_state.runs = []
        st.rerun()

# --- Run 실행 로직 ---
# (기존 Run 실행 로직 코드는 변경 없음, 생략)
if run_button and data_txt:
    try:
        T, X = load_dataset_from_text(data_txt)
        is_valid, msg = validate_dataset(T,X)
        if not is_valid: 
            st.error(f"데이터 형식 오류: {msg}")
        else:
            with st.spinner('새로운 Run을 분석 중입니다...'):
                mu_hat, beta_hat = fit_gumbel_per_time(T, X)
                mu_func, beta_func, smooth_params = build_smoothed_mu_beta(T, mu_hat, beta_hat, beta_model_option)
                tB1 = solve_time_for_q99(mu_func, beta_func, H=h_fail, p=p_tail)
                forecast_grid_time = np.linspace(max(1.0, float(np.min(T))), max_time, 400)
                q_model_func = lambda tt: gumbel_r.ppf(p_tail, loc=mu_func(tt), scale=beta_func(tt))
                forecast_grid_q99 = np.array([q_model_func(tt) for tt in forecast_grid_time])
                run_id_counter = len(st.session_state.runs) + 1
                final_run_name = run_name_input.strip() if run_name_input.strip() else f"Run {run_id_counter}"
                new_run_data = {
                    "run_id": run_id_counter, "run_name": final_run_name,
                    "T": T, "X": X, "h_fail": h_fail, "p_tail": p_tail, "max_time": max_time,
                    "mu_func": mu_func, "beta_func": beta_func, "smooth_params": smooth_params,
                    "tB1": tB1, "forecast_time": forecast_grid_time, "forecast_q99": forecast_grid_q99,
                    "beta_model": beta_model_option,
                    "fixed_m": fixed_m_value if fix_m_checkbox else None,
                    "n_units": n_units
                }
                st.session_state.runs.append(new_run_data)
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}", icon="🔥")

# --- 3. 메인 화면 (Tab 시스템 표시) ---
# [수정] 탭 리스트에 '무고장 시험 설계' 추가
tab_list = ["📝 무고장 시험 설계"]
if st.session_state.runs:
    tab_list.append("🌟 종합 비교 (Summary)")
    tab_list.extend([run['run_name'] for run in st.session_state.runs])

tabs = st.tabs(tab_list)

# --- [신규] '무고장 시험 설계' 탭 ---
with tabs[0]:
    st.header("📝 무고장 시험 설계 (Success Run Test Planning)")
    st.info("목표 수명과 신뢰도를 만족함을 입증하기 위해 필요한 최소 시험 시간을 계산합니다.")

    plan_col1, plan_col2 = st.columns(2)
    with plan_col1:
        st.subheader("입력 변수")
        conf_level = st.slider("신뢰 수준 (Confidence Level)", 0.50, 0.99, 0.90, 0.01, "%.2f")
        shape_m = st.number_input("Weibull 형상 모수 (m)", 0.1, 20.0, 2.0, 0.1, "%.2f")
        reliability = st.slider("신뢰도 (Reliability, R)", 0.80, 0.999, 0.99, 0.001, "%.3f")
        b_life_yr = st.number_input("목표 B-Life (년)", 1, 50, 10, 1)
        af = st.number_input("가속 계수 (AF)", min_value=1.0, value=1.0, step=0.1, format="%.1f")
        samples_n = st.number_input("시료 수 (n)", 1, 100, 5, 1)
        
        # 시험 시간 계산
        test_time = calculate_test_time(conf_level, reliability, b_life_yr, shape_m, af, samples_n)
        
        st.markdown("---")
        st.subheader("계산 결과")
        if not np.isnan(test_time):
            st.success(f"**필요 시험 시간: `{test_time:,.1f}` 시간** (약 `{(test_time/24):.1f}` 일)")
        else:
            st.error("계산 중 오류가 발생했습니다. 입력값을 확인하세요.")

    with plan_col2:
        st.subheader("시료 수에 따른 시험 시간 변화")
        
        plot_s_col1, plot_s_col2 = st.columns(2)
        min_n = plot_s_col1.number_input("Min 시료 수 (X축)", 1, 50, 1)
        # [수정] Max 시료 수의 하한을 Min 값에 연동
        max_n = plot_s_col2.number_input("Max 시료 수 (X축)", min_value=min_n + 1, value=100)

        plot_t_col1, plot_t_col2 = st.columns(2)
        min_t = plot_t_col1.number_input("Min 시험 시간 (Y축)", min_value=1, value=1)
        max_t = plot_t_col2.number_input("Max 시험 시간 (Y축)", min_value=min_t + 1, value=20000)

        sample_range = np.arange(min_n, max_n + 1)
        time_range = [calculate_test_time(conf_level, reliability, b_life_yr, shape_m, af, n) for n in sample_range]
        
        fig_plan, ax_plan = plt.subplots()
        ax_plan.plot(sample_range, time_range, marker='', linestyle='-')
        ax_plan.set_xlabel("Number of Samples, n")
        ax_plan.set_ylabel("Test Time, hours")
        ax_plan.set_title("Number of Samples vs Test Time Trade-off")
        ax_plan.grid(True)
        ax_plan.set_xlim(min_n, max_n)
        ax_plan.set_ylim(min_t, max_t)
        st.pyplot(fig_plan)

        # --- [추가] 그래프 데이터 다운로드 버튼 ---
        # 1. Pandas DataFrame으로 데이터 생성
        plan_df = pd.DataFrame({
            'number_of_samples': sample_range,
            'required_test_time_hours': time_range
        })
        
        # 2. DataFrame을 CSV 문자열로 변환
        plan_csv_string = plan_df.to_csv(index=False, lineterminator='\n').encode('utf-8')

        # 3. 다운로드 버튼 생성
        st.download_button(
           label="📥 그래프 데이터 다운로드 (.csv)",
           data=plan_csv_string,
           file_name="success_run_plan_data.csv",
           mime="text/csv",
           use_container_width=True
        )
        # -----------------------------------------

# --- 기존 '종합 비교' 및 '개별 Run' 탭 ---
if st.session_state.runs:
     # '종합 비교' 탭 (tabs[1]이 됨)
    with tabs[1]:
        st.subheader("모든 Run의 성장 예측선 종합 비교")
        fig_sum, ax_sum = plt.subplots(figsize=(10, 6))
        if st.session_state.runs:
            for run in st.session_state.runs:
                label_text = f"{run['run_name']} ({run['beta_model']})"
                ax_sum.plot(run['forecast_time'], run['forecast_q99'], lw=2, label=label_text)
            last_run = st.session_state.runs[-1]
            ax_sum.axhline(last_run['h_fail'], color="gray", ls="--", lw=1.5, label=f"Target H_FAIL ({last_run['h_fail']})")
            ax_sum.set_xlim(0, max([r['max_time'] for r in st.session_state.runs]))
        ax_sum.set_xlabel("Time (h)"); ax_sum.set_ylabel("Max pit depth")
        ax_sum.set_title("Forecast Toplines Comparison", fontsize=14)
        ax_sum.grid(True, alpha=0.4); ax_sum.legend(); ax_sum.set_ylim(bottom=0)
        st.pyplot(fig_sum)

        if len(st.session_state.runs) >= 2:
            st.markdown("---")
            st.subheader("🤝 2-Run 공동 적합 및 가속 계수 분석")
            run_options = {run['run_name']: run for run in st.session_state.runs}
            selected_run_names = st.multiselect("비교할 두 개의 Run을 선택하세요.", options=list(run_options.keys()), help="정확히 두 개의 Run을 선택해야 분석이 활성화됩니다.")

            if len(selected_run_names) != 2:
                st.info("위 목록에서 비교하고 싶은 Run 2개를 선택하면 분석이 시작됩니다.")
            else:
                selected_runs = [run_options[name] for name in selected_run_names]
                st.markdown("##### 1. '고정 m' 기반 추가 분석")
                with st.expander("여기를 눌러 '고정 m' 값으로 공동 적합을 수행하세요."):
                    fixed_m_input = st.number_input("비교에 사용할 공통 m 값", min_value=0.1, value=10.0, step=0.1, format="%.2f", key="joint_fixed_m_input")
                    # --- [수정] 아래 if 문의 들여쓰기를 맞춥니다 ---
                    if st.button("📈 '고정 m'으로 공동 적합 실행"):
                        valid_b1s = [r['tB1'] for r in selected_runs if np.isfinite(r['tB1']) and r['tB1'] > 0]
                        joint_fit_max_time = np.median(valid_b1s) * 1.5 if valid_b1s else max([r['max_time'] for r in selected_runs])
                        weibull_fit_time = np.linspace(0.001, joint_fit_max_time, 800)
                        
                        F_sys_list = [system_failure_prob(failure_prob_from_gumbel_model(run['mu_func'], run['beta_func'], run['h_fail'], weibull_fit_time), run['n_units']) for run in selected_runs]
                        
                        m_joint, etas_joint, r2_list = fit_weibull_joint_cdf(weibull_fit_time, F_sys_list, joint_fixed_m=fixed_m_input)
                        st.session_state.joint_fit_result = {"m": m_joint, "etas": etas_joint, "r2s": r2_list, "run_names": selected_run_names}
                        st.rerun()

                if 'joint_fit_result' in st.session_state and st.session_state.joint_fit_result and set(selected_run_names) == set(st.session_state.joint_fit_result.get("run_names", [])):
                    result = st.session_state.joint_fit_result
                    m_joint, etas_joint, r2_list = result["m"], result["etas"], result["r2s"]
                    st.markdown("##### 2. 공동 적합 결과")
                    joint_col1, joint_col2 = st.columns([1, 1.5])
                    with joint_col1:
                        st.success(f"**적용된 공통 m**: `{m_joint:.3f}`")
                        res_data = [{"Run 이름": name, "척도 모수 (η)": f"{etas_joint[i]:.1f} h", "적합도 (R²)": f"{r2_list[i]*100:.1f}%"} for i, name in enumerate(selected_run_names)]
                        st.dataframe(pd.DataFrame(res_data), use_container_width=True)
                    with joint_col2:
                        joint_plot_max_x = st.slider("그래프 X축 범위 조절", int(max([r['max_time'] for r in selected_runs])), int(max(etas_joint) * 2.5), int(max([r['max_time'] for r in selected_runs])), 2000, key="joint_plot_slider")
                        fig_joint, ax_joint = plt.subplots(figsize=(8, 5))
                        pdf_plot_time = np.linspace(0.001, joint_plot_max_x, 800)
                        for i, name in enumerate(selected_run_names):
                            eta_i = etas_joint[i]
                            pdf = (m_joint / eta_i) * (pdf_plot_time / eta_i)**(m_joint - 1) * np.exp(-(pdf_plot_time / eta_i)**m_joint)
                            ax_joint.plot(pdf_plot_time, pdf, lw=2, label=f"{name} (η={eta_i:.1f}h)")
                        ax_joint.set_title(f"Joint Weibull PDF (Common m={m_joint:.3f})", fontsize=12); ax_joint.legend(); ax_joint.grid(True, alpha=0.3)
                        st.pyplot(fig_joint)

                    st.markdown("##### 3. 가속 계수 (AF)")
                    baseline_run_name = st.selectbox("기준(Baseline) Run 선택", options=selected_run_names, key="af_baseline_select_joint")
                    if baseline_run_name:
                        baseline_idx = selected_run_names.index(baseline_run_name)
                        eta_baseline = etas_joint[baseline_idx]
                        st.info(f"**기준:** `{baseline_run_name}` (η = {eta_baseline:.1f} h)")
                        af_results = []
                        for i, name in enumerate(selected_run_names):
                            if i == baseline_idx: continue
                            eta_i = etas_joint[i]; af = eta_baseline / eta_i
                            af_results.append({"비교 대상": name, "가속 계수 (AF)": f"{af:.2f}", "설명": f"`{baseline_run_name}` 대비 약 **{af:.2f}배** 수명이 짧음" if af > 1 else f"`{baseline_run_name}` 대비 약 **{1/af:.2f}배** 수명이 김"})
                        if af_results: st.dataframe(pd.DataFrame(af_results), use_container_width=True)

    # '개별 Run' 탭 (tabs[2]부터 시작)
    for i, run in enumerate(st.session_state.runs):
        with tabs[i+2]:
            # (기존 '개별 Run' 탭의 모든 코드... 변경 없음, 생략)
            header_col1, header_col2 = st.columns([0.85, 0.15])
            with header_col1: st.header(f"📈 분석 결과: {run['run_name']}")
            with header_col2:
                if st.button(f"🗑️ 이 Run 삭제", key=f"delete_run_{run['run_id']}", use_container_width=True):
                    st.session_state.runs = [r for r in st.session_state.runs if r['run_id'] != run['run_id']]
                    st.rerun()
            col1, col2 = st.columns([2, 1.5])
            with col1:
                st.subheader(f"시간별 최대 공식 깊이 성장 (Unit Level)")
                fig_forecast = plot_forecast_q99_with_boxplot(run['T'], run['X'], run['forecast_time'], run['forecast_q99'], run['mu_func'], run['beta_func'], run['p_tail'], run['h_fail'], run['tB1'], run['max_time'])
                st.pyplot(fig_forecast)
                forecast_df = pd.DataFrame({"time_h": run['forecast_time'], "predicted_q99_depth": run['forecast_q99']})
                csv_string = forecast_df.to_csv(index=False, lineterminator='\n').encode('utf-8')
                st.download_button(label=f"📥 예측선 데이터 다운로드 (.csv)", data=csv_string, file_name=f"{run['run_name']}_forecast.csv", mime="text/csv", use_container_width=True, key=f"dl_btn_{run['run_id']}")
            with col2:
                st.subheader("Unit Weibull 고장률(PDF)")
                
                # --- [수정] Weibull 적합 시간 범위를 B1 수명에 연동 ---
                # B1 수명이 계산 가능한 경우, 그 값의 1.5배까지를 적합 및 플롯 범위로 사용
                # B1 수명이 너무 길거나 계산 불가 시, 기존 max_time을 사용
                if np.isfinite(run['tB1']) and run['tB1'] > 0:
                    fit_and_plot_max_time = run['tB1'] * 1.5
                else:
                    fit_and_plot_max_time = run['max_time']

                weibull_fit_time = np.linspace(0.001, fit_and_plot_max_time, 800)
                # --------------------------------------------------

                f_fail_unit = failure_prob_from_gumbel_model(run['mu_func'], run['beta_func'], run['h_fail'], weibull_fit_time)
                weibull_params_unit = fit_weibull_from_cdf_curve(weibull_fit_time, f_fail_unit, fixed_m=run['fixed_m'])
                
                fig_weibull_unit = plot_weibull_pdf(weibull_fit_time, weibull_params_unit['shape_m'], weibull_params_unit['scale_eta'], title_prefix="Unit")
                st.pyplot(fig_weibull_unit)

                st.caption(f"**적합 결과**: m = `{weibull_params_unit['shape_m']:.3f}`, η = `{weibull_params_unit['scale_eta']:.1f} h`, **R² = `{weibull_params_unit['r2']*100:.1f}%`**")
            st.markdown("---")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.subheader("QQ Regression R²")
                mu_smooth = np.array([run['mu_func'](t) for t in run['T']]); beta_smooth = np.array([run['beta_func'](t) for t in run['T']])
                r2_stage2 = calc_r2_trend(run['T'], run['X'], mu_smooth, beta_smooth)
                r2_df = pd.DataFrame([r2_stage2 * 100], index=["R²(%)"], columns=[f"{int(t)} hr" for t in run['T']])
                st.dataframe(r2_df.style.format("{:.1f}%").background_gradient(cmap='viridis', axis=1))
            with info_col2:
                st.subheader("적합 정보 (Parameters)")
                st.markdown(f"**μ(t) = a * t^b** | a=`{run['smooth_params']['mu']['a']:.4f}`, b=`{run['smooth_params']['mu']['b']:.4f}`")
                beta_p = run['smooth_params']['beta']
                if 'proportional_C' in beta_p: st.markdown(f"**β(t) = C * μ(t)** | C=`{beta_p['proportional_C']:.4f}`")
                else: st.markdown(f"**β(t) = poly(log t)** | Coeff=`{np.round(beta_p['coeff_beta_poly'], 4)}`")
                if np.isfinite(run['tB1']): st.success(f"Unit B1 시간: **{run['tB1']:.1f} hr**")
            st.markdown("---")
            st.subheader(f"🏢 제품 (System) 신뢰성 분석 - {run['n_units']}개 Unit 직렬 모델")
            sys_col1, sys_col2 = st.columns([1, 1.5])
            
            # [수정] Unit Weibull과 동일한 동적 시간 범위 사용
            if np.isfinite(run['tB1']) and run['tB1'] > 0:
                fit_and_plot_max_time_sys = run['tB1'] * 1.5
            else:
                fit_and_plot_max_time_sys = run['max_time']
            weibull_fit_time_sys = np.linspace(0.001, fit_and_plot_max_time_sys, 800)
            f_fail_unit_sys = failure_prob_from_gumbel_model(run['mu_func'], run['beta_func'], run['h_fail'], weibull_fit_time_sys)
            # ---

            f_fail_sys = system_failure_prob(f_fail_unit_sys, run['n_units'])
            weibull_params_sys = fit_weibull_from_cdf_curve(weibull_fit_time_sys, f_fail_sys, fixed_m=run['fixed_m'])
            
            with sys_col1:
                # ... (sys_col1 내부의 st.info, st.success 등은 기존과 동일)
                st.info(f"""**제품 파라미터 도출 결과**\n- **제품 고장 모델**: ...""") # 내용은 생략
                p_fail_target_for_sys_b1 = 1.0 - (0.99 ** (1.0 / run['n_units']))
                p_quantile_for_sys_b1 = 1.0 - p_fail_target_for_sys_b1
                t_sys_B1 = solve_time_for_q99(run['mu_func'], run['beta_func'], H=run['h_fail'], p=p_quantile_for_sys_b1)
                if np.isfinite(t_sys_B1): st.success(f"제품(System) B1 시간: **{t_sys_B1:.1f} hr**")
                else: st.warning("제품 B1 시간을 계산할 수 없습니다.")

            with sys_col2:
                fig_weibull_sys = plot_weibull_pdf(weibull_fit_time_sys, weibull_params_sys['shape_m'], weibull_params_sys['scale_eta'], title_prefix="Product")
                st.pyplot(fig_weibull_sys)
