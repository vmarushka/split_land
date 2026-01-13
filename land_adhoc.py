
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# --- Configuration ---
st.set_page_config(
    page_title="Landing Page Analysis (Bayesian)",
    page_icon="🔮",
    layout="wide"
)

# --- Constants ---
VARIANTS_OF_INTEREST = ['mm-nsp-v1', 'mm-nsp-v2', 'mm-nsp-v4']
DEFAULT_CSV_PATH = 'land_1.csv'

# --- 1. Data Loading & Preprocessing ---

@st.cache_data
def load_data(file_path_or_buffer):
    """Loads and preprocesses data."""
    try:
        df = pd.read_csv(file_path_or_buffer)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Filter Variants
    if 'landingId' in df.columns:
        df = df[df['landingId'].isin(VARIANTS_OF_INTEREST)]
    else:
        st.error("Column 'landingId' not found in data.")
        return None

    # Date Conversion
    date_cols = ['created_at', 'profile_created_at', 'reg_at', 'fo_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Currency Adjustment (Divide by 100)
    if 'amount' in df.columns:
        df['amount'] = df['amount'] / 100.0

    # Product Clustering Logic
    if 'id' in df.columns:
        def assign_cluster(pkg_id):
            if not isinstance(pkg_id, str): return 'other'
            pkg_id = pkg_id.lower()
            
            if pkg_id in ['credits-1550-taxes-v4', 'credits-2950-taxes-v4', 'credits-595-taxes-v4']:
                return 'credits'
            if pkg_id in ['pkg-premium-advanced-taxes-ntf3-6m-v1', 'pkg-premium-advanced-taxes-ntfm25-6m-v1', 'pkg-premium-advanced-taxes-ntfm25-smart-6m-v1']:
                return 'premium 6m'
            if pkg_id == 'pkg-premium-gold-taxes-ntf3-3m':
                return 'gold 3m'
            if pkg_id == 'pkg-premium-gold-taxes-ntf3-6m':
                return 'gold 6m'
            if pkg_id == 'pkg-premium-gold-taxes-ntf3-v2':
                return 'gold 1m'
            if pkg_id in ['pkg-premium-intermidiate-taxes-ntf3-3m-v1', 'pkg-premium-intermidiate-taxes-ntfm25-3m-v1', 'pkg-premium-intermidiate-taxes-ntfm25-smart-3m-v1']:
                return 'premium 3m'
            if pkg_id in ['pkg-premium-standard-taxes-ntf3-1m-v7', 'pkg-premium-standard-taxes-ntfm25-1m-v7', 'pkg-premium-standard-taxes-ntfm25-smart-1m-v7']:
                return 'premium 1m'
            
            return 'other'
        
        df['product_cluster'] = df['id'].apply(assign_cluster)
    
    return df

# ... (Metrics and Stats functions remain unchanged) ...

# --- Visual Analysis --- (Updating Deep Dive Section mostly)

# ... (Rendering functions) ...

# We need to jump to the Deep Dive section in render_dashboard
# For safety, I will target the specific Deep Dive block in a separate replacement or update the whole file if preferred.
# Since I am using replace_file_content with line ranges, I'll restrict this to load_data first, 
# then I will do a second edit for the Deep Dive section to be precise.


# --- 2. Metrics Definition ---

def calculate_metrics(df):
    """Calculates funnel and financial metrics for each variant."""
    
    # 1. Base Counts
    visitors = df['user_id'].nunique()
    onboarding = df[df['profile_created_at'].notnull()]['user_id'].nunique()
    registered = df[df['reg_at'].notnull()]['user_id'].nunique()
    payers = df[df['fo_at'].notnull()]['user_id'].nunique()
    
    # Payers D0
    df_payers = df.dropna(subset=['fo_at', 'reg_at'])
    df_payers_d0 = df_payers[(df_payers['fo_at'] - df_payers['reg_at']) <= pd.Timedelta(hours=24)]
    payers_d0 = df_payers_d0['user_id'].nunique()

    # 2. Financials
    total_amount = df['amount'].sum()
    
    # ARPU (Revenue / Registered)
    arpu = total_amount / visitors if visitors > 0 else 0
    
    # ARPPU (Revenue / Payers)
    arppu = total_amount / payers if payers > 0 else 0

    return {
        'Visitors': visitors,
        'Onboarding Users': onboarding,
        'Registered Users': registered,
        'Payers': payers,
        'Payers (Day 0)': payers_d0,
        'Total Revenue': total_amount,
        'ARPU': arpu,
        'ARPPU': arppu
    }

def get_conversion_rates(metrics):
    """Calculates conversion rates based on metrics dict."""
    visitors = metrics['Visitors']
    onboarding = metrics['Onboarding Users']
    registered = metrics['Registered Users']
    payers = metrics['Payers']
    
    conv_landing_onboarding = (onboarding / visitors * 100) if visitors > 0 else 0
    conv_landing_registration = (registered / visitors * 100) if visitors > 0 else 0
    conv_onboarding_registration = (registered / onboarding * 100) if onboarding > 0 else 0
    conv_registration_payer = (payers / registered * 100) if registered > 0 else 0
    conv_registration_payer_d0 = (metrics['Payers (Day 0)'] / registered * 100) if registered > 0 else 0
    
    return {
        'Landing -> Onboarding': conv_landing_onboarding,
        'Landing -> Registration': conv_landing_registration,
        'Onboarding -> Registration': conv_onboarding_registration,
        'Registration -> Payer': conv_registration_payer,
        'Registration -> Payer D0': conv_registration_payer_d0
    }

# --- 3. Bayesian Statistical Analysis ---

@st.cache_data
def run_bayesian_simulation_proportion(success_c, n_c, success_v, n_v, n_samples=10000):
    """
    Runs Bayesian simulation for proportions using Beta distribution.
    Includes robust error (ValueError: b <= 0) prevention.
    Returns: (prob_v_beats_c, expected_uplift_percent)
    """
    # --- Guard Clauses & Sanitization ---
    n_c, success_c = max(0, int(n_c)), max(0, int(success_c))
    n_v, success_v = max(0, int(n_v)), max(0, int(success_v))
    
    if n_c == 0 or n_v == 0: return 0.5, 0.0
    if success_c > n_c: success_c = n_c
    if success_v > n_v: success_v = n_v
        
    try:
        posterior_c = np.random.beta(1 + success_c, 1 + (n_c - success_c), n_samples)
        posterior_v = np.random.beta(1 + success_v, 1 + (n_v - success_v), n_samples)
        prob_beat = np.mean(posterior_v > posterior_c)
        
        mean_c = np.mean(posterior_c)
        uplift = (np.mean(posterior_v) - mean_c) / mean_c * 100 if mean_c > 0 else 0.0
             
        return prob_beat, uplift
    except Exception:
        return 0.5, 0.0

@st.cache_data
def run_bayesian_simulation_revenue(data_c, data_v, n_samples=10000):
    """
    Runs Bayesian simulation for means using Normal approximation.
    Returns: (prob_v_beats_c, expected_uplift_value) 
    """
    if len(data_c) < 2 or len(data_v) < 2: return 0.5, 0.0

    mu_c, std_c = np.mean(data_c), np.std(data_c, ddof=1)
    mu_v, std_v = np.mean(data_v), np.std(data_v, ddof=1)
    
    sem_c = std_c / np.sqrt(len(data_c))
    sem_v = std_v / np.sqrt(len(data_v))
    
    posterior_c = np.random.normal(mu_c, sem_c, n_samples)
    posterior_v = np.random.normal(mu_v, sem_v, n_samples)
    
    prob_beat = np.mean(posterior_v > posterior_c)
    uplift = np.mean(posterior_v) - np.mean(posterior_c)
    
    return prob_beat, uplift

    return prob_beat, uplift

def run_frequentist_tests(success_c, n_c, success_v, n_v):
    """
    Runs Z-test for proportions.
    Returns: (p_value, uplift_percent)
    """
    if n_c == 0 or n_v == 0: return 1.0, 0.0
    
    count = np.array([success_v, success_c])
    nobs = np.array([n_v, n_c])
    
    try:
        stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
        rate_c = success_c / n_c
        rate_v = success_v / n_v
        uplift = (rate_v - rate_c) / rate_c * 100 if rate_c > 0 else 0.0
        return pval, uplift
    except:
        return 1.0, 0.0

def run_frequentist_means(data_c, data_v):
    """
    Runs T-test for means (Welch's t-test).
    Returns: (p_value, uplift_value)
    """
    if len(data_c) < 2 or len(data_v) < 2: return 1.0, 0.0
    
    try:
        stat, pval = stats.ttest_ind(data_v, data_c, equal_var=False)
        uplift = np.mean(data_v) - np.mean(data_c)
        return pval, uplift
    except:
        return 1.0, 0.0

def run_statistics(df, control_variant='mm-nsp-v1', method='Bayesian'):
    """Runs tests based on selected method."""
    results = {}
    variants = df['landingId'].dropna().unique()
    variants = [v for v in variants if v != control_variant]
    
    df_control = df[df['landingId'] == control_variant]
    m_control = calculate_metrics(df_control)
    
    def get_rev_vec(dframe):
        reg = dframe[dframe['reg_at'].notnull()]['user_id'].unique()
        pay = dframe.groupby('user_id')['amount'].sum().reset_index()
        merged = pd.Series(reg, name='user_id').to_frame().merge(pay, on='user_id', how='left').fillna(0)
        return merged['amount'].values, dframe.dropna(subset=['fo_at'])['amount'].values

    vec_arpu_c, vec_arppu_c = get_rev_vec(df_control)
    
    n_vis_c, n_onb_c, n_reg_c = m_control['Visitors'], m_control['Onboarding Users'], m_control['Registered Users']

    for v in variants:
        df_v = df[df['landingId'] == v]
        m_v = calculate_metrics(df_v)
        vec_arpu_v, vec_arppu_v = get_rev_vec(df_v)
        
        n_vis_v, n_onb_v, n_reg_v = m_v['Visitors'], m_v['Onboarding Users'], m_v['Registered Users']
        
        tests = {}
        
        if method == 'Frequentist (Classical)':
            tests['Landing -> Onboarding'] = run_frequentist_tests(
                m_v['Onboarding Users'], n_vis_v, m_control['Onboarding Users'], n_vis_c) # Note: order matters for ztest input usually, but function handles it
            # Re-checking my helper: count=[v, c], nobs=[v, c].
            # Calling helper: success_c, n_c, success_v, n_v.
            # Helper logic: count = [success_v, success_c]. Correct.
            
            tests['Landing -> Onboarding'] = run_frequentist_tests(
                m_control['Onboarding Users'], n_vis_c, m_v['Onboarding Users'], n_vis_v)
            tests['Landing -> Registration'] = run_frequentist_tests(
                m_control['Registered Users'], n_vis_c, m_v['Registered Users'], n_vis_v)
            tests['Registration -> Payer'] = run_frequentist_tests(
                m_control['Payers'], n_reg_c, m_v['Payers'], n_reg_v)
            tests['ARPU'] = run_frequentist_means(vec_arpu_c, vec_arpu_v)
            tests['ARPPU'] = run_frequentist_means(vec_arppu_c, vec_arppu_v)
            
        else: # Bayesian
            tests['Landing -> Onboarding'] = run_bayesian_simulation_proportion(
                m_control['Onboarding Users'], n_vis_c, m_v['Onboarding Users'], n_vis_v)
            tests['Landing -> Registration'] = run_bayesian_simulation_proportion(
                m_control['Registered Users'], n_vis_c, m_v['Registered Users'], n_vis_v)
            tests['Registration -> Payer'] = run_bayesian_simulation_proportion(
                m_control['Payers'], n_reg_c, m_v['Payers'], n_reg_v)
            tests['ARPU'] = run_bayesian_simulation_revenue(vec_arpu_c, vec_arpu_v)
            tests['ARPPU'] = run_bayesian_simulation_revenue(vec_arppu_c, vec_arppu_v)
        
        results[v] = tests

    return results

# --- 4. Logic & Conclusions ---

def generate_automated_summary(df, stats_results, method, traffic_share_text=""):
    """Generates a text summary."""
    summary = []
    variants = [v for v in df['landingId'].unique() if v != 'mm-nsp-v1']

    is_bayesian = method.startswith('Bayesian')

    for v in variants:
        res = stats_results.get(v, {})
        val, uplift = res.get('Registration -> Payer', (1.0, 0.0) if not is_bayesian else (0.5, 0.0))

        is_sig = (val > 0.95) if is_bayesian else (val < 0.05)

        if is_sig:
            sig_label = f"Prob ({val:.1%})" if is_bayesian else f"p={val:.4f}"
            summary.append(f"**{v}** > Control (Pay Conv, {sig_label}, +{uplift:.1f}%).")

        val_arpu, diff_arpu = res.get('ARPU', (1.0, 0.0) if not is_bayesian else (0.5, 0.0))
        is_sig_arpu = (val_arpu > 0.95) if is_bayesian else (val_arpu < 0.05)

        if is_sig_arpu:
             sig_label_arpu = f"Prob ({val_arpu:.1%})" if is_bayesian else f"p={val_arpu:.4f}"
             summary.append(f"**{v}** > Control (ARPU, {sig_label_arpu}, +${diff_arpu:.2f}).")

    if not summary:
        crit = ">95% probability" if is_bayesian else "p<0.05"
        return f"No strong evidence ({crit}) that any variant beats Control for {traffic_share_text}."
    return " ".join(summary)


# --- 5. UI Layout ---

def render_dashboard():
    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration")
    stat_method = st.sidebar.radio(
        "Statistical Approach",
        ['Bayesian', 'Frequentist (Classical)'],
        help="Bayesian: Prob to Beat Control. Frequentist: P-value (Z-test/T-test)."
    )

    st.title(f"Land Analysis Dashboard ({stat_method.split()[0]}) 🔮")

    data = load_data(DEFAULT_CSV_PATH)
    if data is None:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file: data = load_data(uploaded_file)
        else: return

    if data is None or data.empty:
        st.error("No data available.")
        return

    # --- Filters ---
    with st.expander("🔍 Filter Data", expanded=True):
        c1, c2 = st.columns(2)

        # Country Filter
        all_c = sorted(data['country'].dropna().unique())
        sel_all_c = c1.checkbox("Select All Countries", value=True)
        if sel_all_c:
            countries = all_c
            c1.multiselect("Countries", all_c, default=all_c, disabled=True)
        else:
            countries = c1.multiselect("Countries", all_c, default=all_c)

        # Platform Filter
        all_p = sorted(data['platform_name'].dropna().unique())
        sel_all_p = c2.checkbox("Select All Platforms", value=True)
        if sel_all_p:
            platforms = all_p
            c2.multiselect("Platforms", all_p, default=all_p, disabled=True)
        else:
            platforms = c2.multiselect("Platforms", all_p, default=all_p)

        # Fallback to all if empty selection to prevent breakage
        if not countries: countries = all_c
        if not platforms: platforms = all_p

    df_filtered = data[(data['country'].isin(countries)) & (data['platform_name'].isin(platforms))]
    st.markdown(f"**Data Points**: {len(df_filtered)} rows | **Traffic Share**: {len(df_filtered)/len(data):.1%}")

    if df_filtered.empty:
        st.warning("No data matches filters.")
        return

    # --- Exec ---
    st.header("Executive Summary")
    stats_data = run_statistics(df_filtered, method=stat_method)
    st.info(generate_automated_summary(df_filtered, stats_data, stat_method, "selected segment"))

    # --- Metrics Table ---
    st.subheader("Detailed Performance")
    variants = sorted(df_filtered['landingId'].unique())
    control = 'mm-nsp-v1'
    metrics_list = ['Landing -> Onboarding', 'Landing -> Registration', 'Registration -> Payer', 'ARPU', 'ARPPU']

    m_c = calculate_metrics(df_filtered[df_filtered['landingId'] == control])
    c_c = get_conversion_rates(m_c)
    c_c['ARPU'], c_c['ARPPU'] = m_c['ARPU'], m_c['ARPPU']

    for v in variants:
        if v == control: continue
        st.markdown(f"### {v} vs {control}")
        m_v = calculate_metrics(df_filtered[df_filtered['landingId'] == v])
        c_v = get_conversion_rates(m_v)
        c_v['ARPU'], c_v['ARPPU'] = m_v['ARPU'], m_v['ARPPU']

        res = stats_data.get(v, {})
        rows = []
        for metric in metrics_list:
            val, uplift = res.get(metric, (0.5, 0.0) if stat_method.startswith('Bayesian') else (1.0, 0.0))
            is_mon = metric in ['ARPU', 'ARPPU']

            # Helper to get context counts
            def get_ctx(m_dict, met_name):
                if met_name == 'Landing -> Onboarding':
                    return int(m_dict['Onboarding Users']), int(m_dict['Visitors'])
                elif met_name == 'Landing -> Registration':
                    return int(m_dict['Registered Users']), int(m_dict['Visitors'])
                elif met_name == 'Registration -> Payer':
                    return int(m_dict['Payers']), int(m_dict['Registered Users'])
                elif met_name == 'ARPU':
                    return m_dict['Total Revenue'], int(m_dict['Visitors'])
                elif met_name == 'ARPPU':
                    return m_dict['Total Revenue'], int(m_dict['Payers'])
                return 0, 0

            # Control Context
            num_c, den_c = get_ctx(m_c, metric)
            if is_mon:
                fmt_c = f"${c_c.get(metric,0):.2f} <small>(${num_c:,.0f}/{den_c:,})</small>"
            else:
                fmt_c = f"{c_c.get(metric,0):.2f}% <small>({num_c:,}/{den_c:,})</small>"

            # Variant Context
            num_v, den_v = get_ctx(m_v, metric)
            if is_mon:
                fmt_v = f"${c_v.get(metric,0):.2f} <small>(${num_v:,.0f}/{den_v:,})</small>"
            else:
                fmt_v = f"{c_v.get(metric,0):.2f}% <small>({num_v:,}/{den_v:,})</small>"

            upl_d = f"${uplift:+.2f}" if is_mon else f"{uplift:+.2f}%"

            if stat_method.startswith('Bayesian'):
                stat_s = f"{val:.1%} {'🟢' if val>0.95 else '🔴' if val<0.05 else '⚪'}"
                stat_col_name = "Prob (V > C)"
            else:
                stat_s = f"{val:.4f} {'🟢' if val<0.05 else '⚪'}"
                stat_col_name = "P-value"

            rows.append({
                "Metric": metric, "Control": fmt_c, "Variant": fmt_v,
                stat_col_name: stat_s, "Diff / Uplift": upl_d
            })

        df_display = pd.DataFrame(rows)
        st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- Visual Analysis ---
    st.subheader("Visual Analysis")
    t1, t2, t3, t4 = st.tabs(["Funnel", "Revenue Dist", "Deep Dive: ARPPU Impact", "Audience Structure"])

    with t1:
        # Funnel with Tooltips
        funnel_data = []
        metric_names = ['Visitors', 'Onboarding Users', 'Registered Users', 'Payers']
        for v in variants + ([control] if control in df_filtered['landingId'].values else []):
            if v not in df_filtered['landingId'].values: continue
            m = calculate_metrics(df_filtered[df_filtered['landingId'] == v])
            for stage in metric_names:
                funnel_data.append({'Variant': v, 'Stage': stage, 'Count': m[stage]})

        df_funnel = pd.DataFrame(funnel_data).drop_duplicates()
        if not df_funnel.empty:
            fig_funnel = px.bar(df_funnel, x='Stage', y='Count', color='Variant', barmode='group')
            # Custom Hover Data: Already showing Count by default in Plotly, but can enforce.
            # Convert count to string for nice formatting if needed.
            fig_funnel.update_traces(hovertemplate='<b>%{x}</b><br>Variant: %{fullData.name}<br>Count: %{y}<extra></extra>')
            st.plotly_chart(fig_funnel, use_container_width=True)

    with t2:
         fig_box = px.box(df_filtered, x='landingId', y='amount', title="Revenue per User Distribution")
         st.plotly_chart(fig_box, use_container_width=True)

    # --- Deep Dive: ARPPU Drivers ---
    with t3:
        st.markdown("##### ARPPU Drivers & Package Impact Analysis")

        pkg_view = st.radio("View Packages By:", ['Product Clusters', 'Raw Package IDs'], horizontal=True)
        col_group = 'product_cluster' if pkg_view == 'Product Clusters' else 'id'

        if col_group in df_filtered.columns:
            # 1. Calc Contribution
            payers_per_variant = df_filtered[df_filtered['fo_at'].notnull()].groupby('landingId')['user_id'].nunique().to_dict()
            df_rev_pkg = df_filtered[df_filtered['fo_at'].notnull()].groupby(['landingId', col_group])['amount'].sum().reset_index()

            df_rev_pkg['total_payers'] = df_rev_pkg['landingId'].map(payers_per_variant)
            df_rev_pkg['contribution'] = df_rev_pkg['amount'] / df_rev_pkg['total_payers']

            # Viz 1: Stacked Bar
            fig_stack = px.bar(df_rev_pkg, x='landingId', y='contribution', color=col_group,
                               title=f"ARPPU Composition by {pkg_view} ($)",
                               labels={'contribution': 'Contribution to ARPPU ($)', col_group: pkg_view})
            fig_stack.update_layout(yaxis_tickformat='$.2f')
            fig_stack.update_traces(hovertemplate='<b>%{data.name}</b><br>Contrib: $%{y:.2f}<extra></extra>')
            st.plotly_chart(fig_stack, use_container_width=True)

            # Viz 2: Impact Analysis
            df_pivot = df_rev_pkg.pivot(index=col_group, columns='landingId', values='contribution').fillna(0)

            if control in df_pivot.columns:
                for v in variants:
                    if v == control: continue
                    if v not in df_pivot.columns: continue

                    st.markdown(f"**Impact Dictionary: {v} vs {control}**")
                    diff_col = f'Impact_{v}'
                    df_pivot[diff_col] = df_pivot[v] - df_pivot[control]
                    df_impact = df_pivot[[diff_col]].sort_values(diff_col, ascending=True).reset_index()

                    fig_imp = px.bar(df_impact, y=col_group, x=diff_col, orientation='h',
                                     title=f"{pkg_view} Impact on ARPPU: {v} vs Control",
                                     color=diff_col, color_continuous_scale='RdBu')
                    fig_imp.update_layout(xaxis_tickformat='$.2f')
                    fig_imp.update_traces(hovertemplate='<b>%{y}</b><br>Impact: $%{x:.2f}<extra></extra>')
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # Insight
                    best_pkg = df_impact.iloc[-1]
                    worst_pkg = df_impact.iloc[0]
                    insight_text = []
                    if best_pkg[diff_col] > 0.1:
                        insight_text.append(f"Growth is primarily driven by **{best_pkg[col_group]}** (+${best_pkg[diff_col]:.2f}).")
                    if worst_pkg[diff_col] < -0.1:
                        insight_text.append(f"Offset by a drop in **{worst_pkg[col_group]}** (${worst_pkg[diff_col]:.2f}).")
                    if insight_text:
                        st.info(" ".join(insight_text))

        else:
            st.warning(f"Column '{col_group}' unavailable.")

    with t4:
        s1, s2, s3 = st.tabs(["Country", "Platform", "Model"])
        with s1:
            df_c = df_filtered.groupby(['landingId', 'country']).size().reset_index(name='count')
            df_c['share'] = df_c.groupby('landingId')['count'].transform(lambda x: x/x.sum())
            fig = px.bar(df_c, x='landingId', y='share', color='country', barmode='stack')
            fig.layout.yaxis.tickformat = '.0%'
            st.plotly_chart(fig, use_container_width=True)
        with s2:
            df_p = df_filtered.groupby(['landingId', 'platform_name']).size().reset_index(name='count')
            df_p['share'] = df_p.groupby('landingId')['count'].transform(lambda x: x/x.sum())
            fig = px.bar(df_p, x='landingId', y='share', color='platform_name', barmode='stack')
            fig.layout.yaxis.tickformat = '.0%'
            st.plotly_chart(fig, use_container_width=True)
        with s3:
            top10 = df_filtered['platform_model'].value_counts().nlargest(10).index
            df_m = df_filtered[df_filtered['platform_model'].isin(top10)]
            df_ma = df_m.groupby(['landingId', 'platform_model']).size().reset_index(name='count')
            df_ma['share'] = df_ma.groupby('landingId')['count'].transform(lambda x: x/x.sum())
            fig = px.bar(df_ma, x='landingId', y='share', color='platform_model', barmode='stack')
            fig.layout.yaxis.tickformat = '.0%'
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_dashboard()
