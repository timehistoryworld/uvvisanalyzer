import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="UV/Vis Analyzer", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; }
    .info-box {
        background: #f0f4ff; border-left: 4px solid #4a6fa5;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0; font-size: 0.9em;
    }
    .result-box {
        background: #f0fff4; border-left: 4px solid #38a169;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0; font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 UV/Vis Absorption Analyzer")
st.caption("Peak analysis · Gaussian fitting · Titration · Isosbestic · Hill equation · Excel export")

QUAL_COLORS = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(file):
    for sep in [',', '\t', ' ', ';']:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=sep, comment='#', engine='python')
            df = df.dropna(axis=1, how='all').dropna(how='all')
            num = df.select_dtypes(include=[np.number]).columns
            if len(num) >= 2:
                return df[num].rename(columns={num[0]: 'wavelength', num[1]: 'absorbance'})
        except Exception:
            pass
    return None

def sort_spectrum(df):
    x = df['wavelength'].values.astype(float)
    y = df['absorbance'].values.astype(float)
    idx = np.argsort(x)
    return x[idx], y[idx]

def smooth(y, window=11, poly=3):
    w = window if window % 2 == 1 else window + 1
    if len(y) < w:
        return y
    return savgol_filter(y, window_length=w, polyorder=poly)

def derivative(x, y, order=1, sg_window=11):
    ys = smooth(y, sg_window)
    dy = np.gradient(ys, x)
    if order == 2:
        dy = np.gradient(dy, x)
    return dy

def gaussian(x, amp, center, sigma):
    return amp * np.exp(-((x - center)**2) / (2 * sigma**2))

def multi_gaussian(x, *params):
    n = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(n):
        result += gaussian(x, params[3*i], params[3*i+1], abs(params[3*i+2]))
    return result

def auto_initial_guess(x, y, n_bands):
    ys = smooth(y, 15)
    peaks, _ = find_peaks(ys, height=np.max(ys)*0.05, distance=max(1, len(x)//20))
    pos = list(x[peaks]) if len(peaks) > 0 else [x[np.argmax(ys)]]
    hts = list(ys[peaks]) if len(peaks) > 0 else [np.max(ys)]
    span = x[-1] - x[0]
    while len(pos) < n_bands:
        step = span / (n_bands + 1)
        extras = [x[0] + step*(i+1) for i in range(n_bands - len(pos))]
        pos += extras
        hts += [np.max(ys)*0.3] * len(extras)
    pos = np.array(pos[:n_bands])
    hts = np.array(hts[:n_bands])
    sigma0 = span / (n_bands * 4)
    p0 = []
    for a, c in zip(hts, pos):
        p0 += [float(a), float(c), float(sigma0)]
    return p0

def science_note(nm, kind):
    regions = {
        (200, 250): "π→π* transition (aromatic / conjugated system)",
        (250, 300): "π→π* / n→π* overlap region",
        (300, 400): "n→π* transition (carbonyl / heteroatom lone pair)",
        (400, 500): "Visible — charge-transfer or extended conjugation",
        (500, 700): "d→d transition or extended chromophore",
        (700, 900): "NIR-edge — radical, metal complex, or ICT state",
    }
    if kind == 'max':
        for (lo, hi), txt in regions.items():
            if lo <= nm < hi:
                return txt
        return "Strong absorption feature"
    if kind == 'd1_zero':
        return "Inflection point — shoulder / hidden sub-band boundary"
    if kind == 'd2_min':
        return "Hidden peak or vibronic shoulder (2nd-deriv minimum)"
    return ""

def df_to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()

def hill_equation(x, A_min, A_max, Kd, n):
    return A_min + (A_max - A_min) * (x**n) / (Kd**n + x**n)

def fit_hill(conc, response):
    A0   = float(np.min(response))
    Ainf = float(np.max(response))
    Kd0  = float(np.median(conc))
    p0   = [A0, Ainf, Kd0, 1.0]
    span = abs(Ainf - A0)
    bounds = (
        [min(A0, Ainf) - span, min(A0, Ainf) - span, 1e-12, 0.1],
        [max(A0, Ainf) + span, max(A0, Ainf) + span, np.inf, 10.0],
    )
    popt, pcov = curve_fit(hill_equation, conc, response, p0=p0,
                           bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def find_isosbestic(spectra_list, tol_abs=0.02, tol_nm=2.0):
    x_min = max(s[0][0]  for s in spectra_list)
    x_max = min(s[0][-1] for s in spectra_list)
    x_common = np.linspace(x_min, x_max, 1000)
    interp_mat = np.array([np.interp(x_common, xs, ys) for xs, ys in spectra_list])
    std_wl = np.std(interp_mat, axis=0)
    candidates, _ = find_peaks(-std_wl, height=-tol_abs)
    iso_pts = []
    last_nm = -999.0
    for c in candidates:
        if std_wl[c] < tol_abs and (x_common[c] - last_nm) > tol_nm:
            iso_pts.append({
                'wavelength_nm':   round(float(x_common[c]), 2),
                'mean_absorbance': round(float(np.mean(interp_mat[:, c])), 4),
                'std_absorbance':  round(float(std_wl[c]), 5),
            })
            last_nm = x_common[c]
    return iso_pts, x_common, interp_mat, std_wl

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Peak & Derivatives",
    "🔧 Gaussian Fitting",
    "🧪 Titration & Hill",
    "🎯 Isosbestic Points",
    "📊 Multi-file Compare",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Peak & Derivatives
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Peak Detection & Derivative Analysis")
    uploaded = st.file_uploader("Upload CSV (wavelength, absorbance)", type="csv", key="t1")

    if uploaded:
        df = load_csv(uploaded)
        if df is None:
            st.error("Could not parse CSV.")
        else:
            x, y = sort_spectrum(df)
            c1, c2, c3 = st.columns(3)
            sg_w  = c1.slider("SG smoothing window", 5, 51, 11, 2, key="sg1")
            p_thr = c2.slider("Peak threshold (% max)", 1, 50, 5, key="pt1")
            p_dst = c3.slider("Min peak distance (nm)", 1, 100, 20, key="pd1")

            ys = smooth(y, sg_w)
            d1 = derivative(x, y, 1, sg_w)
            d2 = derivative(x, y, 2, sg_w)

            dx   = abs(x[1]-x[0]) if len(x) > 1 else 1.0
            dist = max(1, int(p_dst / dx))
            peaks, _ = find_peaks(ys, height=np.max(ys)*p_thr/100, distance=dist)
            d1z = [i for i in range(1, len(d1)-1) if d1[i-1]*d1[i+1] < 0]
            d2m, _ = find_peaks(-d2, height=np.max(np.abs(d2))*0.05, distance=dist)

            fig = make_subplots(3, 1, shared_xaxes=True,
                subplot_titles=("Absorption", "1st Derivative", "2nd Derivative"),
                vertical_spacing=0.07, row_heights=[0.5, 0.25, 0.25])

            fig.add_trace(go.Scatter(x=x, y=ys, name="Abs",
                                     line=dict(color='royalblue', width=2)), 1, 1)
            for p in peaks:
                fig.add_trace(go.Scatter(x=[x[p]], y=[ys[p]], mode='markers+text',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    text=[f"λ={x[p]:.1f}"], textposition='top center',
                    name=f"λmax {x[p]:.0f}",
                    hovertext=science_note(x[p], 'max'), hoverinfo='text+x+y'), 1, 1)

            fig.add_trace(go.Scatter(x=x, y=d1, name="d1",
                                     line=dict(color='darkorange', width=1.5)), 2, 1)
            for i in d1z:
                fig.add_trace(go.Scatter(x=[x[i]], y=[d1[i]], mode='markers',
                    marker=dict(color='orange', size=7), name=f"d1z {x[i]:.0f}",
                    hovertext=science_note(x[i], 'd1_zero'), hoverinfo='text+x+y'), 2, 1)
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)

            fig.add_trace(go.Scatter(x=x, y=d2, name="d2",
                                     line=dict(color='green', width=1.5)), 3, 1)
            for p in d2m:
                fig.add_trace(go.Scatter(x=[x[p]], y=[d2[p]], mode='markers',
                    marker=dict(color='darkgreen', size=7, symbol='square'),
                    name=f"d2m {x[p]:.0f}",
                    hovertext=science_note(x[p], 'd2_min'), hoverinfo='text+x+y'), 3, 1)
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)

            fig.update_layout(height=700, showlegend=False, hovermode='x unified',
                xaxis3_title="Wavelength (nm)", yaxis_title="Absorbance",
                yaxis2_title="dA/dλ", yaxis3_title="d²A/dλ²")
            st.plotly_chart(fig, use_container_width=True)

            rows = []
            for p in peaks:
                rows.append({'Type': '🔴 λmax', 'Wavelength (nm)': f"{x[p]:.2f}",
                             'Value': f"{ys[p]:.4f}", 'Interpretation': science_note(x[p], 'max')})
            for i in d1z:
                rows.append({'Type': '🟠 1st deriv zero', 'Wavelength (nm)': f"{x[i]:.2f}",
                             'Value': f"{d1[i]:.4f}", 'Interpretation': science_note(x[i], 'd1_zero')})
            for p in d2m:
                rows.append({'Type': '🟢 2nd deriv min', 'Wavelength (nm)': f"{x[p]:.2f}",
                             'Value': f"{d2[p]:.4f}", 'Interpretation': science_note(x[p], 'd2_min')})

            if rows:
                summary_df = pd.DataFrame(rows)
                st.subheader("Feature Summary")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                raw_df = pd.DataFrame({'wavelength_nm': x, 'absorbance': y,
                                       'smoothed': ys, 'd1': d1, 'd2': d2})
                xl = df_to_excel_bytes({'Spectra & Derivatives': raw_df,
                                        'Feature Summary': summary_df})
                st.download_button("⬇ Download Excel", xl, file_name="peak_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Upload a CSV. Expected: two columns — wavelength (nm) and absorbance.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Gaussian Fitting
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Multi-Gaussian Band Fitting")
    uploaded2 = st.file_uploader("Upload CSV", type="csv", key="t2")

    if uploaded2:
        df2 = load_csv(uploaded2)
        if df2 is None:
            st.error("Could not parse CSV.")
        else:
            x2, y2 = sort_spectrum(df2)
            c1, c2 = st.columns(2)
            n_bands = c1.slider("Number of Gaussian bands", 1, 8, 2, key="nb")
            sg_w2   = c2.slider("Smoothing window", 5, 51, 11, 2, key="sg2")

            p0_auto = auto_initial_guess(x2, y2, n_bands)
            st.markdown("**Adjust initial guesses per band:**")
            hc = st.columns([2, 2, 2])
            hc[0].markdown("**Amplitude**")
            hc[1].markdown("**Center (nm)**")
            hc[2].markdown("**Width σ (nm)**")
            params_ui = []
            for i in range(n_bands):
                cc = st.columns([2, 2, 2])
                amp = cc[0].number_input(f"B{i+1} amp", value=round(float(p0_auto[3*i]),   4), step=0.01, key=f"amp{i}")
                cen = cc[1].number_input(f"B{i+1} cen", value=round(float(p0_auto[3*i+1]), 1), step=1.0,  key=f"cen{i}")
                sig = cc[2].number_input(f"B{i+1} σ",   value=round(abs(float(p0_auto[3*i+2])), 1), step=1.0, key=f"sig{i}")
                params_ui += [amp, cen, sig]

            if st.button("▶ Run Gaussian Fit", type="primary", key="gauss_btn"):
                try:
                    blo = [0,     x2[0],  0.5] * n_bands
                    bhi = [1e4,  x2[-1], (x2[-1]-x2[0])/2] * n_bands
                    popt, pcov = curve_fit(multi_gaussian, x2, y2, p0=params_ui,
                                           bounds=(blo, bhi), maxfev=20000)
                    perr = np.sqrt(np.diag(pcov))
                    yfit  = multi_gaussian(x2, *popt)
                    resid = y2 - yfit
                    r2    = 1 - np.sum(resid**2) / np.sum((y2 - np.mean(y2))**2)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=x2, y=y2, name="Data",
                                              line=dict(color='black', width=2)))
                    fig2.add_trace(go.Scatter(x=x2, y=yfit,
                                              name=f"Total fit (R²={r2:.4f})",
                                              line=dict(color='red', width=2, dash='dash')))
                    band_rows = []
                    fit_df    = pd.DataFrame({'wavelength_nm': x2, 'absorbance': y2,
                                             'total_fit': yfit, 'residuals': resid})
                    for i in range(n_bands):
                        ai, ci, si = popt[3*i], popt[3*i+1], abs(popt[3*i+2])
                        yb   = gaussian(x2, ai, ci, si)
                        fwhm = 2.355 * si
                        area = ai * si * np.sqrt(2 * np.pi)
                        col  = QUAL_COLORS[i % len(QUAL_COLORS)]
                        fig2.add_trace(go.Scatter(x=x2, y=yb,
                            name=f"Band {i+1}: {ci:.1f} nm",
                            line=dict(color=col, width=1.5),
                            fill='tozeroy'))
                        band_rows.append({
                            'Band': i+1,
                            'Center (nm)': f"{ci:.2f} ± {perr[3*i+1]:.2f}",
                            'Amplitude':   f"{ai:.4f} ± {perr[3*i]:.4f}",
                            'σ (nm)':      f"{si:.2f}",
                            'FWHM (nm)':   f"{fwhm:.2f}",
                            'Area (a.u.)': f"{area:.4f}",
                            'Interpretation': science_note(ci, 'max'),
                        })
                        fit_df[f'band_{i+1}'] = yb

                    fig2.add_trace(go.Scatter(x=x2, y=resid, name="Residuals",
                                              line=dict(color='gray', width=1), yaxis='y2'))
                    fig2.update_layout(height=560, hovermode='x unified',
                        xaxis_title="Wavelength (nm)", yaxis_title="Absorbance",
                        yaxis2=dict(title="Residuals", overlaying='y', side='right', showgrid=False),
                        legend=dict(x=1.05, y=1))
                    st.plotly_chart(fig2, use_container_width=True)

                    band_df = pd.DataFrame(band_rows)
                    st.subheader(f"Fit Results  (R² = {r2:.5f})")
                    st.dataframe(band_df, use_container_width=True, hide_index=True)

                    xl2 = df_to_excel_bytes({'Fit Curves': fit_df, 'Band Parameters': band_df})
                    st.download_button("⬇ Download Excel", xl2, file_name="gaussian_fit.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"Fitting failed: {e}. Try adjusting initial parameters.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Titration & Hill Equation
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Titration Analysis + Hill Equation Fitting")
    st.markdown("""
<div class="info-box">
Upload one CSV per titration point. Dilution is corrected automatically.<br>
<b>Hill equation</b>: A = A<sub>min</sub> + (A<sub>max</sub> − A<sub>min</sub>) · [L]<sup>n</sup> / (K<sub>d</sub><sup>n</sup> + [L]<sup>n</sup>)<br>
n = 1: simple 1:1 binding &nbsp;|&nbsp; n &gt; 1: positive cooperativity &nbsp;|&nbsp; n &lt; 1: negative cooperativity
</div>""", unsafe_allow_html=True)

    tit_files = st.file_uploader("Upload titration CSVs (one per point)", type="csv",
                                  accept_multiple_files=True, key="t3")

    if tit_files:
        tit_files = sorted(tit_files, key=lambda f: f.name)
        n_tit = len(tit_files)
        st.success(f"{n_tit} files: {', '.join(f.name for f in tit_files)}")

        st.subheader("Experimental Setup")
        c1, c2, c3 = st.columns(3)
        c_titrant  = c1.number_input("Titrant conc. (mM)",        value=1.0,  step=0.1,  min_value=0.0)
        c_analyte  = c2.number_input("Analyte conc. (mM)",         value=0.1,  step=0.01, min_value=0.0)
        V0         = c3.number_input("Initial analyte vol. (mL)",  value=2.0,  step=0.1,  min_value=0.0)
        eps0       = c1.number_input("ε₀ at λref (M⁻¹cm⁻¹, 0=skip)", value=0.0, step=100.0)
        path_len   = c2.number_input("Path length (cm)",           value=1.0,  step=0.1,  min_value=0.1)
        do_hill    = c3.checkbox("Fit Hill equation", value=True)

        st.markdown("**Volume of titrant added at each point (mL):**")
        vol_cols = st.columns(min(n_tit, 6))
        vols = []
        for i, f in enumerate(tit_files):
            v = vol_cols[i % 6].number_input(f.name[:10], value=float(i)*0.2,
                                              step=0.05, min_value=0.0, key=f"vol{i}")
            vols.append(v)

        if st.button("▶ Compute Titration", type="primary", key="tit_btn"):
            spectra_tit = []
            for f, v in zip(tit_files, vols):
                df_t = load_csv(f)
                if df_t is None:
                    continue
                xt, yt = sort_spectrum(df_t)
                V_total = V0 + v
                dil     = V0 / V_total if V_total > 0 else 1.0
                spectra_tit.append((xt, yt / dil, v))

            if not spectra_tit:
                st.error("No valid spectra.")
            else:
                ref_x0 = spectra_tit[0][0]
                ref_nm_default = float(ref_x0[np.argmax(spectra_tit[0][1])])
                ref_nm_tit = st.number_input("Reference λ for binding curve (nm)",
                                              value=ref_nm_default, step=1.0, key="ref_nm_tit")

                grad = [f"hsl({int(360*i/max(len(spectra_tit)-1,1))},70%,50%)"
                        for i in range(len(spectra_tit))]

                A_vals, conc_free, equiv_vals = [], [], []
                for xt, yt_c, v in spectra_tit:
                    a = float(np.interp(ref_nm_tit, xt, yt_c))
                    if eps0 > 0:
                        a = a / (eps0 * path_len * c_analyte * 1e-3)
                    A_vals.append(a)
                    V_total = V0 + v
                    conc_free.append((c_titrant * v / V_total) if V_total > 0 else 0.0)
                    equiv_vals.append((c_titrant * v) / (c_analyte * V0)
                                      if (c_analyte * V0) > 0 else v)

                fig3 = make_subplots(1, 2,
                    subplot_titles=("Dilution-Corrected Spectra", "Binding Curve"))
                for idx, (xt, yt_c, v) in enumerate(spectra_tit):
                    fig3.add_trace(go.Scatter(x=xt, y=yt_c, name=f"V={v:.2f} mL",
                                              line=dict(color=grad[idx], width=1.5)), 1, 1)
                fig3.add_trace(go.Scatter(x=equiv_vals, y=A_vals, mode='markers',
                    name="Data", marker=dict(size=9, color='steelblue')), 1, 2)

                hill_result = {}
                if do_hill and len(A_vals) >= 4:
                    try:
                        ca = np.array(conc_free) + 1e-12
                        Aa = np.array(A_vals)
                        popt_h, perr_h = fit_hill(ca, Aa)
                        A_min_h, A_max_h, Kd_h, n_h = popt_h
                        Kd_err, n_err = perr_h[2], perr_h[3]

                        x_fit    = np.linspace(ca.min(), ca.max(), 400)
                        y_fit_h  = hill_equation(x_fit, *popt_h)
                        eq_fit   = x_fit / (c_titrant / 1) if c_titrant > 0 else x_fit

                        fig3.add_trace(go.Scatter(x=eq_fit, y=y_fit_h,
                            name=f"Hill fit  Kd={Kd_h:.4f} mM  n={n_h:.2f}",
                            line=dict(color='crimson', width=2.5, dash='dash')), 1, 2)

                        cooperativity = ('Positive cooperativity' if n_h > 1.1
                                         else 'Negative cooperativity' if n_h < 0.9
                                         else 'Non-cooperative (1:1)')
                        hill_result = {
                            'A_min': round(A_min_h, 5),
                            'A_max': round(A_max_h, 5),
                            'Kd (mM)': f"{Kd_h:.4f} ± {Kd_err:.4f}",
                            'Hill n':  f"{n_h:.3f} ± {n_err:.3f}",
                            'Interpretation': cooperativity,
                        }
                    except Exception as e:
                        st.warning(f"Hill fitting failed: {e}")

                ylabel2 = "A / (ε₀·l·c₀)" if eps0 > 0 else f"A_corr @ {ref_nm_tit:.0f} nm"
                fig3.update_layout(height=480, hovermode='x unified',
                    xaxis_title="Wavelength (nm)", yaxis_title="Corrected Absorbance",
                    xaxis2_title="Equivalents of titrant", yaxis2_title=ylabel2,
                    legend=dict(x=1.02, y=1))
                st.plotly_chart(fig3, use_container_width=True)

                if hill_result:
                    st.markdown(f"""
<div class="result-box">
<b>Hill Fit Results</b><br>
K<sub>d</sub> = {hill_result["Kd (mM)"]} mM &nbsp;|&nbsp;
n = {hill_result["Hill n"]} &nbsp;|&nbsp;
{hill_result["Interpretation"]}
</div>""", unsafe_allow_html=True)

                # Excel export
                ref_x_grid = spectra_tit[0][0]
                spec_sheet = {'wavelength_nm': ref_x_grid}
                for xt, yt_c, v in spectra_tit:
                    spec_sheet[f'A_corr_V={v:.2f}mL'] = np.interp(ref_x_grid, xt, yt_c)
                binding_sheet = pd.DataFrame({
                    'equiv': equiv_vals, 'free_conc_mM': conc_free, 'response': A_vals})
                sheets_out = {'Corrected Spectra': pd.DataFrame(spec_sheet),
                              'Binding Curve':     binding_sheet}
                if hill_result:
                    sheets_out['Hill Fit Params'] = pd.DataFrame([hill_result])
                xl3 = df_to_excel_bytes(sheets_out)
                st.download_button("⬇ Download Excel", xl3, file_name="titration_hill.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Isosbestic Points
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Isosbestic Point Detection")
    st.markdown("""
<div class="info-box">
<b>Isosbestic point (등흡광점)</b>: a wavelength at which all spectra in a series share the
same absorbance value, indicating a clean two-state equilibrium with no intermediate species.<br><br>
Algorithm: finds wavelengths where the standard deviation of absorbance across all spectra
drops below the set tolerance. Narrow, deep minima → sharp isosbestic points.
</div>""", unsafe_allow_html=True)

    iso_files = st.file_uploader("Upload spectral series CSVs (≥ 2 files)",
                                  type="csv", accept_multiple_files=True, key="t4")

    if iso_files and len(iso_files) >= 2:
        iso_files = sorted(iso_files, key=lambda f: f.name)
        spectra_iso, names_iso = [], []
        for f in iso_files:
            df_i = load_csv(f)
            if df_i is not None:
                xi, yi = sort_spectrum(df_i)
                spectra_iso.append((xi, yi))
                names_iso.append(f.name)

        if len(spectra_iso) < 2:
            st.error("Need at least 2 valid spectra.")
        else:
            st.success(f"{len(spectra_iso)} spectra loaded.")
            c1, c2 = st.columns(2)
            tol_abs = c1.slider("Absorbance std tolerance", 0.001, 0.10, 0.02, 0.001,
                                 format="%.3f", key="iso_tol",
                                 help="Lower = stricter criterion for isosbestic points")
            tol_nm  = c2.slider("Min spacing between iso points (nm)", 0.5, 20.0, 2.0, 0.5,
                                 key="iso_nm")

            iso_pts, x_common, interp_mat, std_wl = find_isosbestic(
                spectra_iso, tol_abs=tol_abs, tol_nm=tol_nm)

            # ── Main spectra plot ──
            fig4a = go.Figure()
            grad_iso = [f"hsl({int(300*i/max(len(spectra_iso)-1,1))},65%,50%)"
                        for i in range(len(spectra_iso))]
            for idx, (xi, yi) in enumerate(spectra_iso):
                fig4a.add_trace(go.Scatter(x=xi, y=yi, name=names_iso[idx],
                                            line=dict(color=grad_iso[idx], width=1.8)))
            for pt in iso_pts:
                fig4a.add_vline(x=pt['wavelength_nm'], line_dash='dot',
                                 line_color='black', line_width=1.5,
                                 annotation_text=f"{pt['wavelength_nm']} nm",
                                 annotation_position="top right",
                                 annotation_font_size=11)
            fig4a.update_layout(height=400, hovermode='x unified',
                xaxis_title="Wavelength (nm)", yaxis_title="Absorbance",
                legend=dict(x=1.02, y=1),
                title="Spectral Series — Isosbestic Points Marked")
            st.plotly_chart(fig4a, use_container_width=True)

            # ── Std curve ──
            fig4b = go.Figure()
            fig4b.add_trace(go.Scatter(x=x_common, y=std_wl,
                name="Std(A) across spectra",
                line=dict(color='steelblue', width=1.5),
                fill='tozeroy', fillcolor='rgba(70,130,180,0.1)'))
            fig4b.add_hline(y=tol_abs, line_dash='dash', line_color='red',
                             annotation_text=f"tolerance = {tol_abs:.3f}",
                             annotation_position="right")
            for pt in iso_pts:
                fig4b.add_vline(x=pt['wavelength_nm'], line_dash='dot',
                                 line_color='black', line_width=1)
            fig4b.update_layout(height=220,
                xaxis_title="Wavelength (nm)",
                yaxis_title="Std of Absorbance",
                title="Absorbance Std — Minima = Isosbestic Candidates")
            st.plotly_chart(fig4b, use_container_width=True)

            # ── Results table ──
            if iso_pts:
                iso_df = pd.DataFrame(iso_pts)
                st.subheader(f"✅ {len(iso_pts)} Isosbestic Point(s) Detected")
                st.dataframe(iso_df, use_container_width=True, hide_index=True)
                st.markdown("""
<div class="result-box">
<b>Interpretation guide</b><br>
• <b>Sharp, well-defined isosbestic points</b> → clean two-state equilibrium (A ⇌ B), 
no intermediate accumulation.<br>
• <b>Multiple isosbestic points</b> → confirms two-state; each point is a wavelength where 
ε(A) = ε(B).<br>
• <b>Broad or drifting pseudo-isosbestic points</b> → possible third species, 
spectral overlap, or inner filter effect.<br>
• <b>No isosbestic point</b> → multi-state process or baseline drift.
</div>""", unsafe_allow_html=True)
            else:
                st.warning("No isosbestic points found. Try increasing the tolerance "
                           "or checking your spectral overlap region.")

            # Excel export
            interp_df = pd.DataFrame(interp_mat.T, columns=names_iso)
            interp_df.insert(0, 'wavelength_nm', x_common)
            std_df = pd.DataFrame({'wavelength_nm': x_common, 'std_absorbance': std_wl})
            iso_export = pd.DataFrame(iso_pts) if iso_pts else pd.DataFrame({'result': ['None found']})
            xl4 = df_to_excel_bytes({'Interpolated Spectra': interp_df,
                                     'Std per Wavelength':   std_df,
                                     'Isosbestic Points':    iso_export})
            st.download_button("⬇ Download Excel", xl4, file_name="isosbestic_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif iso_files and len(iso_files) < 2:
        st.warning("Upload at least 2 spectra.")
    else:
        st.info("Upload 2 or more CSV spectra from a titration or kinetic series.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Multi-file Compare
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Multi-file Comparison & Normalization")
    multi_files = st.file_uploader("Upload CSVs", type="csv",
                                    accept_multiple_files=True, key="t5")

    if multi_files:
        datasets = {}
        for f in multi_files:
            df_m = load_csv(f)
            if df_m is not None:
                xm, ym = sort_spectrum(df_m)
                datasets[f.name] = (xm, ym)

        if not datasets:
            st.error("No valid CSVs loaded.")
        else:
            st.success(f"{len(datasets)} spectra loaded.")
            norm_mode = st.radio("Normalization mode",
                ["None (raw)", "Max peak", "Specific wavelength", "Area (trapz)"],
                horizontal=True)

            norm_nm = None
            if norm_mode == "Specific wavelength":
                all_x = np.concatenate([v[0] for v in datasets.values()])
                norm_nm = st.slider("Normalize at (nm)", float(all_x.min()),
                                     float(all_x.max()), float(np.median(all_x)),
                                     step=0.5, key="norm_nm")

            offset = st.slider("Vertical offset between spectra", 0.0, 2.0, 0.0, 0.05)

            fig5   = go.Figure()
            rows5  = []
            norm_data = {}

            for idx, (name, (xm, ym)) in enumerate(datasets.items()):
                if norm_mode == "None (raw)":
                    yn, nv = ym, 1.0
                elif norm_mode == "Max peak":
                    nv = float(np.max(ym))
                    yn = ym / nv if nv else ym
                elif norm_mode == "Specific wavelength" and norm_nm is not None:
                    nv = float(np.interp(norm_nm, xm, ym))
                    yn = ym / nv if nv else ym
                elif norm_mode == "Area (trapz)":
                    nv = float(np.trapz(ym, xm))
                    yn = ym / nv if nv else ym
                else:
                    yn, nv = ym, 1.0

                col = QUAL_COLORS[idx % len(QUAL_COLORS)]
                fig5.add_trace(go.Scatter(x=xm, y=yn + idx*offset,
                                           name=name, line=dict(color=col, width=2)))
                norm_data[name] = (xm, yn)

                ys_m = smooth(ym, 11)
                pk, _ = find_peaks(ys_m, height=np.max(ys_m)*0.05)
                peak_str = ", ".join(f"{xm[p]:.1f}" for p in pk) if len(pk) else f"{xm[np.argmax(ys_m)]:.1f}"
                rows5.append({'File': name, 'λmax (nm)': peak_str,
                              'Max A (raw)': f"{np.max(ym):.4f}",
                              'Norm factor': f"{nv:.5g}"})

            if norm_mode == "Specific wavelength" and norm_nm:
                fig5.add_vline(x=norm_nm, line_dash="dot", line_color="gray",
                                annotation_text=f"norm @ {norm_nm:.1f} nm")

            ylab_map = {"None (raw)": "Absorbance", "Max peak": "Norm. Absorbance",
                        "Specific wavelength": f"A / A({norm_nm:.0f} nm)" if norm_nm else "Norm.",
                        "Area (trapz)": "Norm. (area)"}
            fig5.update_layout(height=500, hovermode='x unified',
                xaxis_title="Wavelength (nm)",
                yaxis_title=ylab_map.get(norm_mode, "Absorbance"),
                legend=dict(x=1.02, y=1))
            st.plotly_chart(fig5, use_container_width=True)

            peak_df = pd.DataFrame(rows5)
            st.subheader("Peak Summary")
            st.dataframe(peak_df, use_container_width=True, hide_index=True)

            # Excel export — raw + normalized on common grid
            x_lo = max(v[0][0]  for v in datasets.values())
            x_hi = min(v[0][-1] for v in datasets.values())
            x_grid = np.linspace(x_lo, x_hi, 1000)
            raw_sheet  = {'wavelength_nm': x_grid}
            norm_sheet = {'wavelength_nm': x_grid}
            for name, (xm, ym) in datasets.items():
                raw_sheet[name]  = np.interp(x_grid, xm, ym)
            for name, (xm, yn) in norm_data.items():
                norm_sheet[name + '_norm'] = np.interp(x_grid, xm, yn)

            xl5 = df_to_excel_bytes({
                'Raw Spectra':        pd.DataFrame(raw_sheet),
                'Normalized Spectra': pd.DataFrame(norm_sheet),
                'Peak Summary':       peak_df,
            })
            st.download_button("⬇ Download Excel", xl5, file_name="multifile_compare.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Upload two or more CSV files to compare.")
