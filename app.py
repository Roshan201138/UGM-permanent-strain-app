

import io
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit, least_squares


st.set_page_config(page_title="UGM Permanent Strain Prediction and Calibration", layout="wide")


def render_header_with_logos():

  logo_files = ["logo1.png", "logo2.png", "logo3.png", "logo4.png"]

  cols_logo = st.columns(4)

  for col, logo_file in zip(cols_logo, logo_files):

    logo_path = Path(__file__).parent / logo_file

    with col:

      if logo_path.exists():

        st.image(str(logo_path), use_container_width=True)

      else:

        st.markdown(

          "<div class='logo-placeholder'>Logo</div>",

          unsafe_allow_html=True

       )


def apply_global_style():

  st.markdown("""
  <style>
    .stApp {
      font-family: "Times New Roman", Times, serif;
      color: #16212B;
      background-color: #FFFFFF;
    }

    .logo-placeholder {
      height: 70px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 2px solid #0B3C5D;
      border-radius: 10px;
      color: #0B3C5D;
      font-weight: bold;
      background-color: #F2F5F7;
      margin-bottom: 10px;
    }

    h1, h2, h3, h4 {
      font-family: "Times New Roman", Times, serif !important;
      color: #0B3C5D !important;
      font-weight: 700 !important;
    }

    p, div, span, label {
      font-family: "Times New Roman", Times, serif;
    }

    section[data-testid="stSidebar"] {
      background-color: #E8EEF3;
      border-right: 1px solid #AAB7C4;
    }

    .stButton > button {
      background-color: #0B3C5D;
      color: white;
      border-radius: 7px;
      border: 1px solid #0B3C5D;
      font-weight: 600;
    }

    .stButton > button:hover {
      background-color: #082B43;
      color: white;
      border: 1px solid #082B43;
    }

    .stDownloadButton > button {
      background-color: #1D4E89;
      color: white;
      border-radius: 7px;
      border: 1px solid #1D4E89;
      font-weight: 600;
    }

    .stDownloadButton > button:hover {
      background-color: #163A6B;
      color: white;
      border: 1px solid #163A6B;
    }

    div[data-testid="stMetricValue"] {
      color: #0B3C5D;
    }

    .main-title {
      text-align: center;
      color: #0B3C5D;
      font-size: 38px;
      font-weight: 700;
      margin-bottom: 4px;
    }

    .subtitle {
      text-align: center;
      color: #163A6B;
      font-size: 19px;
      margin-bottom: 6px;
    }

    .developer-line {
      text-align: center;
      color: #253746;
      font-size: 16px;
      margin-bottom: 20px;
    }
  </style>
  """, unsafe_allow_html=True)


apply_global_style()

render_header_with_logos()


def to_numeric_safe(s):

  return pd.to_numeric(s, errors="coerce")


def safe_array(df, col, n, default=None):

  if col and col != "None" and col in df.columns:

    return to_numeric_safe(df[col]).to_numpy(dtype=float)

  if default is None:

    return None

  return np.full(n, float(default), dtype=float)


def metrics(y_true, y_pred):

  y_true = np.asarray(y_true, dtype=float)

  y_pred = np.asarray(y_pred, dtype=float)

  mask = np.isfinite(y_true) & np.isfinite(y_pred)

  if mask.sum() < 2:

    return {"R²": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "MAPE (%)": np.nan}

  yt = y_true[mask]

  yp = y_pred[mask]

  mape_mask = np.abs(yt) > 1e-12

  return {

    "R²": r2_score(yt, yp),

    "RMSE": np.sqrt(mean_squared_error(yt, yp)),

    "MAE": mean_absolute_error(yt, yp),

    "MSE": mean_squared_error(yt, yp),

    "MAPE (%)": np.mean(np.abs((yt[mape_mask] - yp[mape_mask]) / yt[mape_mask])) * 100 if mape_mask.any() else np.nan,

  }


def add_metric_text(ax, m):

  txt = "\n".join([

    f"R² = {m['R²']:.3f}" if np.isfinite(m["R²"]) else "R² = NA",

    f"RMSE = {m['RMSE']:.4g}" if np.isfinite(m["RMSE"]) else "RMSE = NA",

    f"MAE = {m['MAE']:.4g}" if np.isfinite(m["MAE"]) else "MAE = NA",

    f"MAPE = {m['MAPE (%)']:.2f}%" if np.isfinite(m["MAPE (%)"]) else "MAPE = NA",

  ])

  ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",

      bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


def model_color(index, n_models=1):

    if n_models == 1:

        return "#C00000"                                   


    palette = [

        "#C00000",       

        "#003F5C",             

        "#2F4F4F",                   

        "#7A1E1E",               

        "#005F3F",         

        "#4B2E83",          

        "#8A4B08",         

        "#1B4965",              

        "#6D597A",                

        "#BC6C25",               

    ]

    return palette[index % len(palette)]


def model_linestyle(index, n_models=1):

    if n_models == 1:

        return "-"

    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2)), (0, (7, 2)), (0, (2, 2, 6, 2))]

    return styles[index % len(styles)]


def apply_plot_style():

  plt.rcParams.update({

    "font.family": "Times New Roman",

    "font.size": 13,

    "axes.titlesize": 15,

    "axes.labelsize": 14,

    "xtick.labelsize": 12,

    "ytick.labelsize": 12,

    "legend.fontsize": 10,

    "figure.dpi": 140,

  })


def cumulative_cycles_for_plot(N, stage=None):

  """
  Creates a continuous plotting x-axis for multi-stage data when N resets in each stage.
  It does not change the model calculation; it only improves plot visibility.
  """

  N = np.asarray(N, dtype=float)

  if len(N) == 0:

    return N


  if stage is None:


    if np.all(np.diff(N) >= 0):

      return N


    out = np.zeros_like(N, dtype=float)

    offset = 0.0

    out[0] = max(N[0], 1.0)

    for i in range(1, len(N)):

      if N[i] >= N[i-1]:

        out[i] = offset + N[i]

      else:

        offset = out[i-1]

        out[i] = offset + max(N[i], 1.0)

    return out


  stage = np.asarray(stage)

  out = np.zeros_like(N, dtype=float)

  offset = 0.0

  prev_stage = stage[0]

  prev_local = N[0]

  out[0] = max(N[0], 1.0)


  for i in range(1, len(N)):

    new_stage = stage[i] != prev_stage

    reset_cycle = N[i] < prev_local

    if new_stage or reset_cycle:

      offset = out[i-1]

    out[i] = offset + max(N[i], 1.0)

    prev_stage = stage[i]

    prev_local = N[i]

  return out


def fig_to_png_bytes(fig):

  buffer = io.BytesIO()

  fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")

  buffer.seek(0)

  return buffer


def show_plot_with_download(fig, filename, label="Download plot as PNG"):

  st.pyplot(fig, use_container_width=True)

  st.download_button(

    label,

    data=fig_to_png_bytes(fig),

    file_name=filename,

    mime="image/png",

    key=f"download_{filename}"

 )


def load_data(uploaded):

  name = uploaded.name.lower()

  if name.endswith(".csv"):

    return pd.read_csv(uploaded)

  if name.endswith((".xlsx", ".xls")):

    return pd.read_excel(uploaded)

  raise ValueError("Please upload CSV or Excel file.")


def normalize_col_name(name):

  return (

    str(name)

    .strip()

    .lower()

    .replace(" ", "")

    .replace("_", "")

    .replace("-", "")

    .replace("(", "")

    .replace(")", "")

    .replace("%", "percent")

    .replace("/", "")

 )


def find_best_column(columns, candidates):

  """
  Returns the best-matching column from a list of candidates.
  Matching is case-insensitive and tolerant of spaces/underscores.
  """

  normalized = {normalize_col_name(c): c for c in columns}

  for cand in candidates:

    nc = normalize_col_name(cand)

    if nc in normalized:

      return normalized[nc]


  for cand in candidates:

    nc = normalize_col_name(cand)

    for k, original in normalized.items():

      if nc and (nc in k or k in nc):

        return original

  return "None"


def selectbox_with_auto(label, columns, candidates, key=None):

  options = ["None"] + list(columns)

  auto = find_best_column(columns, candidates)

  index = options.index(auto) if auto in options else 0

  return st.sidebar.selectbox(label, options, index=index, key=key)


def apply_zero_cycle_filter(df, N_col, remove_zero=True):

  if not remove_zero or N_col not in df.columns:

    return df, 0

  N_values = pd.to_numeric(df[N_col], errors="coerce")

  mask = N_values > 0

  removed = int((~mask & N_values.notna()).sum())

  return df.loc[mask].reset_index(drop=True), removed


def required_inputs_for_models(models):

  req = set(["N"])

  for model in models:

    clean = clean_model_label(model)


    if is_ms_model_name(model):

      req.add("stage")


    if "Rahman et al. (2023)" in clean:

      req.add("eps_r")

      if "saturation-based extension" in clean:

        req.add("S")


    if "Rahman & Erlingsson (2015)" in clean:

      req.add("p_or_sigma")

      req.add("q_or_sigma")


    if "Korkiala" in clean:

      req.add("p_or_sigma")

      req.add("q_or_sigma")

      req.add("qf")


    if "Gidel" in clean:

      req.update(["pmax", "qmax", "m", "s"])


    if "Chen" in clean:

      req.update(["pam", "qam", "pini", "qini", "m", "s"])


    if "Tseng & Lytton" in clean:

      req.add("eps_r")


    if "MEPDG" in clean:

      req.update(["eps_r", "eps_v"])


    if "Chow" in clean:

      req.update(["sigma_d", "tau_f", "tau_max"])


    if "Hyde" in clean:

      req.update(["q_or_sigma", "sigma3"])


    if "Veverka" in clean:

      req.add("eps_r")


    if "Lentz" in clean:

      req.update(["sigma_d", "Sd", "eps095Sd"])


    if "Khedr" in clean:

      req.update(["Ro", "MR"])


    if "Lin" in clean:

      req.update(["sigma_max", "tau_f", "tau_max"])


    if "Ooi" in clean:

      req.add("SSR")


  return req


def sort_by_cycles(N, *arrays):

  order = np.argsort(N)

  return (N[order],) + tuple(np.asarray(a)[order] if a is not None else None for a in arrays)


def mean_stress(s1, s2, s3):

  # Mean (hydrostatic) stress used in Rahman & Erlingsson models
  return (s1 + s2 + s3) / 3.0


def deviator_stress(s1, s2, s3):

  return np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))


def constant_or_column(df, col, default):

  n = len(df)

  if col != "None":

    return to_numeric_safe(df[col]).to_numpy(dtype=float)

  return np.full(n, float(default), dtype=float)


def remove_invalid_for_fit(y, *xs):

  mask = np.isfinite(y)

  for x in xs:

    mask &= np.isfinite(x)

  return (y[mask],) + tuple(np.asarray(x)[mask] for x in xs)


def ensure_1d_len(x, n, name):

  # Ensure array length matches dataset size

  if x is None:

    raise ValueError(f"{name} is required.")

  arr = np.asarray(x, dtype=float).reshape(-1)

  if arr.size == 1 and n > 1:

    arr = np.full(n, float(arr[0]), dtype=float)

  if arr.size != n:

    raise ValueError(f"{name} has length {arr.size}, but expected {n}. Check the column mapping.")

  return arr


def require_valid_rows(y, *xs, positive_y=True, positive_N=True, names=None):

  """Filter arrays for calibration and raise a clear error if no rows remain."""

  arrays = [np.asarray(y, dtype=float).reshape(-1)] + [np.asarray(x, dtype=float).reshape(-1) for x in xs]

  n = arrays[0].size

  for k, arr in enumerate(arrays[1:], start=1):

    if arr.size != n:

      label = names[k] if names and k < len(names) else f"input {k}"

      raise ValueError(f"{label} has length {arr.size}, but expected {n}. Check the selected columns.")

  mask = np.ones(n, dtype=bool)

  for arr in arrays:

    mask &= np.isfinite(arr)

  if positive_y:

    mask &= arrays[0] > 0

  if positive_N and len(arrays) > 1:

    mask &= arrays[1] > 0

  if mask.sum() < 3:

    raise ValueError("Not enough valid rows for calibration after filtering. Check N, measured strain, εr and S columns.")

  return tuple(arr[mask] for arr in arrays)


def stage_local_cycles(N, idx, previous_stage_end=None):

  """Return load cycles counted within the current stress path.
  Works with either cumulative N or local N that resets at each stage.
  """

  nvals = np.asarray(N[idx], dtype=float)

  if len(nvals) == 0:

    return nvals

  first = nvals[0]

  if previous_stage_end is not None and first >= previous_stage_end:

    local = nvals - previous_stage_end

  else:

    local = nvals.copy()

  return np.maximum(local, 1e-12)


def ordered_stages(stage):

  try:

    vals = pd.Series(stage).dropna().unique()

    return sorted(vals)

  except Exception:

    return []


def safe_R(q, qf):

  q = np.asarray(q, dtype=float)

  qf = np.asarray(qf, dtype=float)

  return q / np.maximum(qf, 1e-12)


MS_MODEL_SET = {
  "Rahman et al. (2023) [MS]",
  "Rahman et al. (2023) – saturation-based extension [MS]",
  "Rahman & Erlingsson (2015) [MS]",
  "Korkiala–Tanttu (2005) [MS]",
  "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]",
  "Tseng & Lytton (1989) [MS]",
}


def is_ms_model_name(model):

  return str(model) in MS_MODEL_SET


def clean_model_label(model):

  label = str(model).strip()

  return label[:-5].strip() if label.endswith("[MS]") else label


def sweere_1990(N, a, b):


  return (10.0 ** a) * (N ** b)


def rahman_et_al_2023_base(N, a, b, er):


  return a * (N ** (b * er)) * er


def rahman_et_al_2023_base_ms(N, stage, a, b, er):

  out = np.full_like(N, np.nan, dtype=float)

  prev_eps = 0.0

  prev_stage_end = None

  for stg in ordered_stages(stage):

    idx = np.where(stage == stg)[0]

    if len(idx) == 0:

      continue

    idx = idx[np.argsort(N[idx])]

    er_i = np.nanmedian(er[idx])

    if not np.isfinite(er_i) or er_i <= 0 or a <= 0 or b <= 0:

      continue

    if prev_eps <= 0:

      Neq = 0.0

    else:

      Neq = (prev_eps / (a * er_i)) ** (1.0 / (b * er_i))

    local_cycles = stage_local_cycles(N, idx, prev_stage_end)

    effectiveN = np.maximum(local_cycles + Neq, 1e-12)

    out[idx] = a * (effectiveN ** (b * er_i)) * er_i

    prev_eps = out[idx][-1]

    prev_stage_end = np.nanmax(N[idx])

  return out


def rahman_et_al_2023_er_moisture(N, b, c1, c2, S, er):


  N = np.asarray(N, dtype=float).reshape(-1)

  er = ensure_1d_len(er, N.size, "εr")

  S = ensure_1d_len(S, N.size, "S")

  N = np.maximum(N, 1e-12)

  a = c1 * S + c2

  return a * (N ** (b * er)) * er


def calibrate_rahman_2023_saturation(y, N, er, S):

  """
  Calibrate Rahman et al. (2023) saturation-based extension model.

  If saturation has sufficient variation:
      eps_p = (c1*S + c2) * N^(b*er) * er

  If saturation is effectively constant:
      eps_p = a * N^(b*er) * er
      then c1 is set to 0 and c2 is set to a, because c1 and c2
      cannot be separately identified from a single saturation level.
  """

  yfit, Nfit, erfit, Sfit = require_valid_rows(

    y, N, er, S,

    names=["measured permanent strain", "N", "εr", "S"]

  )

  mask = (erfit > 0) & np.isfinite(Sfit)

  yfit, Nfit, erfit, Sfit = yfit[mask], Nfit[mask], erfit[mask], Sfit[mask]

  if len(yfit) < 3:

    raise ValueError("Not enough valid rows after requiring εr > 0 and finite S.")


  unique_S = np.unique(np.round(Sfit[np.isfinite(Sfit)], 8))

  S_range = float(np.nanmax(Sfit) - np.nanmin(Sfit)) if len(Sfit) else 0.0


  if len(unique_S) < 2 or S_range < 1e-9:


    def residual_ab(theta):

      a, b = theta

      pred = rahman_et_al_2023_base(Nfit, a, b, erfit)

      mask_res = np.isfinite(pred) & np.isfinite(yfit)

      if mask_res.sum() < 3:

        return np.ones_like(yfit) * 1e6

      return pred[mask_res] - yfit[mask_res]


    res = least_squares(

      residual_ab,

      x0=[1.0, 250.0],

      bounds=([1e-12, 1e-12], [np.inf, 1000]),

      max_nfev=50000

    )

    if not res.success:

      raise ValueError(res.message)

    a_fit, b_fit = float(res.x[0]), float(res.x[1])

    return {"b": b_fit, "c1": 0.0, "c2": a_fit, "a_equivalent": a_fit, "note": "S was constant; fitted a and b, with c1=0 and c2=a."}


  def residual_ab(theta):

    a, b = theta

    pred = rahman_et_al_2023_base(Nfit, a, b, erfit)

    mask_res = np.isfinite(pred) & np.isfinite(yfit)

    if mask_res.sum() < 3:

      return np.ones_like(yfit) * 1e6

    return pred[mask_res] - yfit[mask_res]


  res_ab = least_squares(

    residual_ab,

    x0=[1.0, 250.0],

    bounds=([1e-12, 1e-12], [np.inf, 1000]),

    max_nfev=20000

  )

  a0 = float(res_ab.x[0]) if res_ab.success else 1.0

  b0 = float(res_ab.x[1]) if res_ab.success else 250.0


  def residual_b_c(theta):

    b, c1, c2 = theta

    a_vals = c1 * Sfit + c2

    pred = a_vals * (np.maximum(Nfit, 1e-12) ** (b * erfit)) * erfit

    mask_res = np.isfinite(pred) & np.isfinite(yfit)


    mask_res &= a_vals > 0

    if mask_res.sum() < 3:

      return np.ones_like(yfit) * 1e6

    return pred[mask_res] - yfit[mask_res]


  res = least_squares(

    residual_b_c,

    x0=[b0, 0.0, a0],

    bounds=([1e-12, -np.inf, -np.inf], [1000, np.inf, np.inf]),

    max_nfev=50000

  )

  if not res.success:

    raise ValueError(res.message)


  b_fit, c1_fit, c2_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])


  a_vals_all = c1_fit * Sfit + c2_fit

  if np.any(~np.isfinite(a_vals_all)) or np.any(a_vals_all <= 0):

    raise ValueError("Calibration produced non-positive a=c1*S+c2 for some rows. Check S units and measured strain units.")


  return {"b": b_fit, "c1": c1_fit, "c2": c2_fit}


def rahman_et_al_2023_er_moisture_ms(N, stage, b, c1, c2, S, er):

  out = np.full_like(N, np.nan, dtype=float)

  prev_eps = 0.0

  prev_stage_end = None

  for stg in ordered_stages(stage):

    idx = np.where(stage == stg)[0]

    if len(idx) == 0:

      continue

    idx = idx[np.argsort(N[idx])]

    er_i = np.nanmedian(er[idx])

    S_i = np.nanmedian(S[idx])

    a_i = c1 * S_i + c2

    if not np.isfinite(er_i) or er_i <= 0 or not np.isfinite(a_i) or a_i <= 0 or b <= 0:

      continue

    if prev_eps <= 0:

      Neq = 0.0

    else:

      Neq = (prev_eps / (a_i * er_i)) ** (1.0 / (b * er_i))

    local_cycles = stage_local_cycles(N, idx, prev_stage_end)

    effectiveN = np.maximum(local_cycles + Neq, 1e-12)

    out[idx] = a_i * (effectiveN ** (b * er_i)) * er_i

    prev_eps = out[idx][-1]

    prev_stage_end = np.nanmax(N[idx])

  return out


def rahman_erlingsson_2015_sf(N, a, b, Sf):


  return a * (N ** (b * Sf)) * Sf


def rahman_erlingsson_2015_sf_ms(N, stage, a, b, Sf):

  out = np.full_like(N, np.nan, dtype=float)

  prev_eps = 0.0

  prev_stage_end = None

  for stg in ordered_stages(stage):

    idx = np.where(stage == stg)[0]

    if len(idx) == 0:

      continue

    idx = idx[np.argsort(N[idx])]

    Sf_i = np.nanmedian(Sf[idx])

    if not np.isfinite(Sf_i) or Sf_i <= 0 or a <= 0 or b <= 0:

      continue

    if prev_eps <= 0:

      Neq = 0.0

    else:

      Neq = (prev_eps / (a * Sf_i)) ** (1.0 / (b * Sf_i))

    local_cycles = stage_local_cycles(N, idx, prev_stage_end)

    effectiveN = np.maximum(local_cycles + Neq, 1e-12)

    out[idx] = a * (effectiveN ** (b * Sf_i)) * Sf_i

    prev_eps = out[idx][-1]

    prev_stage_end = np.nanmax(N[idx])

  return out


def korkiala_tanttu_2005(N, C, b, A, q, qf):

  R = q / np.maximum(qf, 1e-12)

  denom = A - R

  return C * (N ** b) * (R / denom)


def korkiala_tanttu_2005_ms(N, stage, C, b, A, q, qf):

  out = np.full_like(N, np.nan, dtype=float)

  R = safe_R(q, qf)

  prev_eps = 0.0

  prev_stage_end = None

  for stg in ordered_stages(stage):

    idx = np.where(stage == stg)[0]

    if len(idx) == 0:

      continue

    idx = idx[np.argsort(N[idx])]

    R_i = np.nanmedian(R[idx])

    if not np.isfinite(R_i) or R_i <= 0 or C <= 0 or b <= 0 or A <= R_i:

      continue

    if prev_eps <= 0:

      Neq = 0.0

    else:

      Neq = (prev_eps * (A - R_i) / (C * R_i)) ** (1.0 / b)

    local_cycles = stage_local_cycles(N, idx, prev_stage_end)

    effectiveN = np.maximum(local_cycles + Neq, 1e-12)

    out[idx] = C * (effectiveN ** b) * (R_i / (A - R_i))

    prev_eps = out[idx][-1]

    prev_stage_end = np.nanmax(N[idx])

  return out


def modified_korkiala_tanttu_2013_ms(N, stage, C, c, d, A, q, qf):


  out = np.full_like(N, np.nan, dtype=float)

  R = safe_R(q, qf)

  prev_eps = 0.0

  prev_stage_end = None

  for stg in ordered_stages(stage):

    idx = np.where(stage == stg)[0]

    if len(idx) == 0:

      continue

    idx = idx[np.argsort(N[idx])]

    R_i = np.nanmedian(R[idx])

    b_i = c * (R_i ** d) if np.isfinite(R_i) and R_i > 0 else np.nan

    if not np.isfinite(R_i) or R_i <= 0 or C <= 0 or c <= 0 or A <= R_i or not np.isfinite(b_i) or b_i <= 0:

      continue

    if prev_eps <= 0:

      Neq = 0.0

    else:

      Neq = (prev_eps * (A - R_i) / (C * R_i)) ** (1.0 / b_i)

    local_cycles = stage_local_cycles(N, idx, prev_stage_end)

    effectiveN = np.maximum(local_cycles + Neq, 1e-12)

    out[idx] = C * (effectiveN ** b_i) * (R_i / (A - R_i))

    prev_eps = out[idx][-1]

    prev_stage_end = np.nanmax(N[idx])

  return out


def huurman_1997(N, A, B, C, D):

  return A * ((N / 1000.0) ** B) + C * (np.exp(D * (N / 1000.0)) - 1.0)


def gidel_2001(N, eps0, B, n, pmax, qmax, pa, m, s):

  Lmax = np.sqrt(pmax**2 + qmax**2)

  denom = m + (s / pmax) - (qmax / pmax)

  return eps0 * (1.0 - ((N / 100.0) ** (-B))) * ((Lmax / pa) ** n) * (1.0 / denom)


def chen_2014(N, epsp0, B, alpha, pam, qam, pa, m, s, pini, qini):


  # alpha corresponds to exponent 'a' in Chen et al. (2014)
  N = np.maximum(np.asarray(N, dtype=float), 0.0)

  # Avoid division by zero
  pam_safe = np.maximum(np.asarray(pam, dtype=float), 1e-12)

  amp = np.sqrt(pam_safe**2 + np.asarray(qam, dtype=float)**2) / pa

  denom = m * (1.0 + np.asarray(pini, dtype=float) / pam_safe) + s / pam_safe - (np.asarray(qini, dtype=float) + np.asarray(qam, dtype=float)) / pam_safe

  return epsp0 * (1.0 - np.exp(-B * N)) * (amp ** alpha) * (1.0 / denom)


def tseng_lytton_1989(N, eps0, rho, beta, er):


  N = np.maximum(np.asarray(N, dtype=float), 1e-12)

  return er * eps0 * np.exp(-((rho / N) ** beta))


def tseng_lytton_1989_ms(N, stage, eps0, rho, beta, er):

  out = np.full_like(N, np.nan, dtype=float)

  prev_eps = 0.0

  prev_stage_end = None

  for stg in ordered_stages(stage):

    idx = np.where(stage == stg)[0]

    if len(idx) == 0:

      continue

    idx = idx[np.argsort(N[idx])]

    er_i = np.nanmedian(er[idx])

    if not np.isfinite(er_i) or er_i <= 0 or eps0 <= 0 or rho <= 0 or beta <= 0:

      continue

    if prev_eps <= 0:

      Neq = 0.0

    else:

      ratio = prev_eps / (er_i * eps0)

      if ratio <= 0 or ratio >= 1:

        Neq = np.inf

      else:

        Neq = rho * ((-np.log(ratio)) ** (-1.0 / beta))

    if not np.isfinite(Neq):

      out[idx] = prev_eps

    else:

      local_cycles = stage_local_cycles(N, idx, prev_stage_end)

      effectiveN = np.maximum(local_cycles + Neq, 1e-12)

      out[idx] = er_i * eps0 * np.exp(-((rho / effectiveN) ** beta))

      prev_eps = out[idx][-1]

    prev_stage_end = np.nanmax(N[idx])

  return out


def mepdg_nchrp_2003(N, beta_s, eps0, rho, beta, er, ev):


  N = np.maximum(np.asarray(N, dtype=float), 1e-12)

  er_safe = np.maximum(np.asarray(er, dtype=float), 1e-12)

  return beta_s * (eps0 / er_safe) * np.exp(-((rho / N) ** beta)) * ev


def beta_from_moisture_mepdg(w_percent):


  return 10.0 ** (-0.61119 - 0.017638 * w_percent)


def chow_2014(N, A, B, C, D, sigma_d, tau_f, tau_max):

  return A * (N ** B) * (sigma_d ** C) * ((tau_f / tau_max) ** D)


def barksdale_1972(N, a, b):


  N = np.maximum(np.asarray(N, dtype=float), 1.000001)

  return a + b * np.log10(np.log10(N))


def hyde_1974(a, q, sigma3):


  return a * (q / sigma3)


def veverka_1979(N, mu, alpha, er):


  return mu * er * (N ** alpha)


def lentz_baladi_1980(sigma_d, Sd, eps095Sd, n, m):


  r = sigma_d / Sd

  denom = 1.0 - (r * m / eps095Sd)

  return (r * (eps095Sd ** n)) / denom


def khedr_1985(N, s1, s2, s3p, m, Ro, MR):


  return s1 * (Ro ** s2) * (MR ** s3p) * (N ** (1.0 - m))


def paute_1988(N, A, D, epsp0):


  sqrtN = np.sqrt(np.maximum(N, 0.0))

  return A * sqrtN / (sqrtN + D) + epsp0


def hornych_1993(N, epsp100, A, B):


  N = np.maximum(np.asarray(N, dtype=float), 1e-12)

  return epsp100 + A * (1.0 - ((N / 100.0) ** (-B)))


def wolff_visser_1994(N, a, b, c):


  return (c * N + a) * (1.0 - np.exp(-b * N))


def lin_2019(N, A, B, C, D, E, sigma_max, pa, tau_f, tau_max):


  return A * (N ** B) * ((sigma_max / pa) ** C) * ((pa / tau_f) ** D) * np.exp(E * (tau_max / sigma_max))


def ooi_2021(N, a, b, c, d, SSR):


  Acoef = a * (SSR ** b)

  Bcoef = c * SSR + d

  return Acoef * (N ** Bcoef)


SINGLE_MODELS = [

  "Barksdale (1972)",

  "Hyde (1974)",

  "Veverka (1979)",

  "Lentz & Baladi (1980)",

  "Khedr (1985)",

  "Paute et al. (1988)",

  "Sweere (1990)",

  "Hornych et al. (1993)",

  "Wolff & Visser (1994)",

  "Rahman et al. (2023)",

  "Rahman et al. (2023) – saturation-based extension",

  "Rahman & Erlingsson (2015)",

  "Korkiala–Tanttu (2005)",

  "Huurman (1997)",

  "Gidel et al. (2001)",

  "Chen et al. (2014)",

  "Tseng & Lytton (1989)",

  "MEPDG / NCHRP (2003)",

  "Chow et al. (2014)",

  "Lin et al. (2019)",

  "Ooi (2021)",

]


MS_MODELS = [

  "Rahman et al. (2023) [MS]",

  "Rahman et al. (2023) – saturation-based extension [MS]",

  "Rahman & Erlingsson (2015) [MS]",

  "Korkiala–Tanttu (2005) [MS]",

  "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]",

  "Tseng & Lytton (1989) [MS]",

]


PARAMETER_NOTES = {

  "Barksdale (1972)": "Early N-only logarithmic model: εp = a + b log10(log10 N). Use only for N > 1.",

  "Hyde (1974)": "Simple stress-ratio model: εp = a(q/σ3). Requires deviator stress q and confining stress σ3.",

  "Veverka (1979)": "Relates permanent and resilient strain: εp = μ εr N^α. Requires εr.",

  "Lentz & Baladi (1980)": "Cyclic-monotonic stress-deformation correlation. Requires σd, Sd and ε0.95Sd. Implemented using the explicit rearranged form of the tabulated equation.",

  "Khedr (1985)": "Relates permanent strain to octahedral stress ratio Ro, resilient modulus MR and N.",

  "Paute et al. (1988)": "Hyperbolic square-root model: εp = A√N/(√N + D) + εp0. Includes early-cycle rearrangement.",


  "Sweere (1990)": "Parameters a and b are material/stress-level dependent. The implemented form is εp = 10^a N^b.",


  "Hornych et al. (1993)": "Asymptotic stabilization model: εp = εp(100) + A[1-(N/100)^(-B)].",

  "Wolff & Visser (1994)": "HVS-based model: εp = (cN+a)(1-exp(-bN)).",

  "Rahman et al. (2023)": "Base resilient-strain model: εp = a N^(b εr) εr. Requires εr and calibrated a, b.",

  "Rahman et al. (2023) [MS]": "MS time-hardening form of the base resilient-strain model. Requires stage ID and εr.",

  "Rahman et al. (2023) – saturation-based extension": "Saturation-dependent parameter relation: a = c1 S + c2 with εp = a N^(b εr) εr. Requires εr and S (%).",

  "Rahman et al. (2023) – saturation-based extension [MS]": "MS time-hardening form with a_i = c1 S_i + c2 for each stress path.",

  "Rahman & Erlingsson (2015)": "Requires p and q, or σ1, σ2, σ3. Sf = (q/pa)/(p/pa)^α. Parameters a, b and α are material-specific.",

  "Rahman & Erlingsson (2015) [MS]": "Requires stage ID and stress function Sf. Equivalent cycles are computed stage-by-stage.",

  "Korkiala–Tanttu (2005)": "Requires q and qf. R = q/qf must remain lower than A; otherwise the model becomes singular.",

  "Korkiala–Tanttu (2005) [MS]": "MS time-hardening extension from Erlingsson & Rahman (2013). Requires stage ID, q and qf.",

  "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]": "Erlingsson & Rahman (2013) modification: b_i = c(R_i)^d in the MS Korkiala–Tanttu framework.",

  "Huurman (1997)": "A, B, C and D are stress-level dependent parameters.",

  "Gidel et al. (2001)": "Requires pmax, qmax, Mohr–Coulomb parameters m and s, and calibrated ε0, B, n.",

  "Chen et al. (2014)": "Modified Gidel-based railway-subgrade model. Requires pam, qam, pini, qini, m and s; calibrated parameters are εp0, B and α. The model predicts permanent strain; settlement is obtained separately as s(N)=ΣH_iεp,i.",

  "Tseng & Lytton (1989)": "Original exponential model. Requires resilient strain εr and calibrated parameters ε0, ρ and β.",

  "Tseng & Lytton (1989) [MS]": "MS time-hardening extension from Erlingsson & Rahman (2013). Requires stage ID and εr.",

  "MEPDG / NCHRP (2003)": "MEPDG/NCHRP design implementation. Requires βs, ε0, ρ, β, laboratory resilient strain εr and vertical resilient strain εv. βs is often reported near 1.673 for UGMs; β can optionally be estimated from moisture content w. Intended for single-stage/design use; for multistage RLT, use the Tseng & Lytton MS time-hardening option unless you intentionally apply an unvalidated stage-wise approximation.",

  "Chow et al. (2014)": "Requires σd, τf, τmax and calibrated A, B, C, D.",

  "Lin et al. (2019)": "Modified UIUC-type model. Requires σmax, τf and τmax with parameters A, B, C, D, E.",

  "Ooi (2021)": "Stress-strength-ratio model: εp=A N^B, A=a SSR^b, B=c SSR+d. Requires SSR.",

}


def fit_model(model_name, df, N, y, p, q, er, S, Sf, stage, pa, fixed):

  """Return fitted params dict, predictions, status."""

  try:


    if model_name == "Sweere (1990)":

      yfit, Nfit = remove_invalid_for_fit(y, N)

      mask = (yfit > 0) & (Nfit > 0)

      yfit, Nfit = yfit[mask], Nfit[mask]

      if len(yfit) < 2:

        raise ValueError("Need at least two positive measured strain values.")


      b0, a0 = np.polyfit(np.log10(Nfit), np.log10(yfit), 1)

      popt, _ = curve_fit(lambda n, a, b: sweere_1990(n, a, b), Nfit, yfit, p0=[a0, b0], maxfev=20000)

      params = {"a": float(popt[0]), "b": float(popt[1])}

      pred = sweere_1990(N, **params)

      return params, pred, "calibrated"


    if model_name == "Rahman et al. (2023)":

      if er is None: raise ValueError("εr is required.")

      yfit, Nfit, erfit = remove_invalid_for_fit(y, N, er)

      mask = (yfit > 0) & (Nfit > 0) & (erfit > 0)

      yfit, Nfit, erfit = yfit[mask], Nfit[mask], erfit[mask]

      popt, _ = curve_fit(lambda x, a, b: rahman_et_al_2023_base(x[0], a, b, x[1]),

                (Nfit, erfit), yfit, p0=[1.0, 250.0], bounds=([1e-12, 1e-12], [np.inf, 1000]), maxfev=50000)

      params = {"a": float(popt[0]), "b": float(popt[1])}

      pred = rahman_et_al_2023_base(N, params["a"], params["b"], er)

      return params, pred, "calibrated"


    if model_name == "Rahman et al. (2023) – saturation-based extension":

      if er is None or S is None:

        raise ValueError("εr and S (%) are required. Select the correct columns in Column mapping.")

      er = ensure_1d_len(er, len(N), "εr")

      S = ensure_1d_len(S, len(N), "S")

      params = calibrate_rahman_2023_saturation(y, N, er, S)

      pred = rahman_et_al_2023_er_moisture(N, params["b"], params["c1"], params["c2"], S, er)

      return params, pred, "calibrated"


    if model_name == "Rahman et al. (2023) [MS]":

      if er is None or stage is None: raise ValueError("stage and εr are required.")

      def residual(theta):

        a, b = theta

        pred = rahman_et_al_2023_base_ms(N, stage, a, b, er)

        mask = np.isfinite(y) & np.isfinite(pred)

        return pred[mask] - y[mask]

      res = least_squares(residual, x0=[1.0, 250.0], bounds=([1e-12, 1e-12], [np.inf, 1000]), max_nfev=50000)

      params = {"a": float(res.x[0]), "b": float(res.x[1])}

      pred = rahman_et_al_2023_base_ms(N, stage, params["a"], params["b"], er)

      return params, pred, "calibrated"


    if model_name == "Rahman et al. (2023) – saturation-based extension [MS]":

      if er is None or S is None or stage is None:

        raise ValueError("stage, εr and S (%) are required.")

      er = ensure_1d_len(er, len(N), "εr")

      S = ensure_1d_len(S, len(N), "S")


      unique_S = np.unique(np.round(S[np.isfinite(S)], 8))

      if len(unique_S) < 2:

        def residual_ab(theta):

          a, b = theta

          pred = rahman_et_al_2023_base_ms(N, stage, a, b, er)

          mask = np.isfinite(y) & np.isfinite(pred)

          if mask.sum() < 3:

            return np.ones_like(y, dtype=float) * 1e6

          return pred[mask] - y[mask]

        res = least_squares(

          residual_ab,

          x0=[1.0, 250.0],

          bounds=([1e-12, 1e-12], [np.inf, 1000]),

          max_nfev=50000

        )

        if not res.success:

          raise ValueError(res.message)

        params = {"b": float(res.x[1]), "c1": 0.0, "c2": float(res.x[0]), "a_equivalent": float(res.x[0]), "note": "S was constant; fitted a and b, with c1=0 and c2=a."}

      else:

        def residual(theta):

          b, c1, c2 = theta

          pred = rahman_et_al_2023_er_moisture_ms(N, stage, b, c1, c2, S, er)

          mask = np.isfinite(y) & np.isfinite(pred)

          if mask.sum() < 3:

            return np.ones_like(y, dtype=float) * 1e6

          return pred[mask] - y[mask]

        res = least_squares(

          residual,

          x0=[250.0, 0.0, 1.0],

          bounds=([1e-12, -np.inf, -np.inf], [1000, np.inf, np.inf]),

          max_nfev=50000

        )

        if not res.success:

          raise ValueError(res.message)

        params = {"b": float(res.x[0]), "c1": float(res.x[1]), "c2": float(res.x[2])}


      pred = rahman_et_al_2023_er_moisture_ms(N, stage, params["b"], params["c1"], params["c2"], S, er)

      return params, pred, "calibrated"


    if model_name == "Rahman & Erlingsson (2015)":

      if p is None or q is None: raise ValueError("p and q are required.")


      yfit, Nfit, pfit, qfit = remove_invalid_for_fit(y, N, p, q)

      mask = (yfit > 0) & (Nfit > 0) & (pfit > 0) & (qfit > 0)

      yfit, Nfit, pfit, qfit = yfit[mask], Nfit[mask], pfit[mask], qfit[mask]

      def f(x, a, b, alpha):

        n, pp, qq = x

        sf = (qq / pa) / ((pp / pa) ** alpha)

        return rahman_erlingsson_2015_sf(n, a, b, sf)

      popt, _ = curve_fit(f, (Nfit, pfit, qfit), yfit, p0=[1.0, 1.0, 1.0],

                bounds=([1e-12, 1e-12, -10], [np.inf, np.inf, 10]), maxfev=50000)

      alpha = float(popt[2])

      sf_all = (q / pa) / ((p / pa) ** alpha)

      params = {"a": float(popt[0]), "b": float(popt[1]), "alpha": alpha}

      pred = rahman_erlingsson_2015_sf(N, params["a"], params["b"], sf_all)

      return params, pred, "calibrated"


    if model_name == "Rahman & Erlingsson (2015) [MS]":

      if p is None or q is None or stage is None: raise ValueError("p, q and stage are required.")

      def residual(theta):

        a, b, alpha = theta

        sf = (q / pa) / ((p / pa) ** alpha)

        pred = rahman_erlingsson_2015_sf_ms(N, stage, a, b, sf)

        mask = np.isfinite(y) & np.isfinite(pred)

        return pred[mask] - y[mask]

      res = least_squares(residual, x0=[1.0, 1.0, 1.0], bounds=([1e-12, 1e-12, -10], [np.inf, np.inf, 10]), max_nfev=50000)

      params = {"a": float(res.x[0]), "b": float(res.x[1]), "alpha": float(res.x[2])}

      sf_all = (q / pa) / ((p / pa) ** params["alpha"])

      pred = rahman_erlingsson_2015_sf_ms(N, stage, params["a"], params["b"], sf_all)

      return params, pred, "calibrated"


    if model_name == "Korkiala–Tanttu (2005)":

      qf = fixed["qf"]

      if q is None or qf is None: raise ValueError("q and qf are required.")

      yfit, Nfit, qfit, qffit = remove_invalid_for_fit(y, N, q, qf)

      mask = (yfit > 0) & (Nfit > 0) & (qfit > 0) & (qffit > 0)

      yfit, Nfit, qfit, qffit = yfit[mask], Nfit[mask], qfit[mask], qffit[mask]

      def f(x, C, b, A):

        n, qq, qqf = x

        return korkiala_tanttu_2005(n, C, b, A, qq, qqf)

      popt, _ = curve_fit(f, (Nfit, qfit, qffit), yfit, p0=[1e-5, 0.2, 1.05],

                bounds=([1e-12, -10, 1e-6], [np.inf, 10, 10]), maxfev=50000)

      params = {"C": float(popt[0]), "b": float(popt[1]), "A": float(popt[2])}

      pred = korkiala_tanttu_2005(N, params["C"], params["b"], params["A"], q, qf)

      return params, pred, "calibrated"


    if model_name == "Korkiala–Tanttu (2005) [MS]":

      if q is None or stage is None: raise ValueError("q, qf and stage are required.")

      qf = fixed.get("qf")

      def residual(theta):

        C, b = theta

        pred = korkiala_tanttu_2005_ms(N, stage, C, b, 1.05, q, qf)

        mask = np.isfinite(y) & np.isfinite(pred)

        return pred[mask] - y[mask]

      res = least_squares(residual, x0=[1e-4, 0.2], bounds=([1e-12, 1e-12], [np.inf, 10]), max_nfev=50000)

      # A is fixed to 1.05 as commonly recommended in literature
      params = {"C": float(res.x[0]), "b": float(res.x[1]), "A": 1.05}

      pred = korkiala_tanttu_2005_ms(N, stage, params["C"], params["b"], params["A"], q, qf)

      return params, pred, "calibrated"


    if model_name == "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]":

      if q is None or stage is None: raise ValueError("q, qf and stage are required.")

      qf = fixed.get("qf")

      def residual(theta):

        C, c, d = theta

        pred = modified_korkiala_tanttu_2013_ms(N, stage, C, c, d, 1.05, q, qf)

        mask = np.isfinite(y) & np.isfinite(pred)

        return pred[mask] - y[mask]

      res = least_squares(residual, x0=[1e-4, 0.2, 0.1], bounds=([1e-12, 1e-12, -10], [np.inf, 10, 10]), max_nfev=50000)

      # A is fixed to 1.05 as commonly recommended in literature
      params = {"C": float(res.x[0]), "c": float(res.x[1]), "d": float(res.x[2]), "A": 1.05}

      pred = modified_korkiala_tanttu_2013_ms(N, stage, params["C"], params["c"], params["d"], params["A"], q, qf)

      return params, pred, "calibrated"


    if model_name == "Huurman (1997)":

      yfit, Nfit = remove_invalid_for_fit(y, N)

      mask = (yfit > 0) & (Nfit > 0)

      yfit, Nfit = yfit[mask], Nfit[mask]

      popt, _ = curve_fit(lambda n, A, B, C, D: huurman_1997(n, A, B, C, D),

                Nfit, yfit, p0=[1e-5, 0.2, 1e-6, 1e-3],

                bounds=([0, -10, 0, -10], [np.inf, 10, np.inf, 10]), maxfev=50000)

      params = {"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2]), "D": float(popt[3])}

      pred = huurman_1997(N, **params)

      return params, pred, "calibrated"


    if model_name == "Gidel et al. (2001)":

      pmax, qmax, m, s = fixed["pmax"], fixed["qmax"], fixed["m"], fixed["s"]

      yfit, Nfit, pfit, qfit = remove_invalid_for_fit(y, N, pmax, qmax)

      mask = (yfit > 0) & (Nfit > 100) & (pfit > 0) & (qfit > 0)

      yfit, Nfit, pfit, qfit = yfit[mask], Nfit[mask], pfit[mask], qfit[mask]

      def f(x, eps0, B, n_exp):

        Ndata, pp, qq = x

        return gidel_2001(Ndata, eps0, B, n_exp, pp, qq, pa, m, s)

      popt, _ = curve_fit(f, (Nfit, pfit, qfit), yfit, p0=[1e-4, 0.2, 1.0],

                bounds=([1e-12, 1e-12, -10], [np.inf, 10, 10]), maxfev=50000)

      params = {"eps0": float(popt[0]), "B": float(popt[1]), "n": float(popt[2]), "m": float(m), "s": float(s)}

      pred = gidel_2001(N, params["eps0"], params["B"], params["n"], pmax, qmax, pa, m, s)

      return params, pred, "calibrated"


    if model_name == "Chen et al. (2014)":

      pam, qam, pini, qini, m, s = fixed["pam"], fixed["qam"], fixed["pini"], fixed["qini"], fixed["m"], fixed["s"]

      yfit, Nfit, pamf, qamf, pinif, qinif = remove_invalid_for_fit(y, N, pam, qam, pini, qini)

      denomf = m * (1.0 + pinif / np.maximum(pamf, 1e-12)) + s / np.maximum(pamf, 1e-12) - (qinif + qamf) / np.maximum(pamf, 1e-12)

      mask = (yfit > 0) & (Nfit >= 0) & (pamf > 0) & (qamf >= 0) & np.isfinite(denomf) & (denomf > 0)

      yfit, Nfit, pamf, qamf, pinif, qinif = yfit[mask], Nfit[mask], pamf[mask], qamf[mask], pinif[mask], qinif[mask]

      def f(x, epsp0, B, alpha):

        n, ppam, qqam, ppini, qqini = x

        return chen_2014(n, epsp0, B, alpha, ppam, qqam, pa, m, s, ppini, qqini)

      popt, _ = curve_fit(f, (Nfit, pamf, qamf, pinif, qinif), yfit, p0=[1e-4, 1e-3, 0.3],

                bounds=([1e-12, 1e-12, -10], [np.inf, 10, 10]), maxfev=50000)

      params = {"epsp0": float(popt[0]), "B": float(popt[1]), "alpha": float(popt[2]), "m": float(m), "s": float(s)}

      pred = chen_2014(N, params["epsp0"], params["B"], params["alpha"], pam, qam, pa, m, s, pini, qini)

      return params, pred, "calibrated"


    if model_name == "Tseng & Lytton (1989)":

      if er is None: raise ValueError("εr is required.")

      yfit, Nfit, erfit = remove_invalid_for_fit(y, N, er)

      mask = (yfit > 0) & (Nfit > 0) & (erfit > 0)

      yfit, Nfit, erfit = yfit[mask], Nfit[mask], erfit[mask]

      def f(x, eps0, rho, beta):

        n, ee = x

        return tseng_lytton_1989(n, eps0, rho, beta, ee)

      popt, _ = curve_fit(

        f, (Nfit, erfit), yfit,

        p0=[1.0, 1000.0, 0.2],

        bounds=([1e-12, 1e-12, 1e-12], [np.inf, np.inf, 10.0]),

        maxfev=50000

     )

      params = {"eps0": float(popt[0]), "rho": float(popt[1]), "beta": float(popt[2])}

      pred = tseng_lytton_1989(N, params["eps0"], params["rho"], params["beta"], er)

      return params, pred, "calibrated"


    if model_name == "Tseng & Lytton (1989) [MS]":

      if er is None or stage is None: raise ValueError("stage and εr are required.")

      yfit, Nfit, erfit = remove_invalid_for_fit(y, N, er)

      def residual(theta):

        eps0, rho, beta = theta

        pred = tseng_lytton_1989_ms(N, stage, eps0, rho, beta, er)

        mask = np.isfinite(y) & np.isfinite(pred)

        return pred[mask] - y[mask]

      res = least_squares(residual, x0=[1.0, 1000.0, 0.2], bounds=([1e-12, 1e-12, 1e-12], [np.inf, np.inf, 10]), max_nfev=50000)

      params = {"eps0": float(res.x[0]), "rho": float(res.x[1]), "beta": float(res.x[2])}

      pred = tseng_lytton_1989_ms(N, stage, params["eps0"], params["rho"], params["beta"], er)

      return params, pred, "calibrated"


    if model_name == "MEPDG / NCHRP (2003)":

      ev = fixed.get("ev")

      if er is None or ev is None: raise ValueError("εr and εv are required.")

      yfit, Nfit, erfit, evfit = remove_invalid_for_fit(y, N, er, ev)

      mask = (yfit > 0) & (Nfit > 0) & (erfit > 0) & np.isfinite(evfit)

      yfit, Nfit, erfit, evfit = yfit[mask], Nfit[mask], erfit[mask], evfit[mask]

      def f(x, beta_s, eps0, rho, beta):

        n, ee, vv = x

        return mepdg_nchrp_2003(n, beta_s, eps0, rho, beta, ee, vv)

      popt, _ = curve_fit(

        f, (Nfit, erfit, evfit), yfit,

        p0=[1.673, 1.0, 1000.0, 0.2],

        bounds=([1e-12, 1e-12, 1e-12, 1e-12], [np.inf, np.inf, np.inf, 10.0]),

        maxfev=50000

     )

      params = {"beta_s": float(popt[0]), "eps0": float(popt[1]), "rho": float(popt[2]), "beta": float(popt[3])}

      pred = mepdg_nchrp_2003(N, params["beta_s"], params["eps0"], params["rho"], params["beta"], er, ev)

      return params, pred, "calibrated"


    if model_name == "Chow et al. (2014)":

      sigma_d, tau_f, tau_max = fixed["sigma_d"], fixed["tau_f"], fixed["tau_max"]

      yfit, Nfit, sdfit, tffit, tmfit = remove_invalid_for_fit(y, N, sigma_d, tau_f, tau_max)

      mask = (yfit > 0) & (Nfit > 0) & (sdfit > 0) & (tffit > 0) & (tmfit > 0)

      yfit, Nfit, sdfit, tffit, tmfit = yfit[mask], Nfit[mask], sdfit[mask], tffit[mask], tmfit[mask]

      def f(x, A, B, C, D):

        n, sd, tf, tm = x

        return chow_2014(n, A, B, C, D, sd, tf, tm)

      popt, _ = curve_fit(f, (Nfit, sdfit, tffit, tmfit), yfit, p0=[1e-6, 0.2, 1.0, 1.0],

                bounds=([1e-12, -10, -10, -10], [np.inf, 10, 10, 10]), maxfev=50000)

      params = {"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2]), "D": float(popt[3])}

      pred = chow_2014(N, params["A"], params["B"], params["C"], params["D"], sigma_d, tau_f, tau_max)

      return params, pred, "calibrated"


    if model_name == "Barksdale (1972)":

      yfit, Nfit = remove_invalid_for_fit(y, N)

      mask = (yfit > 0) & (Nfit > 1.0)

      yfit, Nfit = yfit[mask], Nfit[mask]

      popt, _ = curve_fit(lambda n, a, b: barksdale_1972(n, a, b), Nfit, yfit, p0=[0.0, 0.01], maxfev=50000)

      params = {"a": float(popt[0]), "b": float(popt[1])}

      return params, barksdale_1972(N, **params), "calibrated"


    if model_name == "Hyde (1974)":

      sigma3 = fixed.get("sigma3")

      if q is None or sigma3 is None: raise ValueError("q and σ3 are required.")

      yfit, qfit, s3fit = remove_invalid_for_fit(y, q, sigma3)

      mask = (yfit > 0) & (qfit > 0) & (s3fit > 0)

      yfit, qfit, s3fit = yfit[mask], qfit[mask], s3fit[mask]

      X = qfit / s3fit

      a = np.sum(X*yfit)/np.sum(X*X)

      params = {"a": float(a)}

      return params, hyde_1974(params["a"], q, sigma3), "calibrated"


    if model_name == "Veverka (1979)":

      if er is None: raise ValueError("εr is required.")

      yfit, Nfit, erfit = remove_invalid_for_fit(y, N, er)

      mask = (yfit > 0) & (Nfit > 0) & (erfit > 0)

      yfit, Nfit, erfit = yfit[mask], Nfit[mask], erfit[mask]

      popt, _ = curve_fit(lambda x, mu, alpha: veverka_1979(x[0], mu, alpha, x[1]), (Nfit, erfit), yfit, p0=[1.0, 0.2], bounds=([1e-12, -10], [np.inf, 10]), maxfev=50000)

      params = {"mu": float(popt[0]), "alpha": float(popt[1])}

      return params, veverka_1979(N, params["mu"], params["alpha"], er), "calibrated"


    if model_name == "Lentz & Baladi (1980)":

      sigma_d, Sd, eps095Sd = fixed.get("sigma_d"), fixed.get("Sd"), fixed.get("eps095Sd")

      if sigma_d is None or Sd is None or eps095Sd is None: raise ValueError("σd, Sd and ε0.95Sd are required.")

      yfit, sdfit, Sdfit, epsfit = remove_invalid_for_fit(y, sigma_d, Sd, eps095Sd)

      mask = (yfit > 0) & (sdfit > 0) & (Sdfit > 0) & (epsfit > 0)

      yfit, sdfit, Sdfit, epsfit = yfit[mask], sdfit[mask], Sdfit[mask], epsfit[mask]

      popt, _ = curve_fit(lambda x, n, m: lentz_baladi_1980(x[0], x[1], x[2], n, m), (sdfit, Sdfit, epsfit), yfit, p0=[1.0, 0.1], maxfev=50000)

      params = {"n": float(popt[0]), "m": float(popt[1])}

      return params, lentz_baladi_1980(sigma_d, Sd, eps095Sd, params["n"], params["m"]), "calibrated"


    if model_name == "Khedr (1985)":

      Ro, MR = fixed.get("Ro"), fixed.get("MR")

      if Ro is None or MR is None: raise ValueError("Ro and MR are required.")

      yfit, Nfit, Rofit, MRfit = remove_invalid_for_fit(y, N, Ro, MR)

      mask = (yfit > 0) & (Nfit > 0) & (Rofit > 0) & (MRfit > 0)

      yfit, Nfit, Rofit, MRfit = yfit[mask], Nfit[mask], Rofit[mask], MRfit[mask]

      popt, _ = curve_fit(lambda x, s1, s2, s3p, m: khedr_1985(x[0], s1, s2, s3p, m, x[1], x[2]), (Nfit, Rofit, MRfit), yfit, p0=[1e-6, 1.0, 1.0, 0.2], maxfev=50000)

      params = {"s1": float(popt[0]), "s2": float(popt[1]), "s3": float(popt[2]), "m": float(popt[3])}

      return params, khedr_1985(N, params["s1"], params["s2"], params["s3"], params["m"], Ro, MR), "calibrated"


    if model_name == "Paute et al. (1988)":

      yfit, Nfit = remove_invalid_for_fit(y, N)

      mask = (yfit > 0) & (Nfit > 0)

      yfit, Nfit = yfit[mask], Nfit[mask]

      popt, _ = curve_fit(lambda n, A, D, epsp0: paute_1988(n, A, D, epsp0), Nfit, yfit, p0=[np.nanmax(yfit), 10.0, np.nanmin(yfit)], maxfev=50000)

      params = {"A": float(popt[0]), "D": float(popt[1]), "epsp0": float(popt[2])}

      return params, paute_1988(N, **params), "calibrated"


    if model_name == "Hornych et al. (1993)":

      yfit, Nfit = remove_invalid_for_fit(y, N)

      mask = (yfit > 0) & (Nfit > 100)

      yfit, Nfit = yfit[mask], Nfit[mask]

      popt, _ = curve_fit(lambda n, epsp100, A, B: hornych_1993(n, epsp100, A, B), Nfit, yfit, p0=[np.nanmin(yfit), np.nanmax(yfit)-np.nanmin(yfit), 0.2], bounds=([0, -np.inf, 1e-12], [np.inf, np.inf, 10]), maxfev=50000)

      params = {"epsp100": float(popt[0]), "A": float(popt[1]), "B": float(popt[2])}

      return params, hornych_1993(N, **params), "calibrated"


    if model_name == "Wolff & Visser (1994)":

      yfit, Nfit = remove_invalid_for_fit(y, N)

      mask = (yfit > 0) & (Nfit > 0)

      yfit, Nfit = yfit[mask], Nfit[mask]

      popt, _ = curve_fit(lambda n, a, b, c: wolff_visser_1994(n, a, b, c), Nfit, yfit, p0=[np.nanmin(yfit), 1e-3, 1e-8], maxfev=50000)

      params = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}

      return params, wolff_visser_1994(N, **params), "calibrated"


    if model_name == "Lin et al. (2019)":

      sigma_max, tau_f, tau_max = fixed.get("sigma_max"), fixed.get("tau_f"), fixed.get("tau_max")

      if sigma_max is None or tau_f is None or tau_max is None: raise ValueError("σmax, τf and τmax are required.")

      yfit, Nfit, smfit, tffit, tmfit = remove_invalid_for_fit(y, N, sigma_max, tau_f, tau_max)

      mask = (yfit > 0) & (Nfit > 0) & (smfit > 0) & (tffit > 0) & (tmfit > 0)

      yfit, Nfit, smfit, tffit, tmfit = yfit[mask], Nfit[mask], smfit[mask], tffit[mask], tmfit[mask]

      popt, _ = curve_fit(lambda x, A, B, C, D, E: lin_2019(x[0], A, B, C, D, E, x[1], pa, x[2], x[3]), (Nfit, smfit, tffit, tmfit), yfit, p0=[1e-6, 0.2, 1.0, 1.0, 0.1], maxfev=50000)

      params = {"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2]), "D": float(popt[3]), "E": float(popt[4])}

      return params, lin_2019(N, params["A"], params["B"], params["C"], params["D"], params["E"], sigma_max, pa, tau_f, tau_max), "calibrated"


    if model_name == "Ooi (2021)":

      SSR = fixed.get("SSR")

      if SSR is None: raise ValueError("SSR is required.")

      yfit, Nfit, SSRfit = remove_invalid_for_fit(y, N, SSR)

      mask = (yfit > 0) & (Nfit > 0) & (SSRfit > 0)

      yfit, Nfit, SSRfit = yfit[mask], Nfit[mask], SSRfit[mask]

      popt, _ = curve_fit(lambda x, a, b, c, d: ooi_2021(x[0], a, b, c, d, x[1]), (Nfit, SSRfit), yfit, p0=[1e-6, 1.0, 0.1, 0.1], maxfev=50000)

      params = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]), "d": float(popt[3])}

      return params, ooi_2021(N, params["a"], params["b"], params["c"], params["d"], SSR), "calibrated"


    raise ValueError("Unsupported model.")


  except Exception as e:

    return {}, np.full_like(N, np.nan, dtype=float), f"calibration failed: {e}"


st.markdown(

  "<h1 style='text-align: center; color:#0B3C5D;'>UGM Permanent Strain Prediction and Calibration</h1>",

  unsafe_allow_html=True

)


st.markdown(

  "<p style='text-align: center; font-size:16px; color:#2F4F4F;'>"

  "<b>Developed by:</b> Mohammad Jawed Roshan, António Gomes Correia, "

  "Ionut Dragos Moldovan, Miguel Azenha"

  "</p>",

  unsafe_allow_html=True

)


st.markdown("<hr style='border:1px solid #0B3C5D;'>", unsafe_allow_html=True)


st.markdown(

  "<p style='text-align: center; font-size:18px;'></p>",

  unsafe_allow_html=True

)


st.sidebar.header("Mode")

mode = st.sidebar.radio("Analysis mode", ["Prediction with known parameters", "Automatic calibration from measured data"])


st.sidebar.header("Model selection")

selected_single = st.sidebar.multiselect("Single-stage models", SINGLE_MODELS)

selected_ms = st.sidebar.multiselect("Multi-stage models", MS_MODELS)

selected_models = selected_single + selected_ms


st.sidebar.header("Plot settings")

strain_unit = st.sidebar.text_input("Permanent strain unit for plot labels", value="%")


st.sidebar.header("Data upload")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])


if not selected_models:

  st.info("Select the analysis mode and at least one model from the sidebar. Then upload your data file.")

  st.stop()


if uploaded is None:

  st.stop()


try:

  df = load_data(uploaded)

except Exception as e:

  st.error(str(e))

  st.stop()


cols = list(df.columns)

n = len(df)


st.subheader("Uploaded data")

st.dataframe(df.head(50), use_container_width=True)


st.sidebar.header("Automatic column identification")

auto_map = st.sidebar.checkbox("Automatically identify columns", value=True)

remove_zero_N = st.sidebar.checkbox("Automatically remove rows where N = 0", value=True)


st.sidebar.header("Column mapping")


if auto_map:

  N_col = selectbox_with_auto(

    "Loading cycles N", cols,

    ["CorrectedCycle", "Corrected_Cycle", "Cycle", "Cycles", "N", "LoadCycles", "LoadingCycles", "NumberOfCycles"]

 )

  measured_col = selectbox_with_auto(

    "Measured permanent strain", cols,

    ["Permanent_strain (%)", "Permanent strain (%)", "Permanent_strain", "PermanentStrain", "eps_p_measured", "epsilon_p", "εp", "epsp"]

 )

  stage_col = selectbox_with_auto(

    "Stage column for MS models", cols,

    ["StageNumber", "Stage", "StressStage", "Sequence", "StressPath", "PathNumber"]

 )

  epsr_col = selectbox_with_auto(

    "Resilient strain εr", cols,

    ["eps_r", "epsilon_r", "ResilientStrain", "Resilient_strain", "Resilient strain", "εr"]

 )

  S_col = selectbox_with_auto(

    "Degree of saturation S (%)", cols,

    ["S", "S (%)", "Sr", "Sr (%)", "Saturation", "DegreeOfSaturation", "Degree_of_saturation", "Degree of saturation", "Saturation (%)", "DegreeSaturationPercent"]

 )

  p_col = selectbox_with_auto(

    "Mean stress p", cols,

    ["MeanEffStress_kPa", "MeanStress_kPa", "Mean stress", "Mean stress p", "p", "p_kPa", "MeanEffectiveStress"]

 )

  q_col = selectbox_with_auto(

    "Deviatoric stress q", cols,

    ["DeviatorStress_kPa", "DeviatoricStress_kPa", "Deviator stress", "Deviatoric stress", "q", "q_kPa", "sigma_d", "σd"]

 )

  s1_col = selectbox_with_auto("σ1", cols, ["sigma1", "sigma_1", "σ1", "MajorPrincipalStress", "Major principal stress"])

  s2_col = selectbox_with_auto("σ2", cols, ["sigma2", "sigma_2", "σ2", "IntermediatePrincipalStress"])

  s3_col = selectbox_with_auto("σ3", cols, ["sigma3", "sigma_3", "σ3", "ConfiningStress", "Confining stress"])

else:

  N_col = st.sidebar.selectbox("Loading cycles N", cols)

  measured_col = st.sidebar.selectbox("Measured permanent strain", ["None"] + cols)

  stage_col = st.sidebar.selectbox("Stage column for MS models", ["None"] + cols)

  epsr_col = st.sidebar.selectbox("Resilient strain εr", ["None"] + cols)

  S_col = st.sidebar.selectbox("Degree of saturation S (%)", ["None"] + cols)

  p_col = st.sidebar.selectbox("Mean stress p", ["None"] + cols)

  q_col = st.sidebar.selectbox("Deviatoric stress q", ["None"] + cols)

  s1_col = st.sidebar.selectbox("σ1", ["None"] + cols)

  s2_col = st.sidebar.selectbox("σ2", ["None"] + cols)

  s3_col = st.sidebar.selectbox("σ3", ["None"] + cols)


if N_col == "None":

  st.error("The loading cycle column N could not be identified. Please select it manually.")

  st.stop()


original_rows = len(df)

df, removed_zero_rows = apply_zero_cycle_filter(df, N_col, remove_zero_N)

if removed_zero_rows > 0:

  st.info(f"Automatically removed {removed_zero_rows} row(s) where {N_col} = 0.")

cols = list(df.columns)

n = len(df)


required = required_inputs_for_models(selected_models)

missing_messages = []

if "stage" in required and stage_col == "None":

  missing_messages.append("stage column for multi-stage models")

if "eps_r" in required and epsr_col == "None":

  missing_messages.append("resilient strain εr")

if "S" in required and S_col == "None":

  missing_messages.append("degree of saturation S (%)")

if ("p_or_sigma" in required or "q_or_sigma" in required):

  has_pq = p_col != "None" and q_col != "None"

  has_principal = all(c != "None" for c in [s1_col, s2_col, s3_col])

  if not (has_pq or has_principal):

    missing_messages.append("p and q columns, or σ1/σ2/σ3 columns")

if missing_messages:

  st.warning("Missing required mapped input(s) for at least one selected model: " + "; ".join(missing_messages) + ".")


N = to_numeric_safe(df[N_col]).to_numpy(dtype=float)

N = np.maximum(N, 1e-12)


y_true = safe_array(df, measured_col, n) if measured_col != "None" else None

stage = df[stage_col].to_numpy() if stage_col != "None" else None

er = safe_array(df, epsr_col, n) if epsr_col != "None" else None

S = safe_array(df, S_col, n) if S_col != "None" else None


p = safe_array(df, p_col, n) if p_col != "None" else None

q = safe_array(df, q_col, n) if q_col != "None" else None


if (p is None or q is None) and all(c != "None" for c in [s1_col, s2_col, s3_col]):

  s1 = safe_array(df, s1_col, n)

  s2 = safe_array(df, s2_col, n)

  s3 = safe_array(df, s3_col, n)

  if p is None:

    p = mean_stress(s1, s2, s3)

  if q is None:

    q = deviator_stress(s1, s2, s3)


st.sidebar.header("General settings")

pa = st.sidebar.number_input("Reference pressure pa / σatm", value=100.0, step=10.0, format="%.6f")

default_alpha_sf = st.sidebar.number_input("Default α for Sf if not calibrated", value=1.0, step=0.05, format="%.6f")


with st.expander("Detected column mapping and filtered data", expanded=False):

  mapping_df = pd.DataFrame([

    {"Input": "N", "Column": N_col},

    {"Input": "Measured permanent strain", "Column": measured_col},

    {"Input": "Stage", "Column": stage_col},

    {"Input": "εr", "Column": epsr_col},

    {"Input": "S (%)", "Column": S_col},

    {"Input": "p", "Column": p_col},

    {"Input": "q", "Column": q_col},

    {"Input": "σ1", "Column": s1_col},

    {"Input": "σ2", "Column": s2_col},

    {"Input": "σ3", "Column": s3_col},

  ])

  st.dataframe(mapping_df, use_container_width=True)

  st.write(f"Rows used after filtering: {len(df)} out of {original_rows}.")

  st.dataframe(df.head(50), use_container_width=True)


Sf_default = None

if p is not None and q is not None:

  Sf_default = (q / pa) / ((p / pa) ** default_alpha_sf)


if mode == "Automatic calibration from measured data" and y_true is None:

  st.error("Automatic calibration requires a measured permanent strain column.")

  st.stop()


st.subheader("Known/fixed variables for selected models")

st.caption("These are not always calibrated. For several models, strength or stress-path variables must be supplied as columns or constants.")


fixed_inputs = {}

with st.expander("Set fixed variables/columns", expanded=True):

  for model in selected_models:

    st.markdown(f"#### {model}")

    st.caption(PARAMETER_NOTES[model])

    c1, c2, c3, c4 = st.columns(4)


    f = {}

    if model in ["Korkiala–Tanttu (2005)", "Korkiala–Tanttu (2005) [MS]", "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]"]:

      qf_default = find_best_column(cols, ["qf", "q_f", "FailureDeviatorStress", "DeviatoricStressFailure", "q_failure"])

      f["qf_col"] = c1.selectbox(f"{model}: qf column", ["None"] + cols, index=(["None"] + cols).index(qf_default) if qf_default in cols else 0, key=f"{model}_qfcol")

      f["qf_const"] = c2.number_input(f"{model}: qf constant", value=300.0, format="%.8f", key=f"{model}_qfconst")

      f["qf"] = constant_or_column(df, f["qf_col"], f["qf_const"])


    elif model in ["Gidel et al. (2001)"]:

      pmax_default = find_best_column(cols, ["pmax", "p_max", "MaximumMeanStress", "MaxMeanStress"])

      qmax_default = find_best_column(cols, ["qmax", "q_max", "MaximumDeviatorStress", "MaxDeviatorStress"])

      f["pmax_col"] = c1.selectbox(f"{model}: pmax column", ["None"] + cols, index=(["None"] + cols).index(pmax_default) if pmax_default in cols else 0, key=f"{model}_pmaxcol")

      f["qmax_col"] = c2.selectbox(f"{model}: qmax column", ["None"] + cols, index=(["None"] + cols).index(qmax_default) if qmax_default in cols else 0, key=f"{model}_qmaxcol")

      f["pmax_const"] = c3.number_input(f"{model}: pmax constant", value=100.0, format="%.8f", key=f"{model}_pmaxconst")

      f["qmax_const"] = c4.number_input(f"{model}: qmax constant", value=80.0, format="%.8f", key=f"{model}_qmaxconst")

      f["m"] = st.number_input(f"{model}: Mohr–Coulomb m", value=1.5, format="%.8f", key=f"{model}_m")

      f["s"] = st.number_input(f"{model}: Mohr–Coulomb s", value=10.0, format="%.8f", key=f"{model}_s")

      f["pmax"] = constant_or_column(df, f["pmax_col"], f["pmax_const"])

      f["qmax"] = constant_or_column(df, f["qmax_col"], f["qmax_const"])


    elif model == "Chen et al. (2014)":

      st.info("Chen et al. (2014) is the modified Gidel-based railway-subgrade model: εp = εp0(1-exp(-BN))(√(pam²+qam²)/pa)^α / [m(1+pini/pam)+s/pam-(qini+qam)/pam]. The original paper used it to calculate layer strain and then settlement as s(N)=ΣH_iεp,i.")

      pam_default = find_best_column(cols, ["pam", "p_am", "MeanStressAmplitude", "MeanStressAmp", "Mean stress amplitude", "DynamicMeanStressAmplitude"])

      qam_default = find_best_column(cols, ["qam", "q_am", "DeviatorStressAmplitude", "DeviatoricStressAmplitude", "DeviatorStressAmp", "Deviator stress amplitude", "DynamicDeviatorStressAmplitude"])

      pini_default = find_best_column(cols, ["pini", "p_ini", "InitialMeanStress", "Initial mean stress"])

      qini_default = find_best_column(cols, ["qini", "q_ini", "InitialDeviatorStress", "Initial deviator stress"])

      f["pam_col"] = c1.selectbox(f"{model}: pam column", ["None"] + cols, index=(["None"] + cols).index(pam_default) if pam_default in cols else 0, key=f"{model}_pamcol")

      f["qam_col"] = c2.selectbox(f"{model}: qam column", ["None"] + cols, index=(["None"] + cols).index(qam_default) if qam_default in cols else 0, key=f"{model}_qamcol")

      f["pini_col"] = c3.selectbox(f"{model}: pini column", ["None"] + cols, index=(["None"] + cols).index(pini_default) if pini_default in cols else 0, key=f"{model}_pinicol")

      f["qini_col"] = c4.selectbox(f"{model}: qini column", ["None"] + cols, index=(["None"] + cols).index(qini_default) if qini_default in cols else 0, key=f"{model}_qinicol")

      f["pam_const"] = st.number_input(f"{model}: pam constant", value=60.0, format="%.8f", key=f"{model}_pamconst")

      f["qam_const"] = st.number_input(f"{model}: qam constant", value=24.0, format="%.8f", key=f"{model}_qamconst")

      f["pini_const"] = st.number_input(f"{model}: pini constant", value=60.0, format="%.8f", key=f"{model}_piniconst")

      f["qini_const"] = st.number_input(f"{model}: qini constant", value=0.0, format="%.8f", key=f"{model}_qiniconst")

      f["m"] = st.number_input(f"{model}: Mohr–Coulomb m", value=1.5, format="%.8f", key=f"{model}_m")

      f["s"] = st.number_input(f"{model}: Mohr–Coulomb s", value=10.0, format="%.8f", key=f"{model}_s")

      f["pam"] = constant_or_column(df, f["pam_col"], f["pam_const"])

      f["qam"] = constant_or_column(df, f["qam_col"], f["qam_const"])

      f["pini"] = constant_or_column(df, f["pini_col"], f["pini_const"])

      f["qini"] = constant_or_column(df, f["qini_col"], f["qini_const"])


    elif model == "MEPDG / NCHRP (2003)":


      st.warning("MEPDG/NCHRP is primarily a design/single-stage formulation. For multistage RLT, the uploaded Erlingsson & Rahman (2013) paper extends the Tseng & Lytton form using time-hardening; direct MEPDG/NCHRP multistage use should be treated as an approximation.")

      epsv_default = find_best_column(cols, ["eps_v", "epsilon_v", "VerticalResilientStrain", "Vertical resilient strain", "ev"])

      f["epsv_col"] = c1.selectbox(f"{model}: εv column", ["None"] + cols, index=(["None"] + cols).index(epsv_default) if epsv_default in cols else 0, key=f"{model}_epsvcol")

      f["epsv_const"] = c2.number_input(f"{model}: εv constant", value=0.001, format="%.10f", key=f"{model}_epsvconst")

      f["ev"] = constant_or_column(df, f["epsv_col"], f["epsv_const"])

      f["use_w_beta"] = st.checkbox(f"{model}: estimate β from moisture content w", value=False, key=f"{model}_use_w_beta")

      if f["use_w_beta"]:

        w_default = find_best_column(cols, ["w", "w (%)", "Moisture", "MoistureContent", "Moisture content", "WaterContent"])

        f["w_col"] = c3.selectbox(f"{model}: w column (%)", ["None"] + cols, index=(["None"] + cols).index(w_default) if w_default in cols else 0, key=f"{model}_wcol")

        f["w_const"] = c4.number_input(f"{model}: w constant (%)", value=5.0, format="%.8f", key=f"{model}_wconst")

        f["w"] = constant_or_column(df, f["w_col"], f["w_const"])

      else:

        f["w"] = None


    elif model == "Chow et al. (2014)":

      sigma_d_default = find_best_column(cols, ["sigma_d", "σd", "DeviatorStress_kPa", "DeviatoricStress_kPa", "Deviatoric stress"])

      tau_f_default = find_best_column(cols, ["tau_f", "τf", "ShearStress", "Shear stress"])

      tau_max_default = find_best_column(cols, ["tau_max", "τmax", "ShearStrength", "MaximumShearStress"])

      f["sigma_d_col"] = c1.selectbox(f"{model}: σd column", ["None"] + cols, index=(["None"] + cols).index(sigma_d_default) if sigma_d_default in cols else 0, key=f"{model}_sdcol")

      f["tau_f_col"] = c2.selectbox(f"{model}: τf column", ["None"] + cols, index=(["None"] + cols).index(tau_f_default) if tau_f_default in cols else 0, key=f"{model}_tfcol")

      f["tau_max_col"] = c3.selectbox(f"{model}: τmax column", ["None"] + cols, index=(["None"] + cols).index(tau_max_default) if tau_max_default in cols else 0, key=f"{model}_tmcol")

      f["sigma_d_const"] = st.number_input(f"{model}: σd constant", value=50.0, format="%.8f", key=f"{model}_sdconst")

      f["tau_f_const"] = st.number_input(f"{model}: τf constant", value=40.0, format="%.8f", key=f"{model}_tfconst")

      f["tau_max_const"] = st.number_input(f"{model}: τmax constant", value=100.0, format="%.8f", key=f"{model}_tmconst")

      f["sigma_d"] = constant_or_column(df, f["sigma_d_col"], f["sigma_d_const"])

      f["tau_f"] = constant_or_column(df, f["tau_f_col"], f["tau_f_const"])

      f["tau_max"] = constant_or_column(df, f["tau_max_col"], f["tau_max_const"])


    elif model == "Hyde (1974)":

      sigma3_default = find_best_column(cols, ["sigma3", "sigma_3", "σ3", "ConfiningStress", "Confining stress"])

      f["sigma3_col"] = c1.selectbox(f"{model}: σ3 column", ["None"] + cols, index=(["None"] + cols).index(sigma3_default) if sigma3_default in cols else 0, key=f"{model}_sigma3col")

      f["sigma3_const"] = c2.number_input(f"{model}: σ3 constant", value=50.0, format="%.8f", key=f"{model}_sigma3const")

      f["sigma3"] = constant_or_column(df, f["sigma3_col"], f["sigma3_const"])


    elif model == "Lentz & Baladi (1980)":

      sd_default = find_best_column(cols, ["Sd", "S_d", "DynamicStrength", "CyclicStrength"])

      eps_default = find_best_column(cols, ["eps095Sd", "epsilon095Sd", "eps_0.95Sd", "eps0_95Sd"])

      sigd_default = find_best_column(cols, ["sigma_d", "σd", "DeviatorStress_kPa", "DeviatoricStress_kPa", "Deviatoric stress"])

      f["sigma_d_col"] = c1.selectbox(f"{model}: σd column", ["None"] + cols, index=(["None"] + cols).index(sigd_default) if sigd_default in cols else 0, key=f"{model}_sdcol")

      f["Sd_col"] = c2.selectbox(f"{model}: Sd column", ["None"] + cols, index=(["None"] + cols).index(sd_default) if sd_default in cols else 0, key=f"{model}_Sdcol")

      f["eps095Sd_col"] = c3.selectbox(f"{model}: ε0.95Sd column", ["None"] + cols, index=(["None"] + cols).index(eps_default) if eps_default in cols else 0, key=f"{model}_eps095col")

      f["sigma_d_const"] = st.number_input(f"{model}: σd constant", value=50.0, format="%.8f", key=f"{model}_sdconst")

      f["Sd_const"] = st.number_input(f"{model}: Sd constant", value=100.0, format="%.8f", key=f"{model}_Sdconst")

      f["eps095Sd_const"] = st.number_input(f"{model}: ε0.95Sd constant", value=1.0, format="%.8f", key=f"{model}_eps095const")

      f["sigma_d"] = constant_or_column(df, f["sigma_d_col"], f["sigma_d_const"])

      f["Sd"] = constant_or_column(df, f["Sd_col"], f["Sd_const"])

      f["eps095Sd"] = constant_or_column(df, f["eps095Sd_col"], f["eps095Sd_const"])


    elif model == "Khedr (1985)":

      ro_default = find_best_column(cols, ["Ro", "R_o", "OctahedralStressRatio", "octahedral stress ratio"])

      mr_default = find_best_column(cols, ["MR", "Mr", "ResilientModulus", "Resilient modulus"])

      f["Ro_col"] = c1.selectbox(f"{model}: Ro column", ["None"] + cols, index=(["None"] + cols).index(ro_default) if ro_default in cols else 0, key=f"{model}_rocol")

      f["MR_col"] = c2.selectbox(f"{model}: MR column", ["None"] + cols, index=(["None"] + cols).index(mr_default) if mr_default in cols else 0, key=f"{model}_mrcol")

      f["Ro_const"] = c3.number_input(f"{model}: Ro constant", value=0.5, format="%.8f", key=f"{model}_roconst")

      f["MR_const"] = c4.number_input(f"{model}: MR constant", value=100.0, format="%.8f", key=f"{model}_mrconst")

      f["Ro"] = constant_or_column(df, f["Ro_col"], f["Ro_const"])

      f["MR"] = constant_or_column(df, f["MR_col"], f["MR_const"])


    elif model == "Lin et al. (2019)":

      sigmax_default = find_best_column(cols, ["sigma_max", "σmax", "MaximumStress", "MaxStress"])

      tau_f_default = find_best_column(cols, ["tau_f", "τf", "ShearStrength", "PeakShearStrength"])

      tau_max_default = find_best_column(cols, ["tau_max", "τmax", "MaximumShearStress", "MaxShearStress"])

      f["sigma_max_col"] = c1.selectbox(f"{model}: σmax column", ["None"] + cols, index=(["None"] + cols).index(sigmax_default) if sigmax_default in cols else 0, key=f"{model}_sigmaxcol")

      f["tau_f_col"] = c2.selectbox(f"{model}: τf column", ["None"] + cols, index=(["None"] + cols).index(tau_f_default) if tau_f_default in cols else 0, key=f"{model}_taufcol")

      f["tau_max_col"] = c3.selectbox(f"{model}: τmax column", ["None"] + cols, index=(["None"] + cols).index(tau_max_default) if tau_max_default in cols else 0, key=f"{model}_taumaxcol")

      f["sigma_max_const"] = st.number_input(f"{model}: σmax constant", value=100.0, format="%.8f", key=f"{model}_sigmaxconst")

      f["tau_f_const"] = st.number_input(f"{model}: τf constant", value=100.0, format="%.8f", key=f"{model}_taufconst")

      f["tau_max_const"] = st.number_input(f"{model}: τmax constant", value=50.0, format="%.8f", key=f"{model}_taumaxconst")

      f["sigma_max"] = constant_or_column(df, f["sigma_max_col"], f["sigma_max_const"])

      f["tau_f"] = constant_or_column(df, f["tau_f_col"], f["tau_f_const"])

      f["tau_max"] = constant_or_column(df, f["tau_max_col"], f["tau_max_const"])


    elif model == "Ooi (2021)":

      ssr_default = find_best_column(cols, ["SSR", "ShearStressRatio", "Shear stress ratio", "stressstrengthratio"])

      f["SSR_col"] = c1.selectbox(f"{model}: SSR column", ["None"] + cols, index=(["None"] + cols).index(ssr_default) if ssr_default in cols else 0, key=f"{model}_ssrcol")

      f["SSR_const"] = c2.number_input(f"{model}: SSR constant", value=0.5, format="%.8f", key=f"{model}_ssrconst")

      f["SSR"] = constant_or_column(df, f["SSR_col"], f["SSR_const"])


    fixed_inputs[model] = f


manual_params = {}

if mode == "Prediction with known parameters":

  st.subheader("Model parameters")

  st.caption("Enter calibrated parameters. Default values are placeholders and should not be used for final design.")

  with st.expander("Parameter input panel", expanded=True):

    for model in selected_models:

      st.markdown(f"#### {model}")

      c1, c2, c3, c4 = st.columns(4)

      pms = {}

      if model == "Barksdale (1972)":

        pms["a"] = c1.number_input(f"{model}: a", value=0.0, format="%.10f")

        pms["b"] = c2.number_input(f"{model}: b", value=0.01, format="%.10f")


      elif model == "Hyde (1974)":

        pms["a"] = c1.number_input(f"{model}: a", value=0.01, format="%.10f")


      elif model == "Veverka (1979)":

        pms["mu"] = c1.number_input(f"{model}: μ", value=1.0, format="%.10f")

        pms["alpha"] = c2.number_input(f"{model}: α", value=0.20, format="%.10f")


      elif model == "Lentz & Baladi (1980)":

        pms["n"] = c1.number_input(f"{model}: n", value=1.0, format="%.10f")

        pms["m"] = c2.number_input(f"{model}: m", value=0.1, format="%.10f")


      elif model == "Khedr (1985)":

        pms["s1"] = c1.number_input(f"{model}: s1", value=1e-6, format="%.12f")

        pms["s2"] = c2.number_input(f"{model}: s2", value=1.0, format="%.10f")

        pms["s3"] = c3.number_input(f"{model}: s3", value=1.0, format="%.10f")

        pms["m"] = c4.number_input(f"{model}: m", value=0.2, format="%.10f")


      elif model == "Paute et al. (1988)":

        pms["A"] = c1.number_input(f"{model}: A", value=0.01, format="%.10f")

        pms["D"] = c2.number_input(f"{model}: D", value=10.0, format="%.10f")

        pms["epsp0"] = c3.number_input(f"{model}: εp0", value=0.0, format="%.10f")


      elif model == "Sweere (1990)":

        pms["a"] = c1.number_input(f"{model}: a", value=-4.0, format="%.10f")

        pms["b"] = c2.number_input(f"{model}: b", value=0.20, format="%.10f")


      elif model == "Hornych et al. (1993)":

        pms["epsp100"] = c1.number_input(f"{model}: εp(100)", value=0.0, format="%.10f")

        pms["A"] = c2.number_input(f"{model}: A", value=0.01, format="%.10f")

        pms["B"] = c3.number_input(f"{model}: B", value=0.20, format="%.10f")


      elif model == "Wolff & Visser (1994)":

        pms["a"] = c1.number_input(f"{model}: a", value=0.01, format="%.10f")

        pms["b"] = c2.number_input(f"{model}: b", value=0.001, format="%.10f")

        pms["c"] = c3.number_input(f"{model}: c", value=1e-8, format="%.12f")


      elif model in ["Rahman et al. (2023)", "Rahman et al. (2023) [MS]"]:

        pms["a"] = c1.number_input(f"{model}: a", value=1.0, format="%.10f")

        pms["b"] = c2.number_input(f"{model}: b", value=250.0, format="%.10f")


      elif model in ["Rahman et al. (2023) – saturation-based extension", "Rahman et al. (2023) – saturation-based extension [MS]"]:

        st.info("For this model, a is calculated as a = c1·S + c2. To identify c1 and c2 separately, the dataset must contain at least two different saturation levels. If S is constant, the app fits an equivalent a and sets c1=0, c2=a.")

        pms["b"] = c1.number_input(f"{model}: b", value=250.0, format="%.10f")

        pms["c1"] = c2.number_input(f"{model}: c1", value=0.05, format="%.10f")

        pms["c2"] = c3.number_input(f"{model}: c2", value=2.0, format="%.10f")


      elif model in ["Rahman & Erlingsson (2015)", "Rahman & Erlingsson (2015) [MS]"]:

        pms["a"] = c1.number_input(f"{model}: a", value=1.0, format="%.10f")

        pms["b"] = c2.number_input(f"{model}: b", value=1.0, format="%.10f")

        pms["alpha"] = c3.number_input(f"{model}: α", value=default_alpha_sf, format="%.10f")


      elif model in ["Korkiala–Tanttu (2005)", "Korkiala–Tanttu (2005) [MS]"]:

        pms["C"] = c1.number_input(f"{model}: C", value=1e-5, format="%.12f")

        pms["b"] = c2.number_input(f"{model}: b", value=0.20, format="%.10f")

        pms["A"] = c3.number_input(f"{model}: A", value=1.05, format="%.10f")


      elif model == "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]":

        pms["C"] = c1.number_input(f"{model}: C", value=1e-5, format="%.12f")

        pms["c"] = c2.number_input(f"{model}: c", value=0.20, format="%.10f")

        pms["d"] = c3.number_input(f"{model}: d", value=0.10, format="%.10f")

        pms["A"] = c4.number_input(f"{model}: A", value=1.05, format="%.10f")


      elif model == "Huurman (1997)":

        pms["A"] = c1.number_input(f"{model}: A", value=1e-5, format="%.12f")

        pms["B"] = c2.number_input(f"{model}: B", value=0.20, format="%.10f")

        pms["C"] = c3.number_input(f"{model}: C", value=1e-6, format="%.12f")

        pms["D"] = c4.number_input(f"{model}: D", value=0.001, format="%.12f")


      elif model in ["Gidel et al. (2001)"]:

        pms["eps0"] = c1.number_input(f"{model}: ε0", value=1e-4, format="%.12f")

        pms["B"] = c2.number_input(f"{model}: B", value=0.20, format="%.10f")

        pms["n"] = c3.number_input(f"{model}: n", value=1.0, format="%.10f")


      elif model == "Chen et al. (2014)":

        pms["epsp0"] = c1.number_input(f"{model}: εp0", value=1e-4, format="%.12f")

        pms["B"] = c2.number_input(f"{model}: B", value=0.001, format="%.12f")

        pms["alpha"] = c3.number_input(f"{model}: α", value=0.3, format="%.10f")


      elif model in ["Tseng & Lytton (1989)", "Tseng & Lytton (1989) [MS]"]:

        pms["eps0"] = c1.number_input(f"{model}: ε0", value=1.0, format="%.10f")

        pms["rho"] = c2.number_input(f"{model}: ρ", value=1000.0, format="%.10f")

        pms["beta"] = c3.number_input(f"{model}: β", value=0.20, format="%.10f")


      elif model == "MEPDG / NCHRP (2003)":

        pms["beta_s"] = c1.number_input(f"{model}: βs", value=1.673, format="%.10f")

        pms["eps0"] = c2.number_input(f"{model}: ε0", value=1.0, format="%.10f")

        pms["rho"] = c3.number_input(f"{model}: ρ", value=1000.0, format="%.10f")

        pms["beta"] = c4.number_input(f"{model}: β", value=0.20, format="%.10f")


      elif model == "Chow et al. (2014)":

        pms["A"] = c1.number_input(f"{model}: A", value=1e-6, format="%.12f")

        pms["B"] = c2.number_input(f"{model}: B", value=0.20, format="%.10f")

        pms["C"] = c3.number_input(f"{model}: C", value=1.0, format="%.10f")

        pms["D"] = c4.number_input(f"{model}: D", value=1.0, format="%.10f")


      elif model == "Lin et al. (2019)":

        pms["A"] = c1.number_input(f"{model}: A", value=1e-6, format="%.12f")

        pms["B"] = c2.number_input(f"{model}: B", value=0.20, format="%.10f")

        pms["C"] = c3.number_input(f"{model}: C", value=1.0, format="%.10f")

        pms["D"] = c4.number_input(f"{model}: D", value=1.0, format="%.10f")

        pms["E"] = st.number_input(f"{model}: E", value=0.1, format="%.10f")


      elif model == "Ooi (2021)":

        pms["a"] = c1.number_input(f"{model}: a", value=1e-6, format="%.12f")

        pms["b"] = c2.number_input(f"{model}: b", value=1.0, format="%.10f")

        pms["c"] = c3.number_input(f"{model}: c", value=0.1, format="%.10f")

        pms["d"] = c4.number_input(f"{model}: d", value=0.1, format="%.10f")

      manual_params[model] = pms


run = st.button("Run analysis", type="primary")


if not run:

  st.stop()


preds = {}

model_params = {}

status = {}


for model in selected_models:

  try:

    if mode == "Automatic calibration from measured data":

      params, pred, stat = fit_model(model, df, N, y_true, p, q, er, S, Sf_default, stage, pa, fixed_inputs.get(model, {}))

      preds[model] = pred

      model_params[model] = params

      status[model] = stat

      continue


    pms = manual_params[model]

    fixed = fixed_inputs.get(model, {})


    if model == "Barksdale (1972)":

      pred = barksdale_1972(N, pms["a"], pms["b"])


    elif model == "Hyde (1974)":

      if q is None or fixed.get("sigma3") is None: raise ValueError("q and σ3 are required.")

      pred = hyde_1974(pms["a"], q, fixed["sigma3"])


    elif model == "Veverka (1979)":

      if er is None: raise ValueError("εr is required.")

      pred = veverka_1979(N, pms["mu"], pms["alpha"], er)


    elif model == "Lentz & Baladi (1980)":

      pred = lentz_baladi_1980(fixed["sigma_d"], fixed["Sd"], fixed["eps095Sd"], pms["n"], pms["m"])


    elif model == "Khedr (1985)":

      pred = khedr_1985(N, pms["s1"], pms["s2"], pms["s3"], pms["m"], fixed["Ro"], fixed["MR"])


    elif model == "Paute et al. (1988)":

      pred = paute_1988(N, pms["A"], pms["D"], pms["epsp0"])


    elif model == "Sweere (1990)":

      pred = sweere_1990(N, pms["a"], pms["b"])


    elif model == "Hornych et al. (1993)":

      pred = hornych_1993(N, pms["epsp100"], pms["A"], pms["B"])


    elif model == "Wolff & Visser (1994)":

      pred = wolff_visser_1994(N, pms["a"], pms["b"], pms["c"])


    elif model == "Rahman et al. (2023)":

      if er is None: raise ValueError("εr is required.")

      pred = rahman_et_al_2023_base(N, pms["a"], pms["b"], er)


    elif model == "Rahman et al. (2023) [MS]":

      if er is None or stage is None: raise ValueError("εr and stage are required.")

      pred = rahman_et_al_2023_base_ms(N, stage, pms["a"], pms["b"], er)


    elif model == "Rahman et al. (2023) – saturation-based extension":

      if er is None or S is None: raise ValueError("εr and S (%) are required.")

      pred = rahman_et_al_2023_er_moisture(N, pms["b"], pms["c1"], pms["c2"], S, er)


    elif model == "Rahman et al. (2023) – saturation-based extension [MS]":

      if er is None or S is None or stage is None: raise ValueError("εr, S (%) and stage are required.")

      pred = rahman_et_al_2023_er_moisture_ms(N, stage, pms["b"], pms["c1"], pms["c2"], S, er)


    elif model == "Rahman & Erlingsson (2015)":

      if p is None or q is None: raise ValueError("p and q are required.")

      Sf = (q / pa) / ((p / pa) ** pms["alpha"])

      pred = rahman_erlingsson_2015_sf(N, pms["a"], pms["b"], Sf)


    elif model == "Rahman & Erlingsson (2015) [MS]":

      if p is None or q is None or stage is None: raise ValueError("p, q and stage are required.")

      Sf = (q / pa) / ((p / pa) ** pms["alpha"])

      pred = rahman_erlingsson_2015_sf_ms(N, stage, pms["a"], pms["b"], Sf)


    elif model == "Korkiala–Tanttu (2005)":

      if q is None: raise ValueError("q is required.")

      pred = korkiala_tanttu_2005(N, pms["C"], pms["b"], pms["A"], q, fixed["qf"])


    elif model == "Korkiala–Tanttu (2005) [MS]":

      if q is None or stage is None: raise ValueError("q and stage are required.")

      pred = korkiala_tanttu_2005_ms(N, stage, pms["C"], pms["b"], pms["A"], q, fixed["qf"])


    elif model == "Erlingsson & Rahman (2013) [Modified Korkiala–Tanttu] [MS]":

      if q is None or stage is None: raise ValueError("q and stage are required.")

      pred = modified_korkiala_tanttu_2013_ms(N, stage, pms["C"], pms["c"], pms["d"], pms["A"], q, fixed["qf"])


    elif model == "Huurman (1997)":

      pred = huurman_1997(N, pms["A"], pms["B"], pms["C"], pms["D"])


    elif model == "Gidel et al. (2001)":

      pred = gidel_2001(N, pms["eps0"], pms["B"], pms["n"], fixed["pmax"], fixed["qmax"], pa, fixed["m"], fixed["s"])


    elif model == "Chen et al. (2014)":

      denom = fixed["m"] * (1.0 + fixed["pini"] / np.maximum(fixed["pam"], 1e-12)) + fixed["s"] / np.maximum(fixed["pam"], 1e-12) - (fixed["qini"] + fixed["qam"]) / np.maximum(fixed["pam"], 1e-12)

      if np.any(~np.isfinite(denom)) or np.any(denom <= 0):

        raise ValueError("Chen et al. (2014): denominator must be positive: m(1+pini/pam)+s/pam-(qini+qam)/pam > 0.")

      pred = chen_2014(N, pms["epsp0"], pms["B"], pms["alpha"], fixed["pam"], fixed["qam"], pa, fixed["m"], fixed["s"], fixed["pini"], fixed["qini"])


    elif model == "Tseng & Lytton (1989)":

      if er is None:

        raise ValueError("εr is required.")

      pred = tseng_lytton_1989(N, pms["eps0"], pms["rho"], pms["beta"], er)


    elif model == "Tseng & Lytton (1989) [MS]":

      if er is None or stage is None:

        raise ValueError("εr and stage are required.")

      pred = tseng_lytton_1989_ms(N, stage, pms["eps0"], pms["rho"], pms["beta"], er)


    elif model == "MEPDG / NCHRP (2003)":

      if er is None:

        raise ValueError("εr is required.")

      if fixed.get("ev") is None:

        raise ValueError("εv is required.")

      beta_val = pms["beta"]

      if fixed.get("use_w_beta") and fixed.get("w") is not None:

        beta_val = beta_from_moisture_mepdg(fixed["w"])

      pred = mepdg_nchrp_2003(N, pms["beta_s"], pms["eps0"], pms["rho"], beta_val, er, fixed["ev"])


    elif model == "Chow et al. (2014)":

      pred = chow_2014(N, pms["A"], pms["B"], pms["C"], pms["D"], fixed["sigma_d"], fixed["tau_f"], fixed["tau_max"])


    elif model == "Lin et al. (2019)":

      pred = lin_2019(N, pms["A"], pms["B"], pms["C"], pms["D"], pms["E"], fixed["sigma_max"], pa, fixed["tau_f"], fixed["tau_max"])


    elif model == "Ooi (2021)":

      pred = ooi_2021(N, pms["a"], pms["b"], pms["c"], pms["d"], fixed["SSR"])


    preds[model] = pred

    model_params[model] = {**pms, **{k: v for k, v in fixed.items() if isinstance(v, (int, float, str))}}

    status[model] = "predicted using known parameters"


  except Exception as e:

    preds[model] = np.full_like(N, np.nan, dtype=float)

    model_params[model] = {}

    status[model] = f"failed: {e}"


for model, stat in status.items():

  if stat.startswith("failed") or stat.startswith("calibration failed"):

    st.warning(f"{model}: {stat}")


valid_preds = {k: v for k, v in preds.items() if np.isfinite(v).any()}

if not valid_preds:

  st.error("No valid predictions were produced. Check the required columns and parameter values.")

  st.stop()


apply_plot_style()

x_plot = N

x_label = "Number of loading cycles, N [-]"


out = df.copy()

for name, pred in valid_preds.items():

  out[f"pred_{name}"] = pred

  if y_true is not None:

    out[f"error_{name}"] = y_true - pred

    out[f"abs_error_{name}"] = np.abs(y_true - pred)


st.subheader("Prediction results")

st.dataframe(out, use_container_width=True)


params_rows = []

for model in valid_preds:

  row = {"Model": clean_model_label(model), "Status": status[model]}

  for k, v in model_params.get(model, {}).items():

    if isinstance(v, (int, float, str)):

      row[k] = v

  params_rows.append(row)

params_df = pd.DataFrame(params_rows)


st.subheader("Model parameters used / calibrated")

st.dataframe(params_df, use_container_width=True)


metrics_df = None

if y_true is not None:

  rows = []

  for name, pred in valid_preds.items():

    rows.append({"Model": clean_model_label(name), **metrics(y_true, pred)})

  metrics_df = pd.DataFrame(rows)

  st.subheader("Statistical metrics")

  st.dataframe(metrics_df, use_container_width=True)


st.subheader("Plots")


n_models = len(valid_preds)

fig, ax = plt.subplots(figsize=(10.5, 6.3))

if y_true is not None:

  ax.scatter(

        x_plot, y_true,

        label="Measured",

        marker="o",

        s=20,

        facecolors="none",

        edgecolors="#6E6E6E",

        linewidths=0.65,

        alpha=0.55,

        zorder=2

    )


for idx, (name, pred) in enumerate(valid_preds.items()):


  ax.plot(

    x_plot, pred,

    label=f"Predicted – {clean_model_label(name)}",

    color=model_color(idx, n_models),

    linewidth=1.8,

    linestyle=model_linestyle(idx, n_models),

    alpha=0.98,

    zorder=3

 )


ax.set_xlabel(x_label)

ax.set_ylabel(f"Permanent strain ({strain_unit})")

ax.set_title("Measured and predicted permanent strain over loading cycles")

ax.grid(True, alpha=0.18)

ax.legend(loc="best", frameon=True)

fig.tight_layout()

show_plot_with_download(fig, "permanent_strain_vs_cycles.png")


if y_true is not None:

  for idx, (name, pred) in enumerate(valid_preds.items()):

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    ax.scatter(

            y_true, pred,

            label=f"Predicted – {clean_model_label(name)}",

            color=model_color(idx, n_models),

            edgecolors="black",

            linewidths=0.35,

            s=28,

            alpha=0.75

        )

    finite = np.isfinite(y_true) & np.isfinite(pred)

    if finite.any():

      mn = np.nanmin([np.nanmin(y_true[finite]), np.nanmin(pred[finite])])

      mx = np.nanmax([np.nanmax(y_true[finite]), np.nanmax(pred[finite])])

      if mx > mn:

        ax.plot([mn, mx], [mn, mx], linestyle="--", color="black", linewidth=1.5, label="1:1 line")

        ax.plot([mn, mx], [0.8*mn, 0.8*mx], linestyle=":", color="gray", linewidth=1.3, label="±20%")

        ax.plot([mn, mx], [1.2*mn, 1.2*mx], linestyle=":", color="gray", linewidth=1.3)

    ax.set_xlabel(f"Measured permanent strain ({strain_unit})")

    ax.set_ylabel(f"Predicted permanent strain ({strain_unit})")

    ax.set_title(f"Predicted vs measured: {clean_model_label(name)}")

    ax.grid(True, alpha=0.18)

    ax.legend(loc="best", frameon=True)

    add_metric_text(ax, metrics(y_true, pred))

    fig.tight_layout()

    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", clean_model_label(name)).strip("_")

    show_plot_with_download(fig, f"predicted_vs_measured_{safe_name}.png")


if y_true is not None:

  fig, ax = plt.subplots(figsize=(10.5, 6.3))

  for idx, (name, pred) in enumerate(valid_preds.items()):

    err = y_true - pred

    ax.plot(

      x_plot, err,

      marker="o",

      markersize=3.2,

      linestyle=model_linestyle(idx, n_models),

      label=clean_model_label(name),

      color=model_color(idx, n_models),

      linewidth=1.2,

      alpha=0.90

   )

  ax.axhline(0, linestyle="--", color="black", linewidth=1.2)

  ax.set_xlabel(x_label)

  ax.set_ylabel(f"Error = measured - predicted ({strain_unit})")

  ax.set_title("Prediction error over loading cycles")

  ax.grid(True, alpha=0.18)

  ax.legend(loc="best", frameon=True)

  fig.tight_layout()

  show_plot_with_download(fig, "prediction_error_vs_cycles.png")


if y_true is not None:

  fig, ax = plt.subplots(figsize=(10.5, 6.3))

  for idx, (name, pred) in enumerate(valid_preds.items()):

    err = np.abs(y_true - pred)

    ax.plot(

      x_plot, err,

      marker="o",

      markersize=3.2,

      linestyle=model_linestyle(idx, n_models),

      label=clean_model_label(name),

      color=model_color(idx, n_models),

      linewidth=1.2,

      alpha=0.90

   )

  ax.set_xlabel(x_label)

  ax.set_ylabel(f"Absolute error ({strain_unit})")

  ax.set_title("Absolute prediction error over loading cycles")

  ax.grid(True, alpha=0.18)

  ax.legend(loc="best", frameon=True)

  fig.tight_layout()

  show_plot_with_download(fig, "absolute_error_vs_cycles.png")


buffer = io.BytesIO()

with pd.ExcelWriter(buffer, engine="openpyxl") as writer:

  out.to_excel(writer, index=False, sheet_name="Predictions")

  params_df.to_excel(writer, index=False, sheet_name="Parameters")

  if metrics_df is not None:

    metrics_df.to_excel(writer, index=False, sheet_name="Metrics")

buffer.seek(0)


st.download_button(

  "Download predictions, parameters and metrics as Excel",

  data=buffer,

  file_name="ugm_permanent_strain_results.xlsx",

  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

)





if __name__ == "__main__":
  pass
