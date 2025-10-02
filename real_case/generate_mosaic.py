import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# CONFIG – update these paths
# -------------------
MANIFEST_CSV = Path(r"C:\VOW\data\Site_A\O\schGAN_sections\manifest_sections.csv")
COORDS_WITH_DIST_CSV = Path(r"C:\VOW\data\Site_A\O\schGAN_sections\coords_with_distances.csv")
GAN_DIR = Path(r"C:\VOW\res\Site_A\O")  # where the *_gan.csv files are
OUT_DIR = Path(r"C:\VOW\res\Site_A\O")  # where to save mosaic csv/png
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Section & image constants used earlier
N_COLS = 512
N_ROWS = 32

# Optional: global pixel size. If None, use the median section pixel size
GLOBAL_DX = None  # meters per pixel horizontally

# Top axis appearance: False -> 0..(W-1) px, True -> 0..32 normalized
TOP_AXIS_0_TO_32 = False

# -------------------
# LOAD
# -------------------
man = pd.read_csv(MANIFEST_CSV)
coords = pd.read_csv(COORDS_WITH_DIST_CSV)

required_m_cols = {"section_index", "span_m", "left_pad_m", "right_pad_m", "start_idx", "csv_path"}
missing = required_m_cols - set(man.columns)
if missing:
    raise ValueError(f"Manifest is missing columns: {missing}")

if "cum_along_m" not in coords.columns:
    raise ValueError("coords_with_distances.csv must have a 'cum_along_m' column")

# Ensure integer indices
man["section_index"] = man["section_index"].astype(int)
man["start_idx"] = man["start_idx"].astype(int)

# -------------------
# Locate GAN csv per section
# -------------------
def find_gan_csv(sec_index: int) -> Path | None:
    # Pattern based on earlier saving: schemaGAN_section_{sec:03d}_seed*_gan.csv
    cand = list(GAN_DIR.glob(f"schemaGAN_section_{sec_index:03d}_*_gan.csv"))
    if not cand:
        return None
    # If multiple seeds exist, pick the newest
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

man["gan_csv"] = man["section_index"].apply(find_gan_csv)
missing_gan = man[man["gan_csv"].isna()]
if not missing_gan.empty:
    print("[WARN] Missing GAN csv for sections:",
          missing_gan["section_index"].tolist())

# Drop sections w/o GAN to proceed
man = man.dropna(subset=["gan_csv"]).reset_index(drop=True)
if man.empty:
    raise RuntimeError("No sections with GAN CSVs found; cannot build mosaic.")

# -------------------
# Compute per-section mapping to global x
# x_global(j) = (cum_along at section-first CPT) - left_pad_m + j * (total_span / (N_COLS-1))
# -------------------
def section_mapping(row):
    total_span = float(row["span_m"] + row["left_pad_m"] + row["right_pad_m"])
    if total_span <= 0:
        raise ValueError(f"Non-positive total_span for section {row['section_index']}")
    start_idx = int(row["start_idx"])
    m0 = float(coords.loc[start_idx, "cum_along_m"])
    x0 = m0 - float(row["left_pad_m"])
    dx = total_span / (N_COLS - 1)
    return x0, dx, total_span

x0_list, dx_list = [], []
for _, r in man.iterrows():
    x0, dx, total_span = section_mapping(r)
    x0_list.append(x0)
    dx_list.append(dx)

man["x0"] = x0_list
man["dx"] = dx_list
man["x1"] = man["x0"] + (N_COLS - 1) * man["dx"]

# -------------------
# Decide global grid (meters)
# -------------------
XMIN = float(man["x0"].min())
XMAX = float(man["x1"].max())
if GLOBAL_DX is None:
    GLOBAL_DX = float(np.median(man["dx"]))  # robust choice across sections
W = int(round((XMAX - XMIN) / GLOBAL_DX)) + 1

print(f"[INFO] Global extent: {XMIN:.2f}..{XMAX:.2f} m "
      f"({XMAX - XMIN:.2f} m), dx={GLOBAL_DX:.4f} m/px, width={W} px")

# -------------------
# Accumulate with linear interpolation to global grid
# -------------------
acc = np.zeros((N_ROWS, W), dtype=float)
wts = np.zeros(W, dtype=float)

def add_section(sec_row):
    # Load GAN csv (32 x 512)
    arr = pd.read_csv(sec_row["gan_csv"]).to_numpy(dtype=float)
    if arr.shape != (N_ROWS, N_COLS):
        raise ValueError(f"{Path(sec_row['gan_csv']).name}: expected {(N_ROWS, N_COLS)}, got {arr.shape}")

    x0 = float(sec_row["x0"])
    dx = float(sec_row["dx"])
    # For each local column j, map to global position; distribute to two nearest global bins
    js = np.arange(N_COLS)
    xj = x0 + js * dx
    pos = (xj - XMIN) / GLOBAL_DX
    k0 = np.floor(pos).astype(int)
    frac = pos - k0

    # Safeguard indices
    valid = (k0 >= 0) & (k0 < W)
    if not np.any(valid):
        return
    k0v = k0[valid]
    f0 = (1.0 - frac[valid])
    k1 = k0v + 1
    f1 = frac[valid]

    # accumulate to k0
    acc[:, k0v] += arr[:, valid] * f0
    wts[k0v] += f0
    # accumulate to k1 when in range
    in_r = k1 < W
    if np.any(in_r):
        acc[:, k1[in_r]] += arr[:, valid][:, in_r] * f1[in_r]
        wts[k1[in_r]] += f1[in_r]

for _, row in man.iterrows():
    add_section(row)

# Normalize (avoid division by zero)
eps = 1e-12
mosaic = acc / np.maximum(wts, eps)[None, :]

# -------------------
# Save outputs
# -------------------
mosaic_csv = OUT_DIR / "schemaGAN_mosaic.csv"
pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)

# --------- Dual X-axes plotting ---------
mosaic_png = OUT_DIR / "schemaGAN_mosaic.png"
plt.figure(figsize=(min(16, W/64), 4))  # adaptive width, cap to 16 inches

# Bottom axis: meters via extent; Y inverted so Depth_Index=0 is at top
plt.imshow(
    mosaic,
    cmap="viridis",
    vmin=0,
    vmax=4.5,
    aspect="auto",
    extent=[XMIN, XMAX, N_ROWS - 1, 0]
)
plt.colorbar(label="Value")

ax = plt.gca()
ax.set_xlabel("Distance along line (m)")
ax.set_ylabel("Depth Index")

# Top axis: global pixel index or normalized 0..32
if not TOP_AXIS_0_TO_32:
    def m_to_px(x): return (x - XMIN) / GLOBAL_DX
    def px_to_m(p): return XMIN + p * GLOBAL_DX
    top = ax.secondary_xaxis('top', functions=(m_to_px, px_to_m))
    top.set_xlabel(f"Pixel index (0…{W-1})")
else:
    def m_to_u32(x): return 32.0 * (x - XMIN) / (XMAX - XMIN + 1e-12)
    def u32_to_m(u): return XMIN + (u / 32.0) * (XMAX - XMIN)
    top = ax.secondary_xaxis('top', functions=(m_to_u32, u32_to_m))
    top.set_xlabel("Normalized distance (0…32)")

plt.title("SchemaGAN Mosaic")
plt.tight_layout()
plt.savefig(mosaic_png, dpi=500)
plt.close()
# ----------------------------------------

print(f"[DONE] Mosaic saved:\n  CSV → {mosaic_csv}\n  PNG → {mosaic_png}")
