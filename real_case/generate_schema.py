import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Utils
from schemaGAN.functions.utils import IC_normalization, reverse_IC_normalization

# -------------------
# CONFIG
# -------------------
SECTIONS_DIR = Path(r"C:\VOW\data\schgan_inputs\betuwepand_dike_north")  # where the section CSVs are
PATH_TO_MODEL = Path(r"D:\schemaGAN\h5\schemaGAN.h5")          # generator .h5
MANIFEST_CSV = Path(r"C:\VOW\data\schgan_inputs\betuwepand_dike_north\manifest_sections.csv")
COORDS_WITH_DIST_CSV = Path(r"C:\VOW\data\schgan_inputs\betuwepand_dike_north\coords_with_distances.csv")
OUT_DIR = Path(r"C:\VOW\res\betuwepand\dike_north")                         # where to save outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE_X = 512
SIZE_Y = 32

# Real-depth range used during equalization/compression (set these!)
Y_TOP_M    = 6.862       # depth at Depth_Index = 0
Y_BOTTOM_M = -13.041     # depth at Depth_Index = 31

# Top x-axis appearance: False -> 0..511 px, True -> 0..32 normalized
TOP_AXIS_0_TO_32 = False

# Make TF not grab all GPU memory
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# Reproducible-ish seed
seed = np.random.randint(20220412, 20230412)
np.random.seed(seed)
tf.random.set_seed(seed)

print(f"[INFO] Using seed: {seed}")

# -------------------
# LOAD MODEL + METADATA
# -------------------
print("[INFO] Loading model…")
model = load_model(PATH_TO_MODEL)

# Load manifest and coords once
man = pd.read_csv(MANIFEST_CSV)
coords = pd.read_csv(COORDS_WITH_DIST_CSV)

for col in ["section_index", "span_m", "left_pad_m", "right_pad_m", "start_idx"]:
    if col not in man.columns:
        raise ValueError(f"Manifest missing column: {col}")
if "cum_along_m" not in coords.columns:
    raise ValueError("coords_with_distances.csv must contain 'cum_along_m'")

man["section_index"] = man["section_index"].astype(int)
man["start_idx"] = man["start_idx"].astype(int)

def _parse_section_index(path: Path) -> int:
    """Extract section index from filename like schemaGAN_section_001.csv"""
    m = re.search(r"schemaGAN_section_(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse section index from {path.name}")
    return int(m.group(1))

def _sec_x0_dx(sec_index: int) -> tuple[float, float]:
    """
    Bottom x axis spans meters via:
      x0 = cum_along(first CPT of section) - left_pad_m
      dx = (span + left_pad + right_pad) / (SIZE_X - 1)
    """
    r = man.loc[man["section_index"] == sec_index]
    if r.empty:
        raise ValueError(f"No manifest row for section {sec_index}")
    r = r.iloc[0]
    total_span = float(r["span_m"] + r["left_pad_m"] + r["right_pad_m"])
    start_idx = int(r["start_idx"])
    m0 = float(coords.loc[start_idx, "cum_along_m"])
    x0 = m0 - float(r["left_pad_m"])
    dx = 1.0 if total_span <= 0 else total_span / (SIZE_X - 1)
    return x0, dx

# Y mapping funcs (primary y = Depth_Index, secondary y = meters)
def idx_to_meters(y_idx: float) -> float:
    return Y_TOP_M + (y_idx / (SIZE_Y - 1)) * (Y_BOTTOM_M - Y_TOP_M)

def meters_to_idx(y_m: float) -> float:
    denom = (Y_BOTTOM_M - Y_TOP_M)
    return 0.0 if abs(denom) < 1e-12 else (y_m - Y_TOP_M) * (SIZE_Y - 1) / denom

# -------------------
# CORE
# -------------------
def run_gan_on_section_csv(csv_path: Path) -> tuple[Path, Path]:
    """Run SchemaGAN on one section CSV and save CSV + PNG image. Returns (csv_out, png_out)."""
    # Load csv and strip Depth_Index column
    df = pd.read_csv(csv_path)
    if df.shape[0] != SIZE_Y:
        raise ValueError(f"{csv_path.name}: expected {SIZE_Y} rows, got {df.shape[0]}")
    df_vals = df.iloc[:, 1:]  # drop first column (Depth_Index)

    if df_vals.shape[1] != SIZE_X:
        raise ValueError(f"{csv_path.name}: expected {SIZE_X} columns (after dropping first), got {df_vals.shape[1]}")

    # To numpy & reshape to (1, 32, 512, 1)
    cs = df_vals.to_numpy(dtype=float).reshape(1, SIZE_Y, SIZE_X, 1)

    # Normalization trick (your util returns a pair)
    cs_norm = IC_normalization([cs, cs])[0]

    # Predict
    gan_res = model.predict(cs_norm, verbose=0)

    # Reverse normalization (back to your plotting range)
    gan_res = reverse_IC_normalization(gan_res)
    gan_res = np.squeeze(gan_res)  # (32, 512)

    # Save CSV
    out_csv = OUT_DIR / f"{csv_path.stem}_seed{seed}_gan.csv"
    pd.DataFrame(gan_res).to_csv(out_csv, index=False)

    # --------- Dual axes plotting ---------
    sec_index = _parse_section_index(csv_path)
    x0, dx = _sec_x0_dx(sec_index)
    x1 = x0 + (SIZE_X - 1) * dx

    out_png = OUT_DIR / f"{csv_path.stem}_seed{seed}_gan.png"
    plt.figure(figsize=(10, 2.4))

    # Bottom axis: meters (extent sets x in meters; y stays in Depth_Index 31..0)
    plt.imshow(
        gan_res,
        cmap='viridis',
        vmin=0,
        vmax=4.5,
        aspect='auto',
        extent=[x0, x1, SIZE_Y - 1, 0]  # x in meters, y inverted so 0 at top
    )
    plt.colorbar(label='Value')

    ax = plt.gca()
    ax.set_xlabel('Distance along line (m)')
    ax.set_ylabel('Depth Index')

    # Top x-axis (pixels or 0..32 normalized)
    if not TOP_AXIS_0_TO_32:
        def m_to_px(x): return (x - x0) / dx
        def px_to_m(p): return x0 + p * dx
        top = ax.secondary_xaxis('top', functions=(m_to_px, px_to_m))
        top.set_xlabel(f'Pixel index (0…{SIZE_X-1})')
    else:
        def m_to_u32(x): return 32.0 * (x - x0) / (x1 - x0 + 1e-12)
        def u32_to_m(u): return x0 + (u / 32.0) * (x1 - x0)
        top = ax.secondary_xaxis('top', functions=(m_to_u32, u32_to_m))
        top.set_xlabel('Normalized distance (0…32)')

    # Right y-axis: real depth (m)
    right = ax.secondary_yaxis('right', functions=(idx_to_meters, meters_to_idx))
    right.set_ylabel('Depth (m)')

    plt.title(f'SchemaGAN Generated Image (Section {sec_index:03d}, Seed: {seed})')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    # ----------------------------------------

    return out_csv, out_png

# -------------------
# MAIN LOOP
# -------------------
section_files = sorted(SECTIONS_DIR.glob("schemaGAN_section_*.csv"))
if not section_files:
    raise FileNotFoundError(f"No section CSVs found in {SECTIONS_DIR}")

print(f"[INFO] Found {len(section_files)} section(s) in {SECTIONS_DIR}")

ok, fail = 0, 0
for i, sec in enumerate(section_files, 1):
    try:
        csv_out, png_out = run_gan_on_section_csv(sec)
        ok += 1
        print(f"[{i:03d}/{len(section_files)}] OK → CSV: {csv_out.name} | PNG: {png_out.name}")
    except Exception as e:
        fail += 1
        print(f"[{i:03d}/{len(section_files)}] FAIL on {sec.name}: {e}")

print(f"[DONE] Success: {ok}, Failed: {fail}. Outputs in: {OUT_DIR}")
