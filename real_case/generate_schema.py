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
SECTIONS_DIR = Path(r"C:\VOW\data\Site_A\O\schGAN_sections")  # where the section CSVs are
PATH_TO_MODEL = Path(r"D:\schemaGAN\h5\schemaGAN.h5")          # generator .h5
MANIFEST_CSV = Path(r"C:\VOW\data\Site_A\O\schGAN_sections\manifest_sections.csv")
COORDS_WITH_DIST_CSV = Path(r"C:\VOW\data\Site_A\O\schGAN_sections\coords_with_distances.csv")
OUT_DIR = Path(r"C:\VOW\res\Site_A\O")                         # where to save outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE_X = 512
SIZE_Y = 32

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
# LOAD MODEL ONCE
# -------------------
print("[INFO] Loading model…")
model = load_model(PATH_TO_MODEL)

# -------------------
# HELPERS
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

    # Normalization trick (your existing pattern returns a pair)
    norm_pair = IC_normalization([cs, cs])
    cs_norm = norm_pair[0]

    # Predict
    gan_res = model.predict(cs_norm, verbose=0)

    # Reverse normalization to 0..255-ish (as per your utils)
    gan_res = reverse_IC_normalization(gan_res)
    gan_res = np.squeeze(gan_res)  # (32, 512)

    # Save CSV
    out_csv = OUT_DIR / f"{csv_path.stem}_seed{seed}_gan.csv"
    pd.DataFrame(gan_res).to_csv(out_csv, index=False)

    # Save image
    out_png = OUT_DIR / f"{csv_path.stem}_seed{seed}_gan.png"
    plt.figure(figsize=(10, 2.2))
    plt.imshow(gan_res, cmap='viridis', vmin=0, vmax=4.5, aspect='auto')
    plt.colorbar(label='Value')
    plt.title(f'SchemaGAN Generated Image (Seed: {seed})')
    plt.xlabel('Column Index')
    plt.ylabel('Depth Index')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

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
