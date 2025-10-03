# This is a script that reads the coordiantes of the CPTs and organizes them from west to east (left to right).
# Then it calculates the euclidean distance in two ways: first the distance between the first cpt and all the rest
# and then the distance between each cpt and the next one.
# Once we know all the distances, we can create the input file for schemaGAN. This uses a csv file of size 512x32
# with 512 columns (representing the distance) and 32 rows (representing the depth. In each position a value is assigned
# to represent the soil type at that depth and distance. To do this we first create a 512x32 matrix filled with 0s
# and then we find the closest cpt to each column and assign the soil type of that cpt to all the rows of that column.
# Finally we save the matrix as a csv file. Key here is to keep track of the distance scale. It is not necesarry
# to have a 1:1 scale. First lets find the max distance, then we can divide that by 512 see how many sections of
# that size fit in the max distance. We want to fit around 6 CPTs per 512 columns, so we can adjust the scale accordingly.

import math
from pathlib import Path
import numpy as np
import pandas as pd

##### CONFIGURATION #####
COORDS_CSV = r"C:\VOW\gis\coords\betuwepand_dike_north.csv"
CPT_DATA_CSV = r"C:\VOW\data\betuwepand\dike_north\compressed_cpt_data.csv"
OUT_DIR = Path(r"C:\VOW\data\schgan_inputs\betuwepand_dike_north")

# If the OUT_DIR does not exist, create it
OUT_DIR.mkdir(parents=True, exist_ok=True)

##### SECTIONING #####
CPTS_PER_SECTION = 6  # How many CPTs we want per section
OVERLAP_CPTS = 2  # How many CPTs we want to overlap between sections
LEFT_PAD_FRACTION = 0.05  # Fraction of the section to pad on the left
RIGHT_PAD_FRACTION = 0.05  # Fraction of the section to pad
N_COLS = 512  # Number of columns in the output matrix
N_ROWS = 32  # Number of rows in the output matrix (depth)

##### LOAD #####
coords_df = pd.read_csv(COORDS_CSV)
cpt_df = pd.read_csv(CPT_DATA_CSV)

# Basic checks
assert "name" in coords_df.columns and "x" in coords_df.columns and "y" in coords_df.columns, \
    "coords CSV must have columns: name, x, y"
assert "Depth_Index" in cpt_df.columns, "CPT data CSV must have a Depth_Index column"
assert len(cpt_df) == N_ROWS, f"CPT data must have {N_ROWS} rows (Depth_Index=0..{N_ROWS-1})"

# Sort east to west
coords_df = coords_df.sort_values("x", ascending=True).reset_index(drop=True)

# Compute the distances
def euclid(x1, y1, x2, y2):
    return float(math.hypot(x2 - x1, y2 - y1))

# from first CPT to each
x0, y0 = coords_df.loc[0, ["x", "y"]]
coords_df["dist_from_first_m"] = coords_df.apply(lambda r: euclid(x0, y0, r["x"], r["y"]), axis=1)

# between neighbors
d_prev = [0.0]
for i in range(1, len(coords_df)):
    d_prev.append(euclid(coords_df.loc[i-1, "x"], coords_df.loc[i-1, "y"], coords_df.loc[i, "x"], coords_df.loc[i, "y"]))
coords_df["dist_from_prev_m"] = d_prev

# cumulative distance along the chain (neighbor-based)
coords_df["cum_along_m"] = coords_df["dist_from_prev_m"].cumsum()

# PRINT THE DISTANCES
print("Distances between CPTs (m):")
print(coords_df[["name", "dist_from_first_m", "dist_from_prev_m", "cum_along_m"]])

# Name matching
cpt_columns = [c for c in cpt_df.columns if c != "Depth_Index"]
name_set_data = set(cpt_columns)
coords_df["has_data"] = coords_df["name"].isin(name_set_data)

unmatched = coords_df.loc[~coords_df["has_data"], "name"].tolist()
if unmatched:
    print(f"[WARN] {len(unmatched)} coordinate names have no matching column in CPT data and will be skipped.")

##### SECTIONING #####
step = max(1, CPTS_PER_SECTION - OVERLAP_CPTS)
n = len(coords_df)
starts = list(range(0, max(1, n - CPTS_PER_SECTION + 1), step))
if (n - 1) not in starts and (n - CPTS_PER_SECTION) > 0:
    # Ensure last section includes the last CPT
    last_start = n - CPTS_PER_SECTION
    if last_start > starts[-1]:
        starts.append(last_start)

manifest = []

for si, start in enumerate(starts, 1):
    end = min(start + CPTS_PER_SECTION, n)
    sect = coords_df.iloc[start:end].copy()
    if sect.empty:
        continue

    # Distances within section relative to the first CPT in this section
    base_x, base_y = sect.iloc[0][["x", "y"]]
    sect["dist_rel_m"] = sect.apply(lambda r: euclid(base_x, base_y, r["x"], r["y"]), axis=1)

    # Span & margins
    span = max(1e-9, sect["dist_rel_m"].max())   # avoid zero span
    left_pad = LEFT_PAD_FRACTION * span
    right_pad = RIGHT_PAD_FRACTION * span
    total_span = span + left_pad + right_pad

    # Map CPT positions to column indices in [0, N_COLS-1]
    def to_col(d_rel):
        u = (d_rel + left_pad) / total_span
        return int(round(u * (N_COLS - 1)))

    sect["col"] = sect["dist_rel_m"].apply(to_col)
    # Clip to bounds
    sect["col"] = sect["col"].clip(0, N_COLS - 1)

    # Resolve collisions by nudging right if needed
    used = set()
    cols_resolved = []
    for c in sect["col"].tolist():
        cc = c
        while cc in used and cc < N_COLS - 1:
            cc += 1
        if cc in used:  # still collision at right edge; nudge left
            cc = c
            while cc in used and cc > 0:
                cc -= 1
        used.add(cc)
        cols_resolved.append(cc)
    sect["col"] = cols_resolved

    # Build grid (zeros) and paint CPT columns
    grid = np.zeros((N_ROWS, N_COLS), dtype=float)

    painted = []
    skipped = []
    # Ensure depth ordering
    cpt_df_sorted = cpt_df.sort_values("Depth_Index")

    for _, r in sect.iterrows():
        name = r["name"]
        col = int(r["col"])
        if name in cpt_df_sorted.columns:
            vals = cpt_df_sorted[name].to_numpy()
            if len(vals) != N_ROWS:
                print(f"[WARN] Column '{name}' has {len(vals)} rows, expected {N_ROWS}. Skipping.")
                skipped.append(name)
                continue
            grid[:, col] = vals  # top row index=0 is surface
            painted.append({"name": name, "col": col})
        else:
            skipped.append(name)

    # Save section CSV
    out_csv = OUT_DIR / f"schemaGAN_section_{si:03d}.csv"
    out_df = pd.DataFrame(grid, columns=[f"x{j:03d}" for j in range(N_COLS)])
    out_df.insert(0, "Depth_Index", np.arange(N_ROWS, dtype=int))
    out_df.to_csv(out_csv, index=False)

    # Record manifest info
    manifest.append({
        "section_index": si,
        "start_idx": int(start),
        "end_idx": int(end - 1),
        "first_name": sect.iloc[0]["name"],
        "last_name": sect.iloc[-1]["name"],
        "span_m": float(span),
        "left_pad_m": float(left_pad),
        "right_pad_m": float(right_pad),
        "painted_count": len(painted),
        "skipped_count": len(skipped),
        "csv_path": str(out_csv),
        "painted": painted,
        "skipped": skipped,
    })

# Save distances + manifest for traceability
coords_out = OUT_DIR / "coords_with_distances.csv"
coords_df.to_csv(coords_out, index=False)

man_df_rows = []
for m in manifest:
    man_df_rows.append({
        "section_index": m["section_index"],
        "start_idx": m["start_idx"],
        "end_idx": m["end_idx"],
        "first_name": m["first_name"],
        "last_name": m["last_name"],
        "span_m": m["span_m"],
        "left_pad_m": m["left_pad_m"],
        "right_pad_m": m["right_pad_m"],
        "painted_count": m["painted_count"],
        "skipped_count": m["skipped_count"],
        "csv_path": m["csv_path"],
    })
man_df = pd.DataFrame(man_df_rows)
man_df.to_csv(OUT_DIR / "manifest_sections.csv", index=False)

print(f"Written {len(manifest)} sections to: {OUT_DIR.resolve()}")
print(f"Coords+distances → {coords_out}")
if any(m['skipped_count'] for m in manifest):
    print("[NOTE] Some CPT names in coords had no matching data columns and were left as zero columns. "
          "Consider aligning names or adding a mapping step.")