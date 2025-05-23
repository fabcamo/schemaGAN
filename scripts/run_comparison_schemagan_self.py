# This code Runs N times SchemaGAN on the same CPTs to see if there is variability in the results.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

from tensorflow.keras.models import load_model

from interpol_compare.functions.utils import (
    get_cptlike_data, format_source_images, compute_mae, generate_gan_image
)
from schemaGAN.functions.utils import load_remove_reshape_data, IC_normalization, get_real_cs_into_image_for_gan, \
    reverse_IC_normalization, add_random_pixel_dropout

import tensorflow as tf

# seed = 12345
# np.random.seed(seed)
# tf.random.set_seed(seed)

def get_fixed_column_positions(csv_file, min_spacing=50, n_cols=5):
    df = pd.read_csv(csv_file, header=None).iloc[2:, 1:].reset_index(drop=True).astype(float)
    df.columns = range(df.shape[1])
    nonzero_columns = df.columns[df.ne(0).any()].tolist()

    max_retries = 25
    for attempt in range(max_retries):
        selected = []
        while len(selected) < n_cols:
            candidate = np.random.choice(nonzero_columns)
            if all(abs(candidate - c) >= min_spacing for c in selected):
                selected.append(candidate)
        selected.sort()
        return df, selected  # <- the full DataFrame and fixed CPT column indices

    raise RuntimeError("Unable to select valid CPT columns")




def plot_schemagan_variability_vertical(
    target_img,
    cpt_img,
    gan_images,
    mae_images,
    path_out,
    img_index=0
):
    """
    Plots an 11x2 grid:
    Left column: Original + 10 SchemaGAN predictions
    Right column: CPT input (row 0) + 10 MAE maps
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    assert len(gan_images) == len(mae_images) == 10, "Expecting 10 GAN and 10 MAE images"

    # Squeeze all input images to 2D (H, W)
    target_img = np.squeeze(target_img)
    cpt_img = np.squeeze(cpt_img)
    gan_images = [np.squeeze(img) for img in gan_images]
    mae_images = [np.squeeze(img) for img in mae_images]

    fig, axs = plt.subplots(11, 2, figsize=(10, 22))
    cmap_main = 'viridis'
    cmap_mae = 'viridis'

    axs[0, 0].imshow(target_img, cmap=cmap_main, vmin=1, vmax=4.5)
    axs[0, 0].set_title("Original", fontsize=10)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cpt_img, cmap=cmap_main, vmin=0, vmax=4.5)
    axs[0, 1].set_title("CPT Input", fontsize=10)
    axs[0, 1].axis('off')

    for i in range(10):
        axs[i+1, 0].imshow(gan_images[i], cmap=cmap_main, vmin=1, vmax=4.5)
        axs[i+1, 0].set_title(f"SchemaGAN {i+1}", fontsize=10)
        axs[i+1, 0].axis('off')

        axs[i+1, 1].imshow(mae_images[i], cmap=cmap_mae, vmin=0, vmax=1)
        axs[i+1, 1].set_title(f"MAE {i+1}: {np.mean(mae_images[i]):.4f}", fontsize=10)
        axs[i+1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(path_out, exist_ok=True)
    fig_path = os.path.join(path_out, f'schemaGAN_variability_vertical_img{img_index}.pdf')
    plt.savefig(fig_path, format='pdf')
    plt.close()
    print("Saved vertical 11x2 comparison plot to:", fig_path)


def plot_schemagan_variability_vs_first(
    cpt_img,
    gan_images,
    path_out,
    img_index=0
):
    """
    Plots an 11x2 grid:
    Left column: 10 SchemaGAN outputs
    Right column:
      - Row 0: CPT input
      - Rows 1–10: difference (abs) vs first GAN output
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    assert len(gan_images) == 10, "Expecting exactly 10 GAN images"

    # Squeeze all input images to 2D
    cpt_img = np.squeeze(cpt_img)
    gan_images = [np.squeeze(img) for img in gan_images]

    ref_img = gan_images[0]
    diff_images = [np.abs(img - ref_img) for img in gan_images]

    fig, axs = plt.subplots(11, 2, figsize=(10, 22))
    cmap_main = 'viridis'
    cmap_diff = 'viridis'

    # Row 0
    axs[0, 0].imshow(ref_img, cmap=cmap_main, vmin=1, vmax=4.5)
    axs[0, 0].set_title("Reference: SchemaGAN 1", fontsize=10)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cpt_img, cmap=cmap_main, vmin=0, vmax=4.5)
    axs[0, 1].set_title("CPT Input", fontsize=10)
    axs[0, 1].axis('off')

    # Rows 1–10
    for i in range(10):
        axs[i+1, 0].imshow(gan_images[i], cmap=cmap_main, vmin=1, vmax=4.5)
        axs[i+1, 0].set_title(f"SchemaGAN {i+1}", fontsize=10)
        axs[i+1, 0].axis('off')

        axs[i+1, 1].imshow(diff_images[i], cmap=cmap_diff, vmin=0, vmax=1)
        axs[i+1, 1].set_title(f"Diff vs Run 1: {np.mean(diff_images[i]):.4f}", fontsize=10)
        axs[i+1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(path_out, exist_ok=True)
    fig_path = os.path.join(path_out, f'schemaGAN_variability_diff_vs_first_img{img_index}.pdf')
    plt.savefig(fig_path, format='pdf')
    plt.close()
    print("Saved variability vs first run plot to:", fig_path)



# ---------------- USER CONFIG ----------------
name_of_model_to_use = 'schemaGAN.h5'
SIZE_X, SIZE_Y = 512, 32
miss_rate = 0.99
min_distance = 51
runs = 10

path_validation = 'D:/schemaGAN/data/compare'
path_real_images = 'D:\schemaGAN\data\eemskanaal\emm02_512x32.csv'
path_to_model = 'D:/schemaGAN/h5'
path_results = 'D:/schemaGAN/tests/schemaGAN_variability'
os.makedirs(path_results, exist_ok=True)

# ---------------- FIXED MODEL ----------------
generator = os.path.join(path_to_model, name_of_model_to_use)
assert os.path.isfile(generator), f"Model not found: {generator}"

# ---------------- STATIC PREPROCESSING ----------------
rows, cols = np.linspace(0, SIZE_Y - 1, SIZE_Y), np.linspace(0, SIZE_X - 1, SIZE_X)
grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

tar_images, src_images = load_remove_reshape_data(
    path_validation, miss_rate, min_distance, SIZE_Y, SIZE_X
)
no_validation_images = src_images.shape[0]
data = [src_images, tar_images]
dataset = IC_normalization(data)
[input_img, orig_img] = dataset

coords_all, pixel_values_all = get_cptlike_data(src_images)
original_images, cptlike_img = format_source_images(dataset)

# ---------------- LOOP: RUN GAN MULTIPLE TIMES ----------------
all_mae = []

# Choose which image to visualize
idx = 0
gan_imgs = []
mae_imgs = []

model = load_model(generator)
#real_cs = get_real_cs_into_image_for_gan(path_real_images, pixel_dropout_rate=0)
df_all_cpt, kept_columns = get_fixed_column_positions(path_real_images)


pixel_dropout_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

for dropout_rate in pixel_dropout_rates:
    print(f"\n### Running for pixel dropout rate = {dropout_rate} ###")

    # Reset for each dropout level
    gan_imgs = []
    mae_imgs = []
    all_mae = []

    for run in range(runs):
        seed = np.random.randint(100000)
        np.random.seed(seed)
        print(f"Run {run + 1}/{runs} - Seed: {seed}")

        start_time = time.time()
        df_reduced = df_all_cpt.copy()
        df_reduced[df_reduced.columns[~df_reduced.columns.isin(kept_columns)]] = 0.0

        df_noisy = add_random_pixel_dropout(df_reduced, kept_columns, dropout_rate)
        cs_to_evaluate = df_noisy.values.astype(float).reshape(1, 32, 512, 1)
        real_cs = IC_normalization([cs_to_evaluate, cs_to_evaluate])
        gan_images = generate_gan_image(model, real_cs)

        elapsed = time.time() - start_time
        print(f"  GAN generation took {elapsed:.2f} seconds")

        mae_gan, *_ = compute_mae(
            original_images, gan_images, gan_images, gan_images, gan_images, gan_images, gan_images,
            path_results
        )

        all_mae.append({
            "dropout": dropout_rate,
            "run": run + 1,
            "seed": seed,
            "mae_mean": np.mean(mae_gan)
        })

        gan_img = gan_images[idx]
        mae_img = np.abs(gan_img - original_images[idx])
        gan_imgs.append(gan_img)
        mae_imgs.append(mae_img)

    # Save and plot results for this dropout level
    df = pd.DataFrame(all_mae)
    df.to_csv(os.path.join(path_results, f'mae_dropout_{dropout_rate:.2f}.csv'), index=False)

    # plot_schemagan_variability_vertical(
    #     target_img=original_images[idx],
    #     cpt_img=cptlike_img[idx],
    #     gan_images=gan_imgs,
    #     mae_images=mae_imgs,
    #     path_out=path_results,
    #     img_index=idx
    # )


    plot_schemagan_variability_vs_first(
        cpt_img=cptlike_img[idx],
        gan_images=gan_imgs,
        path_out=os.path.join(path_results, f"dropout_{int(dropout_rate * 100):02d}"),
        img_index=idx
    )





