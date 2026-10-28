# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:29:30 2026

@author: Admin
"""


import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

# Paths
indexPath = r'C:\Users\pawach\OneDrive - Tallinna Tehnikaülikool\Article_1_Data\Databases\Cations\Index.xlsx'

# Read Excel file
df = pd.read_excel(indexPath)

# Fill only missing InChI Keys
for i, row in df.iterrows():
    if pd.isna(row['InChI_Key']) or str(row['InChI_Key']).strip() == '':
        smiles = row['SMILES']

        if pd.notna(smiles) and str(smiles).strip() != '':
            mol = Chem.MolFromSmiles(str(smiles))

            if mol is not None:
                df.at[i, 'InChI_Key'] = inchi.MolToInchiKey(mol)
            else:
                print(f"Invalid SMILES at row {i + 2}: {smiles}")

# Save back to the same file
df.to_excel(indexPath, index=False)

print("Done. Missing InChI Keys have been added.")

# Check for duplicate InChI Keys
duplicate_mask = df['InChI_Key'].duplicated(keep=False)

if duplicate_mask.any():
    print("\nDuplicate InChI Keys found:")
    print(df.loc[duplicate_mask, ['InChI_Key', 'SMILES']].sort_values('InChI_Key'))
    
    print(f"\nNumber of duplicated rows: {duplicate_mask.sum()}")
    print(f"Number of unique duplicated InChI Keys: {df.loc[duplicate_mask, 'InChI_Key'].nunique()}")
else:
    print("\nNo duplicate InChI Keys found.")
    
    
    

    
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageChops
import string
import numpy as np
# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing the 7 plot images
folder_path = Path(
    r"C:\Users\pawach\OneDrive - Tallinna Tehnikaülikool\Article_1_Data\ILThermo_Test\Histogram_Plots"
)

# Output files will be saved in the same folder
output_svg = folder_path / "Combined_Histograms.svg"
output_png = folder_path / "Combined_Histograms_600dpi.png"


ADD_PANEL_LABELS = True
TRIM_WHITE_SPACE = True

# Space between panels
horizontal_gap = 0.012
vertical_gap = 0.035

# Outer figure margins
left_margin = 0.015
right_margin = 0.015
top_margin = 0.02
bottom_margin = 0.02


# ============================================================
# TRIM WHITE BORDER
# ============================================================

def trim_white_border(img, margin=8):

    img = img.convert("RGB")

    background = Image.new(
        "RGB",
        img.size,
        "white"
    )

    diff = ImageChops.difference(
        img,
        background
    )

    bbox = diff.getbbox()

    if bbox is not None:

        left = max(0, bbox[0] - margin)
        top = max(0, bbox[1] - margin)
        right = min(img.width, bbox[2] + margin)
        bottom = min(img.height, bbox[3] + margin)

        img = img.crop(
            (left, top, right, bottom)
        )

    return img


# ============================================================
# FIND IMAGES
# ============================================================

extensions = [
    "*.jpg", "*.jpeg", "*.png",
    "*.JPG", "*.JPEG", "*.PNG"
]

image_files = []

for ext in extensions:
    image_files.extend(folder_path.glob(ext))

image_files = sorted(
    set(image_files),
    key=lambda x: x.name.lower()
)

if len(image_files) != 7:
    raise ValueError(
        f"Expected 7 images, found {len(image_files)}."
    )

print("\nPlot order:")

for i, file in enumerate(image_files):
    print(f"{string.ascii_lowercase[i]}) {file.name}")


# ============================================================
# LOAD AND CROP
# ============================================================

images = []

for file in image_files:

    img = Image.open(file)

    if TRIM_WHITE_SPACE:
        img = trim_white_border(img)

    images.append(img)


# ============================================================
# DETERMINE AVERAGE PANEL ASPECT RATIO
# ============================================================

aspect_ratios = [
    img.width / img.height
    for img in images
]

average_aspect = np.median(aspect_ratios)

print(f"\nMedian panel aspect ratio: {average_aspect:.3f}")


# ============================================================
# FIGURE DIMENSIONS
# ============================================================

# Journal double-column width
figure_width_in = 180 / 25.4

# Width available for four panels
usable_width = (
    1
    - left_margin
    - right_margin
    - 3 * horizontal_gap
)

panel_width = usable_width / 4

# Calculate panel height from image aspect ratio
#
# Normalized panel height depends on final figure aspect.
# Start from physical panel width and derive physical height.

physical_panel_width = figure_width_in * panel_width

physical_panel_height = (
    physical_panel_width / average_aspect
)

# Two rows + small vertical gap
figure_height_in = (
    2 * physical_panel_height
    + physical_panel_height * 0.10
)

fig = plt.figure(
    figsize=(
        figure_width_in,
        figure_height_in
    ),
    facecolor="white"
)


# ============================================================
# NORMALIZED PANEL DIMENSIONS
# ============================================================

panel_height = (
    physical_panel_height /
    figure_height_in
)

vertical_gap_norm = vertical_gap

# Top and bottom row positions
bottom_y = bottom_margin

top_y = (
    bottom_y
    + panel_height
    + vertical_gap_norm
)


# ============================================================
# X POSITIONS
# ============================================================

top_x = [
    left_margin
    + i * (panel_width + horizontal_gap)
    for i in range(4)
]

# Centre three plots in second row
total_bottom_width = (
    3 * panel_width
    + 2 * horizontal_gap
)

bottom_start = (
    1 - total_bottom_width
) / 2

bottom_x = [
    bottom_start
    + i * (panel_width + horizontal_gap)
    for i in range(3)
]


positions = []

# Top row
for x in top_x:
    positions.append([
        x,
        top_y,
        panel_width,
        panel_height
    ])

# Bottom row
for x in bottom_x:
    positions.append([
        x,
        bottom_y,
        panel_width,
        panel_height
    ])


# ============================================================
# ADD IMAGES
# ============================================================

for i, (img, pos) in enumerate(
    zip(images, positions)
):

    ax = fig.add_axes(pos)

    ax.imshow(img)

    ax.set_axis_off()

    if ADD_PANEL_LABELS:

        ax.text(
            0.005,
            0.995,
            f"({string.ascii_lowercase[i]})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                pad=0.5
            )
        )


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    output_svg,
    format="svg",
    bbox_inches="tight",
    pad_inches=0.01,
    facecolor="white"
)

plt.savefig(
    output_png,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01,
    facecolor="white"
)

plt.show()

print("\nSaved:")
print(output_svg)
print(output_png)