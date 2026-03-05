import math
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import hdbscan

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

import umap


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / 'moldash.html'
CSV_PATH = BASE_DIR / 'df_candidates.csv'
HTML_TEMPLATE_PATH = BASE_DIR / "spherical_space_template.html"

# df = pd.read_csv(CSV_PATH)
# df = df.drop_duplicates(subset="ISOSMILES").reset_index(drop=True)
# smiles_list = df['ISOSMILES'].tolist()
# mols = [Chem.MolFromSmiles(s) for s in smiles_list]

# # ----------------------------
# # 2) Fingerprints (Morgan/ECFP-like)
# # ----------------------------
# n_bits = 2048
# radius = 2
# morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

# fps = []
# for mol in mols:
#   fp = morgan_generator.GetFingerprint(mol)
#   arr = np.zeros((n_bits,), dtype=np.uint8)
#   DataStructs.ConvertToNumpyArray(fp, arr)
#   fps.append(arr)

# X = np.vstack(fps).astype(np.float32)

# # ----------------------------
# # 3) UMAP to 3D
# # ----------------------------
# reducer = umap.UMAP(
#   n_components=3,
#   n_neighbors=15,
#   min_dist=0.1,
#   metric="jaccard",
#   random_state=7,
#   n_jobs=1,
# )

# with warnings.catch_warnings():
#   warnings.filterwarnings(
#     "ignore",
#     message="gradient function is not yet implemented for jaccard distance metric; inverse_transform will be unavailable",
#     category=UserWarning,
#     module="umap\\.umap_",
#   )
#   xyz = reducer.fit_transform(X)

# # Recentre embedding (recommended before spherical conversion)
# xyz_centered = xyz - xyz.mean(axis=0, keepdims=True)

# x = xyz_centered[:, 0]
# y = xyz_centered[:, 1]
# z = xyz_centered[:, 2]

# # ----------------------------
# # 4) Cartesian -> spherical
# #    r >= 0
# #    theta in [0, pi] is polar angle from +z
# #    phi in (-pi, pi] is azimuth in x-y plane
# # ----------------------------
# r = np.sqrt(x * x + y * y + z * z)

# # Avoid divide-by-zero when r is extremely small
# eps = 1e-12
# theta = np.arccos(np.clip(z / np.maximum(r, eps), -1.0, 1.0))
# phi = np.arctan2(y, x)

# # ----------------------------
# # 4b) HDBSCAN on spherical coordinates
# #     Labels are cluster ids; -1 denotes noise.
# # ----------------------------
# spherical_coords = np.column_stack((r, theta, phi))
# clusterer = hdbscan.HDBSCAN(
#   min_cluster_size=5,
#   metric="euclidean",
# )
# group_labels = clusterer.fit_predict(spherical_coords)

# df = pd.DataFrame({
#   "smiles": smiles_list,
#   "x": x,
#   "y": y,
#   "z": z,
#   "r": r,
#   "theta": theta,
#   "phi": phi,
#   "group_label": group_labels,
# })
# numeric_columns = ["x", "y", "z", "r", "theta", "phi"]
# df[numeric_columns] = df[numeric_columns].round(4)

# df.to_csv(BASE_DIR / "umap3d_spherical.csv", index=False)
# print("\nSaved: umap3d_spherical.csv")

# ----------------------------
# 5) Build the final HTML with your 300 molecules embedded
#    This writes a single self-contained HTML you can double click.
# ----------------------------
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "umap3d_spherical.csv")
smiles_list = df["smiles"].tolist()
r = df["r"].values
theta = df["theta"].values
phi = df["phi"].values
group_labels = df["group_label"].fillna(-1).astype(int).values

sphere_radius = 3.2

r_norm = (r - r.min()) / (r.max() - r.min() + 1e-12)
r_web = sphere_radius * np.cbrt(r_norm)

x_web = r_web * np.sin(theta) * np.cos(phi)
y_web = r_web * np.sin(theta) * np.sin(phi)
z_web = r_web * np.cos(theta)

points = []
for i in range(len(smiles_list)):
  points.append({
    "smiles": smiles_list[i],
    "x": round(float(x_web[i]), 4),
    "y": round(float(y_web[i]), 4),
    "z": round(float(z_web[i]), 4),
    "t": round(float(r_norm[i]), 4),
    "group_label": int(group_labels[i]),
  })

data_js = "const DATA = " + json.dumps(points, ensure_ascii=False) + ";"
html = (
  HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
  .replace("__DATA_JS__", data_js)
  .replace("__SPHERE_RADIUS__", str(sphere_radius))
)

out_path = BASE_DIR / "umap_sphere_300.html"
with open(out_path, "w", encoding="utf-8") as f:
  f.write(html)

print(f"Saved: {out_path}")
