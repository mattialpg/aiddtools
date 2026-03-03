import math
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

import umap


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / 'moldash.html'
CSV_PATH = BASE_DIR / 'df_candidates.csv'

# df = pd.read_csv(CSV_PATH)
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

# df = pd.DataFrame({
#   "smiles": smiles_list,
#   "x": x,
#   "y": y,
#   "z": z,
#   "r": r,
#   "theta": theta,
#   "phi": phi
# })

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
    "x": float(x_web[i]),
    "y": float(y_web[i]),
    "z": float(z_web[i]),
    "t": float(r_norm[i])
  })

data_js = "const DATA = " + json.dumps(points, ensure_ascii=False) + ";"

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive UMAP Sphere</title>
<style>
html, body {{
  margin: 0;
  height: 100%;
  background: #070A10;
  overflow: hidden;
}}
canvas {{
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
}}
.hud {{
  position: fixed;
  left: 12px;
  top: 12px;
  font: 12px/1.35 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  color: rgba(255,255,255,0.9);
  background: rgba(0,0,0,0.35);
  padding: 10px 12px;
  border-radius: 10px;
  user-select: none;
}}
.hud button {{
  margin-top: 8px;
  width: 100%;
  border: 0;
  border-radius: 8px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.95);
  cursor: pointer;
}}
</style>
</head>
<body>

<div class="hud">
  <div>Left drag rotate. Middle drag pan. Scroll zoom.</div>
  <button id="reset-view" type="button">Reset view</button>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/examples/jsm/controls/OrbitControls.js": "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"
  }}
}}
</script>

<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/examples/jsm/controls/OrbitControls.js";

{data_js}

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x070A10, 1);
document.body.appendChild(renderer.domElement);

renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x070A10, 6, 22);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(0, 0, 8.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);

controls.enableDamping = true;
controls.dampingFactor = 0.08;

controls.enableRotate = true;
controls.enablePan = true;

// We implement zoom-to-point ourselves
controls.enableZoom = false;

controls.minDistance = 0.02;
controls.maxDistance = 18;

// left drag rotate, middle drag pan
controls.mouseButtons = {{
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.PAN,
  RIGHT: THREE.MOUSE.DOLLY
}};

controls.update();
controls.saveState();

document.getElementById("reset-view").addEventListener("click", () => {{
  controls.reset();
  controls.update();
}});

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(6, 8, 10);
scene.add(dir);

// Sphere wireframe
const sphereRadius = {sphere_radius};
const wire = new THREE.LineSegments(
  new THREE.WireframeGeometry(new THREE.SphereGeometry(sphereRadius, 32, 20)),
  new THREE.LineBasicMaterial({{ color: 0x6f86c9, transparent: true, opacity: 0.2 }})
);
scene.add(wire);

// Points
const base = new THREE.Color(0x9bb7ff);
const alt = new THREE.Color(0x7ef0c1);

const pointMeshes = [];
const sphereGeometry = new THREE.SphereGeometry(0.05, 16, 12);
let clickStartX = 0;
let clickStartY = 0;

for (let i = 0; i < DATA.length; i++) {{
  const p = DATA[i];
  const color = base.clone().lerp(alt, p.t);

  const material = new THREE.MeshStandardMaterial({{
    color: color,
    emissive: color,
    emissiveIntensity: 0.8,
    roughness: 0.4,
    metalness: 0.1
  }});

  const m = new THREE.Mesh(sphereGeometry, material);
  m.position.set(p.x, p.y, p.z);
  scene.add(m);
  pointMeshes.push(m);
}}

// Zoom-to-point on zoom-in
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const crossEps = 0.035;
let zoomSide = 1;
const zoomAxis = new THREE.Vector3(0, 0, 1);

function getZoomAnchor(clientX, clientY) {{
  const rect = renderer.domElement.getBoundingClientRect();

  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(pointer, camera);

  const pointHits = raycaster.intersectObjects(pointMeshes, false);
  if (pointHits.length > 0) return pointHits[0].point.clone();

  const wireHits = raycaster.intersectObject(wire, false);
  if (wireHits.length > 0) return wireHits[0].point.clone();

  const viewNormal = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  const viewPlane = new THREE.Plane().setFromNormalAndCoplanarPoint(viewNormal, controls.target);
  const anchor = new THREE.Vector3();

  if (raycaster.ray.intersectPlane(viewPlane, anchor)) return anchor;

  return controls.target.clone();
}}

function getPointHit(clientX, clientY) {{
  const rect = renderer.domElement.getBoundingClientRect();

  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(pointer, camera);
  const pointHits = raycaster.intersectObjects(pointMeshes, false);
  return pointHits.length > 0 ? pointHits[0] : null;
}}

renderer.domElement.addEventListener("pointerdown", (e) => {{
  if (e.button !== 0) return;
  clickStartX = e.clientX;
  clickStartY = e.clientY;
}});

renderer.domElement.addEventListener("pointerup", (e) => {{
  if (e.button !== 0) return;

  const dx = e.clientX - clickStartX;
  const dy = e.clientY - clickStartY;
  if ((dx * dx + dy * dy) > 16) return;

  const hit = getPointHit(e.clientX, e.clientY);
  if (!hit) return;

  const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
  controls.target.copy(hit.object.position);
  camera.position.copy(controls.target).add(offset);
  zoomAxis.copy(offset).normalize();
  zoomSide = 1;
  controls.update();
}});

renderer.domElement.addEventListener("wheel", (e) => {{
  e.preventDefault();

  const zoomingOut = e.deltaY > 0;
  const factor = zoomingOut ? 1.08 : 0.92;

  const camToTarget = new THREE.Vector3().subVectors(camera.position, controls.target);
  const dist = camToTarget.length();
  if (dist > crossEps) {{
    zoomAxis.copy(camToTarget).normalize();
  }}

  let nextDist = dist * factor;
  nextDist = Math.max(controls.minDistance, Math.min(controls.maxDistance, nextDist));

  if (Math.abs(nextDist - dist) < 1e-9) return;

  if (!zoomingOut && dist <= crossEps) {{
    zoomSide *= -1;
  }}

  camera.near = nextDist < 0.2 ? 0.001 : 0.05;
  camera.updateProjectionMatrix();

  if (zoomingOut) {{
    camera.position.copy(controls.target).addScaledVector(zoomAxis, zoomSide * nextDist);
    controls.update();
    return;
  }}

  const anchor = getZoomAnchor(e.clientX, e.clientY);
  const ratio = nextDist / dist;

  const newTarget = anchor.clone().add(
    controls.target.clone().sub(anchor).multiplyScalar(ratio)
  );

  controls.target.copy(newTarget);
  camera.position.copy(newTarget).addScaledVector(zoomAxis, zoomSide * nextDist);
  controls.update();
}}, {{ passive: false }});

function resize() {{
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}

window.addEventListener("resize", resize);
resize();

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

animate();
</script>

</body>
</html>
"""

out_path = BASE_DIR / "umap_sphere_300.html"
with open(out_path, "w", encoding="utf-8") as f:
  f.write(html)

print(f"Saved: {out_path}")
