import json
import random
import webbrowser
import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / 'moldash.html'
CSV_PATH = BASE_DIR / 'df_candidates.csv'

df = pd.read_csv(CSV_PATH)

selected_row = df.sample(1).iloc[0]
smiles = selected_row['ISOSMILES']
mol = Chem.MolFromSmiles(smiles)
mol_name = selected_row.get('SERIES')

mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.UFFOptimizeMolecule(mol)

mol_block = Chem.MolToMolBlock(mol) + '\n$$$$\n'

props = [
    ('Molecular weight (Da)', float(Descriptors.MolWt(mol))),
    ('cLogP (Crippen)', float(Crippen.MolLogP(mol))),
    ('TPSA (Å²)', float(rdMolDescriptors.CalcTPSA(mol))),
    ('H-bond donors', int(Lipinski.NumHDonors(mol))),
    ('H-bond acceptors', int(Lipinski.NumHAcceptors(mol))),
    ('Rotatable bonds', int(Lipinski.NumRotatableBonds(mol))),
    ('Ring count', int(rdMolDescriptors.CalcNumRings(mol))),
    ('Aromatic rings', int(rdMolDescriptors.CalcNumAromaticRings(mol))),
    ('Heavy atoms', int(mol.GetNumHeavyAtoms())),
    ('Fraction Csp3', float(rdMolDescriptors.CalcFractionCSP3(mol)))
]
properties = [{'name': k, 'value': v} for k, v in props]


ENDPOINTS = {
    # ----------------
    # Medicinal Chemistry
    # ----------------
    'QED': {
        'label': 'QED', 'category': 'Medicinal Chemistry', 'kind': 'classification',
    },
    'Lipinski': {
        'label': 'Lipinski Rules', 'category': 'Medicinal Chemistry', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 4.0},
        'acceptance': {'min': 0.0, 'max': 1.0},
    },
    'SAscore': {
        'label': 'SA_score',
        'category': 'Medicinal Chemistry',
        'kind': 'continuous',
        'domain': {'min': 1.0, 'max': 10.0},
        'acceptance': {'min': 1.0, 'max': 6.0},
    },
    # ----------------
    # Physicochemical Properties
    # ----------------
    'molecular_weight': {
        'label': 'MW', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 1000.0},
        'acceptance': {'min': 100.0, 'max': 600.0},
    },
    'logP': {
        'label': 'logP', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': -2.0, 'max': 8.0},
        'acceptance': {'min': 0.0, 'max': 5.0},
    },
    'Solubility_AqSolDB': {
        'label': 'logS', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': -12.0, 'max': 2.0},
        'acceptance': {'min': 1e-4},
    },
    'hydrogen_bond_acceptors': {
        'label': 'nHA', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 20.0},
        'acceptance': {'min': 0.0, 'max': 12.0},
    },
    'hydrogen_bond_donors': {
        'label': 'nHD', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 10.0},
        'acceptance': {'min': 0.0, 'max': 7.0},
    },
    'stereo_centers': {
        'label': 'Stereo Centers', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 10.0},
        'acceptance': {'min': 0.0, 'max': 2.0},
    },
    'tpsa': {
        'label': 'TPSA', 'category': 'Physicochemical Properties', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 300.0},
        'acceptance': {'min': 0.0, 'max': 140.0},
    },

    # -----------
    # Absorption
    # -----------
    'Caco2_Wang': {
        'label': 'Caco-2 Perm.', 'category': 'Absorption', 'kind': 'continuous',
        'domain': {'min': -8.0, 'max': 1.0},
        'acceptance': {'min': -5.0},
    },
    'PAMPA_NCATS': {
        'label': 'PAMPA', 'category': 'Absorption', 'kind': 'classification',
    },
    'Pgp_Broccatelli': {
        'label': 'Pgp inhib.', 'category': 'Absorption', 'kind': 'classification',
    },
    'HIA_Hou': {
        'label': 'HIA', 'category': 'Absorption', 'kind': 'classification',
    },
    'Bioavailability_Ma': {
        'label': 'F50%', 'category': 'Absorption', 'kind': 'classification',
    },
    'HydrationFreeEnergy_FreeSolv': {
        'label': 'Hydration Free Energy', 'category': 'Absorption', 'kind': 'continuous',
        'domain': {'min': -30.0, 'max': 5.0},
        'acceptance': {'min': -10.0},
    },
    'Lipophilicity_AstraZeneca': {
        'label': 'Lipophilicity', 'category': 'Absorption', 'kind': 'continuous',
        'domain': {'min': -2.0, 'max': 8.0},
        'acceptance': {'min': 0.0, 'max': 3.0},
    },

    # ------------
    # Distribution
    # ------------
    'BBB_Martins': {
        'label': 'BBB', 'category': 'Distribution', 'kind': 'classification',
    },
    'PPBR_AZ': {
        'label': 'PPB', 'category': 'Distribution', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 100.0},
        'acceptance': {'max': 90.0},
    },
    'VDss_Lombardo': {
        'label': 'VDss', 'category': 'Distribution', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 50.0},
        'acceptance': {'min': 0.04, 'max': 20.0},
    },

    # ----------
    # Metabolism
    # ----------
    'CYP1A2_Veith': {'label': 'CYP1A2 inhib.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP2C9_Veith': {'label': 'CYP2C9 inhib.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP2C19_Veith': {'label': 'CYP2C19 inhib.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP2D6_Veith': {'label': 'CYP2D6 inhib.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP3A4_Veith': {'label': 'CYP3A4 inhib.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP2C9_Substrate_CarbonMangels': {'label': 'CYP2C9 substr.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP2D6_Substrate_CarbonMangels': {'label': 'CYP2D6 substr.', 'category': 'Metabolism', 'kind': 'classification'},
    'CYP3A4_Substrate_CarbonMangels': {'label': 'CYP3A4 substr.', 'category': 'Metabolism', 'kind': 'classification'},

    # --------
    # Toxicity
    # --------
    'LD50_Zhu': {
        'label': 'LD50', 'category': 'Toxicity', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 5.0},
        'acceptance': {'min': 1.5},
    },
    'AMES': {'label': 'AMES Toxicity', 'category': 'Toxicity', 'kind': 'classification'},
    'DILI': {'label': 'DILI', 'category': 'Toxicity', 'kind': 'classification'},
    'hERG': {'label': 'hERG blockers', 'category': 'Toxicity', 'kind': 'classification'},
    'Carcinogens_Lagunin': {'label': 'Carcinogenicity', 'category': 'Toxicity', 'kind': 'classification'},
    'ClinTox': {'label': 'ClinTox', 'category': 'Toxicity', 'kind': 'classification'},
    'Skin_Reaction': {'label': 'Skin Reaction', 'category': 'Toxicity', 'kind': 'classification'},

    # Tox21 Pathways
    'NR-AR-LBD': {'label': 'NR-AR-LBD', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'NR-AR': {'label': 'NR-AR', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'NR-AhR': {'label': 'NR-AhR', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'NR-Aromatase': {'label': 'NR-Aromatase', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'NR-ER-LBD': {'label': 'NR-ER-LBD', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'NR-ER': {'label': 'NR-ER', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'NR-PPAR-gamma': {'label': 'NR-PPAR-gamma', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'SR-ARE': {'label': 'SR-ARE', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'SR-ATAD5': {'label': 'SR-ATAD5', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'SR-HSE': {'label': 'SR-HSE', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'SR-MMP': {'label': 'SR-MMP', 'category': 'Tox21 Pathways', 'kind': 'classification'},
    'SR-p53': {'label': 'SR-p53', 'category': 'Tox21 Pathways', 'kind': 'classification'},

    # ---------
    # Excretion
    # ---------
    'Half_Life_Obach': {
        'label': 'Half-life', 'category': 'Excretion', 'kind': 'continuous',
        'domain': {'min': -3.0, 'max': 3.0},
        'acceptance': {'min': 2.0, 'max': 24.0},
    },
    'Clearance_Hepatocyte_AZ': {
        'label': 'Hepatocyte Clear.', 'category': 'Excretion', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 200.0},
        'acceptance': {'max': 50.0},
    },
    'Clearance_Microsome_AZ': {
        'label': 'Microsomal Clear.', 'category': 'Excretion', 'kind': 'continuous',
        'domain': {'min': 0.0, 'max': 200.0},
        'acceptance': {'max': 50.0},
    },
}

# Build endpoints list from ENDPOINTS only
endpoints = []
for old_name, meta in ENDPOINTS.items():
    new_name = meta['label']
    if old_name not in df.columns:
        continue

    percentile_col = f'{old_name}_drugbank_approved_percentile'
    if percentile_col not in df.columns:
        continue

    raw_value = pd.to_numeric(selected_row[old_name], errors='coerce')
    percentile = pd.to_numeric(selected_row[percentile_col], errors='coerce')
    if pd.isna(raw_value) or pd.isna(percentile):
        continue

    endpoint_data = {
        'id': new_name,
        'kind': meta['kind'],
        'category': meta['category'],
        'percentile': float(percentile),
    }

    if meta['kind'] == 'classification':
        endpoint_data['probability'] = float(raw_value)
    else:
        endpoint_data['value'] = float(raw_value)

        domain_rule = meta.get('domain')
        if domain_rule is not None:
            endpoint_data['domain'] = domain_rule

        acceptance_rule = meta.get('acceptance')
        if acceptance_rule is not None:
            endpoint_data['acceptance'] = acceptance_rule

    endpoints.append(endpoint_data)

data = {
    'viewer': {
        'background': '#070A10',
        'stickRadius': 0.08,      # thinner bonds
        'sphereScale': 0.35,
        'zoomOut': 0.90
    },
    'molecule': {
        'name': str(mol_name),
        'smiles': smiles,
        'atomCount': int(mol.GetNumAtoms())
    },
    'sdf': mol_block,
    'properties': properties,
    'endpoints': endpoints
}

data_json = json.dumps(data, ensure_ascii=False)


# -------------------------
# HTML
# -------------------------

html_content = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Molecule viewer</title>

<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="admet_style.css">

<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
</head>

<body>

<div id="viewer"></div>

<div id="mol-title">
  <div class="mol-name">{data['molecule']['name']}</div>
  <div class="mol-meta">{data['molecule']['atomCount']} atoms · small molecule</div>
</div>

<div id="property-panel">
  <div class="card">
    <div class="title">PROPERTIES</div>
    <table id="props"></table>
  </div>
</div>

<div id="admet-panel">
  <div class="card admet-card">
    <div class="title">ADMET</div>
    <div id="admet"></div>
  </div>
</div>

<script>
  window.MOLDASH = {data_json};
</script>

<script>
  const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: window.MOLDASH.viewer.background }});
  const sdf = window.MOLDASH.sdf;

  viewer.addModel(sdf, "sdf");
  viewer.setStyle({{}}, {{ stick: {{ radius: 0.09 }}, sphere: {{ scale: 0.15 }} }});
  viewer.zoomTo();
  viewer.zoom(window.MOLDASH.viewer.zoomOut);
  viewer.translate(-80, -20);
  viewer.render();

  function fmtValue(x) {{
    if (typeof x === "number") {{
      if (Number.isInteger(x)) return String(x);
      return x.toFixed(3);
    }}
    return String(x);
  }}

  const table = document.getElementById("props");
  table.innerHTML = window.MOLDASH.properties.map(p => `
    <tr>
      <td>${{p.name}}</td>
      <td>${{fmtValue(p.value)}}</td>
    </tr>
  `).join("");
</script>

<script src="admet.js"></script>

</body>
</html>
"""

HTML_PATH.write_text(html_content, encoding='utf-8')

HTML_PATH.write_text(html_content, encoding="utf-8")

print(f"Random SMILES used: {smiles}")
print(f"Viewer saved to: {HTML_PATH}")