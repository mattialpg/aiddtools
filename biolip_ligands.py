# https://aideepmed.com/BioLiP/ligand_list
# This list of potential artifact (non-biological) ligands includes compounds frequently used for 
# crystallization additives and protein purification buffers [10.1093/nar/gkad630].
# In addition, the dataset was extended with all inorganic ligands, compounds not containing carbon
# and those composed exclusively of C and H.

import pandas as pd
import yaml
import requests
from rdkit import Chem
from rdkit.Chem import PandasTools

df_cci = pd.read_csv('rcsb_fragmentation/data/cci_database.csv')
df_cci = df_cci.drop_duplicates('Ligand ID').dropna(subset=['Ligand SMILES'])
PandasTools.AddMoleculeColumnToFrame(df_cci, 'Ligand SMILES')

df_cci = df_cci[df_cci['ROMol'].map(lambda m: (m is not None and 
    ((6 not in {a.GetAtomicNum() for a in m.GetAtoms()}) or
    ({a.GetAtomicNum() for a in m.GetAtoms()} == {1, 6}))))]

df_cci = df_cci[['Ligand ID', 'Ligand Name', 'Ligand SMILES', 'InChI']]
df_cci = df_cci.rename(columns={'Ligand ID': 'CCI', 'Ligand Name': 'NAME',
    'Ligand SMILES': 'ISOSMILES', 'InChI': 'InChI'})

with open('/home/mattia/aiddtools/biolip_list.yaml') as f:
    biolip_data = yaml.safe_load(f)
    biolip_list = biolip_data.get("biolip_list", biolip_data)

aux = [x for x in biolip_list if x not in df_cci['CCI'].tolist()]
props = []
for chem_id in biolip_list:
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{chem_id}"

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        name = data['chem_comp']['name']
        inchi = data['rcsb_chem_comp_descriptor']['in_ch_i']
        smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
        props.append([chem_id, name, smiles, inchi])
    except:
        pass

df = pd.DataFrame(props,
    columns=["CCI", "NAME", "ISOSMILES", "InChI"])

df_biolip = pd.concat([df_cci, df], ignore_index=True).drop_duplicates('CCI')
df_biolip['NAME'] = df_biolip['NAME'].astype(str).str.upper()

df_biolip.to_csv('/home/mattia/aiddtools/biolip_ligands.csv', index=False)