# Surrogate Model Pipeline for Vapour Pressure of Alcohols (XML-based, InChI support)
# Updated: Handles nested Compound -> Sample -> MeasuredProperties structure in XML files

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Step 1: Load XML Files ---
def strip_namespace(tag):
    """Remove namespace from XML tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag

def load_xml_data(xml_dir):
    records = []
    print(f"Scanning directory: {xml_dir}")

    for root_dir, _, files in os.walk(xml_dir):
        for file in files:
            if not file.lower().endswith(".xml"):
                continue

            filepath = os.path.join(root_dir, file)
            print(f"Processing file: {os.path.basename(filepath)}")

            try:
                tree = ET.parse(filepath)
                root = tree.getroot()

                # --- Strip namespaces ---
                for elem in root.iter():
                    elem.tag = strip_namespace(elem.tag)

                # --- Map nOrgNum -> InChI ---
                orgnum_to_inchi = {}
                for compound in root.findall(".//Compound"):
                    inchi = compound.findtext("sStandardInChI")
                    nOrgNum = compound.findtext("RegNum/nOrgNum")
                    if inchi and nOrgNum:
                        orgnum_to_inchi[nOrgNum] = inchi

                # --- Loop through experimental data ---
                for pom in root.findall(".//PureOrMixtureData"):
                    comp_nums = [
                        c.findtext("RegNum/nOrgNum")
                        for c in pom.findall("Component")
                    ]
                    inchis = [orgnum_to_inchi.get(cn) for cn in comp_nums if cn in orgnum_to_inchi]

                    for prop in pom.findall("Property"):
                        prop_name = prop.findtext(".//ePropName")
                        if not prop_name:
                            continue

                        # Focus on vapor-pressure-like properties
                        if "Vapor" in prop_name or "Vapour" in prop_name:
                            prop_number = prop.findtext("nPropNumber")

                            for numval in pom.findall("NumValues"):
                                # Match PropertyValue with correct prop_number
                                for propval in numval.findall("PropertyValue"):
                                    if propval.findtext("nPropNumber") == prop_number:
                                        val = propval.findtext("nPropValue")
                                        temp_val = None
                                        var_elem = numval.find("VariableValue/nVarValue")
                                        if var_elem is not None:
                                            temp_val = var_elem.text

                                        if val and inchis:
                                            try:
                                                records.append({
                                                    "InChI": inchis[0],  # first component for now
                                                    "VapourPressure": float(val),
                                                    "Temperature_K": float(temp_val) if temp_val else None
                                                })
                                            except ValueError:
                                                continue

            except ET.ParseError as e:
                print(f"Skipping invalid XML file {file}: {e}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if records:
        print(f"Loaded {len(records)} records from XML files.")
        return pd.DataFrame(records)
    else:
        print("No records found in XML files.")
        return pd.DataFrame()

# --- Step 2: Compute Molecular Descriptors from InChI ---
def compute_descriptors(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'MolLogP': Descriptors.MolLogP(mol)
        }
    else:
        return {
            'MolWt': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'MolLogP': np.nan
        }

# --- Step 3: Filter Alcohols ---
def is_alkaneetc(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'C' or atom.GetSymbol() != 'H':
                return False
    return True

# --- Step 4: Build and Evaluate Linear Regression Model ---
def train_linear_model(df):
    desc_df = df['InChI'].apply(compute_descriptors).apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1).dropna()

    features = ['Temperature_K', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP']
    df = df[df['VapourPressure'] > 0]
    X = df[features]
    y = np.log(df['VapourPressure'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression Model Evaluation:")
    print("MAE (kPa):", mae/1000)
    print("R2 Score:", r2)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    data_dir = "ThermoML.v2020-09-30"  # folder containing XML files

    df = load_xml_data(data_dir)
    if not df.empty:
        df = df[df['InChI'].apply(is_alkaneetc)]
        print(f"Filtered {len(df)} alcohol records with vapor pressure data.")
        if not df.empty:
            trained_model = train_linear_model(df)
        else:
            print("No alcohol vapor pressure data found after filtering.")
    else:
        print("No XML data found.")