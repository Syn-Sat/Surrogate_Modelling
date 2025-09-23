import os
import xml.etree.ElementTree as ET
import pandas as pd

# --- Utility: Strip XML namespace ---
def strip_namespace(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag

# --- Parse ThermoML XMLs ---
def parse_thermoml_to_csv(xml_dir, output_csv="thermoml_vapor_pressure.csv"):
    records = []

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
                                                    "VapourPressure_kPa": float(val),
                                                    "Temperature_K": float(temp_val) if temp_val else None
                                                })
                                            except ValueError:
                                                continue

            except ET.ParseError as e:
                print(f"Skipping invalid XML file {file}: {e}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

    # Save to CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(records)} records to {output_csv}")
        return df
    else:
        print("\nNo vapor pressure records found.")
        return pd.DataFrame()

# --- Run Parser ---
if __name__ == "__main__":
    data_dir = "ThermoML.v2020-09-30"  # folder with ThermoML XML files
    parse_thermoml_to_csv(data_dir, "thermoml_vapor_pressure.csv")