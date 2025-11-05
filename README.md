# Surrogate Modelling For Thermodynamic Properties Using ML

## Tool Stack
- Data/featurization: numpy,pandas, RDKit,matplotlib,scipy,seaborn
- Models: scikit-learn (RandomForestRegressor), XGBoost.
- Optimization: pandarallel for multithreading.

## BLUE PRINT

# Surrogate Modelling of Vapor Pressure for Hydrocarbons (Machine Learning Approach)

This guide outlines the steps to create a surrogate model for predicting vapor pressure (\(p_\mathrm{sat}(T)\)) of hydrocarbons using Machine Learning.

---

### 1. Define the Property
- Target property: Vapor pressure of hydrocarbons (H,C,N atoms).
- Model Vapour Pressure (kPa) as a function of temperature and molecular descriptors.

---

### 2. Choose Dataset
- Temperature range: 275–375 K (up to near-critical point).
- Data sources: **NIST ThermoML**.

---

### 3. Data Schema
Create a CSV with columns:

| Column                | Description                      |
|-----------------------|----------------------------------|
| SMILES                | SMILES string for RDKit features |
| InChI                 | InChI string for RDKit features  |
| T_K                   | Temperature in Kelvin            |
| VapourPressure_kPa    | Vapor pressure in kPa            |
---

### 4. Data Pre-Processing
- Remove missing or unphysical data points, NaN, etc.
- Convert pressure to **kPa** and ensure temperature is in **Kelvin**.

---

### 5. Feature Selection
- RDKit-derived descriptors (polar surface area, H-bond donors, etc.).

---


### 6. Machine Learning Baseline
- Use **XGBoost** and **RandomForestRegressor** for tabular regression.
- Train on all compounds simultaneously.
- Inputs: selected features (Step 5).
- Target: \(\log_{10}(p_\mathrm{sat}/\text{bar})\).

---

### 7. Metrics
Evaluate:
- MAE and MAPE on `vapour pressure`
- R^2 score

---

### 8. Visualization
- Plot predicted vs true vapor pressure.
- Plot skewness of vp data



