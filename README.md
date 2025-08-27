# Surrogate Modelling For Thermodynamic Properties Using ML

## Tool Stack
Data/featurization: numpy,pandas, RDKit,matplotlib,CoolProp

Models: scikit-learn (GPR/GBDT/SVR), PyTorch/JAX (neural EOS + autodiff).

Uncertainty: GPR, Deep Ensembles, MC-Dropout.

Optimization: optuna for hyperparams.

## BLUE PRINT

# Surrogate Modelling of Vapor Pressure for Alcohols (Machine Learning Approach)

This guide outlines the **first 15 steps** to create a surrogate model for predicting vapor pressure (\(p_\mathrm{sat}(T)\)) of alcohols using Machine Learning.

---

### 1. Define the Property
- Target property: Vapor pressure of pure-component alcohols.
- Model \(\log_{10}(p_\mathrm{sat}/\text{bar})\) as a function of temperature and molecular descriptors.

---

### 2. Choose Alcohol Dataset
- Include compounds: Methanol, Ethanol, Propanol, Butanol, Pentanol, Hexanol.
- Temperature range: 250–500 K (up to near-critical point).
- Data sources: **NIST ThermoML**, DIPPR 801, or REFPROP (if licensed).

---

### 3. Data Schema
Create a CSV with columns:

| Column       | Description |
|--------------|-------------|
| compound     | Alcohol name (e.g., Ethanol) |
| smiles       | SMILES string for RDKit features |
| T_K          | Temperature in Kelvin |
| p_sat_bar    | Vapor pressure in bar |
| Tc_K         | Critical temperature in K |
| Pc_bar       | Critical pressure in bar |
| omega        | Acentric factor |
| chain_len    | Carbon chain length |

---

### 4. Data Pre-Processing
- Remove missing or unphysical data points.
- Convert pressure to **bar** and ensure temperature is in **Kelvin**.
- Add derived features: \(1/T\), reduced temperature \(T/T_c\).

---

### 5. Feature Selection
Minimum useful features:
- Inverse temperature `1/T`
- Critical temperature `Tc_K`
- Critical pressure `Pc_bar`
- Acentric factor `omega`
- Carbon chain length `chain_len`

Optional features:
- RDKit-derived descriptors (polar surface area, H-bond donors, etc.).

---

### 6. Baseline Model (Antoine Equation)
Fit Antoine equation to each alcohol:
\[
\log_{10} p_\text{sat} = A - \frac{B}{T + C}
\]
- Use `scipy.optimize.curve_fit` to obtain (A, B, C) for each compound.
- Provides analytic baseline to compare with ML model.

---

### 7. Machine Learning Baseline
- Use **XGBoost** or **LightGBM** for tabular regression.
- Train on all compounds simultaneously.
- Inputs: selected features (Step 5).
- Target: \(\log_{10}(p_\mathrm{sat}/\text{bar})\).

---

### 8. Enforce Monotonicity
- Vapor pressure should increase with temperature.
- Since model uses `1/T`, impose monotone constraint:
  - **Decreasing** in `1/T` ⇒ **Increasing** in `T`.

---

### 9. Cross-Validation Strategy
- Use **GroupKFold** by `compound` to test generalization to unseen alcohols.
- Prevents data leakage across temperature points of same compound.

---

### 10. Metrics
Evaluate:
- MAE and RMSE on `log10(p/bar)`
- MAE and RMSE on `p/bar` (after back-transform)
- Compare XGBoost vs Antoine per compound.

---

### 11. Physics-Informed Checks
- Ensure predictions are monotonically increasing with T.
- Check extrapolation near `Tc_K` → vapor pressure approaches `Pc_bar`.

---

### 12. Optional Neural Baseline
- Use a small MLP in PyTorch with monotone layers.
- Train on same features and loss functions as XGBoost.
- Autodiff available for future Helmholtz-based models.

---

### 13. Uncertainty Quantification
- Use Deep Ensembles or Monte Carlo Dropout.
- Identify regions where surrogate confidence is low → collect more data.

---

### 14. Visualization
- Plot predicted vs true vapor pressure.
- Plot residuals vs T and vs compound index.
- Plot Antoine vs ML predictions for selected alcohols.

---

### 15. Repository Setup
- Use GitHub for version control.
- Folder structure:


