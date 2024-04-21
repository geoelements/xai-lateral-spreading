# XAI - Lateral Spreading
This project investigates the application of eXplainable AI (XAI) techniques on predictive machine learning models for lateral spreading phenomena. We have developed multiple XGBoost models using a dataset sourced from [Durante and Rathje (2022)](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-2998v2). The repository provides resources for data preprocessing, model training, and interpretation using SHAP (SHapley Additive exPlanations) explainers.

## Folder Structure
**`data` Folder**: Contains both the original and processed datasets. The original dataset, derived from [Durante and Rathje (2021)](https://doi.org/10.1177/87552930211004613), comprises 6,500 datapoints from Christchurch, New Zealand, pertaining to the 2011 Christchurch Earthquake. It includes various features such as geometry features, event-specific features like groundwater depth (GWD) and peak ground acceleration (PGA), CPT (cone penetration test) related features, and binary indicators for lateral spreading. Refer to Table 1 for a breakdown of features used in each model.


**Table 1.** Summary of features used in each XGBoost model.
|Model|L<br>(km)|GWD<br>(m)|PGA<br>(g)|Elevation<br>(m)|Slope<br>(%)|I<sub>c</sub><br>(med)|I<sub>c</sub><br>(std)|q<sub>c1Ncs</sub><br>(med)|q<sub>c1Ncs</sub><br>(std)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|A|✓|✓|✓|✓|✓|O|O|O|O|
|B|✓|✓|✓|✓|✓|✓|✓|✓|✓|
|C|✓|✓|✓|✓|O|✓|✓|O|O|
<br>

**`model_development` Folder**: Includes Jupyter notebooks for data preprocessing (_`data_preprocessing.ipynb`_) and XGBoost model training (_`xgb_training.ipynb`_). The data preprocessing notebook loads the dataset, performs data splitting, and feature selection according to Table 1, saving the processed data as pickle files (_`data_x.pkl`_) in **`data`** folder. The XGBoost training notebook demonstrates the model training process and saves the trained models as pickle files (_`opt_XGB_X.pkl`_) in the **`xgb_models`** folder.
<br>

**`model_usage` Folder**: Contains Jupyter notebooks (_`shap_explainer_X.ipynb`_) for generating SHAP explanations for each XGBoost model. These notebooks load the trained models and corresponding data to create SHAP visualizations.
<br>

**`xgb_models` Folder**: Stores the trained XGBoost models developed from different datasets in the **`data`** folder.

## References
Durante, M. G. and Rathje, E. (2022). Machine learning models for the evaluation of the lateral spreading hazard in the Avon river area following the 2011 Christchurch earthquake. doi:10.17603/DS2-3ZDJ-4937

Durante, M. G. and Rathje, E. M. (2021). An exploration of the use of machine learning to predict lateral spreading. Earthquake Spectra 37, 2288–2314. doi:10.1177/87552930211004613

## Citation
```
@software{krishna_kumar_2024_11003110,
  author       = {Krishna Kumar and
                  Cheng-Hsi Hsiao},
  title        = {geoelements/xai-lateral-spreading: v1.0.1},
  month        = apr,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.11003110},
  url          = {https://doi.org/10.5281/zenodo.11003110}
}
```
