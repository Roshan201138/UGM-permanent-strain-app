# UGM Permanent Strain Prediction and Calibration App

This repository contains a Streamlit application for calibration, prediction, and comparison of empirical and mechanistic permanent strain models for unbound granular materials under repeated loading.

## Main features

- Single-stage and multi-stage permanent strain prediction
- Time-hardening implementation for selected multi-stage models
- Automatic parameter calibration from measured permanent strain data
- Manual parameter input for prediction
- CSV and Excel input support
- Model comparison plots and downloadable figures
- Statistical performance indicators including R², RMSE, MAE, MSE, and MAPE

## Implemented model families

The app includes classical and recent permanent deformation models, including Barksdale, Hyde, Veverka, Lentz and Baladi, Khedr, Paute, Sweere, Hornych, Wolff and Visser, Rahman et al., Rahman and Erlingsson, Korkiala–Tanttu, Erlingsson and Rahman modified Korkiala–Tanttu, Huurman, Gidel, Chen, Tseng and Lytton, MEPDG/NCHRP, Chow, Lin, and Ooi models.

## Online Application
The application can be accessed online at:


## Running the app locally
If you prefer to run the app locally:

pip install -r requirements.txt
streamlit run app.py

## Notes

- Multi-stage models use an equivalent-cycle time-hardening formulation.
- The saturation-based Rahman model is implemented as an extension of the resilient-strain-based formulation.
- Users should verify input stress definitions and strain units before calibration or prediction.

## Developer

Mohammad Jawed Roshan
