# Tableau de bord churn télécom

Application Streamlit et API FastAPI pour estimer la probabilité de résiliation à partir du jeu de données `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Installation

Python 3.10 ou plus récent est requis.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Entraînement

À la racine du projet, avec le CSV présent :

```powershell
python train_model.py
```

Le script compare une régression logistique et XGBoost sur un jeu de validation selon le ROC-AUC, évalue le modèle retenu sur le jeu de test, puis met à jour les fichiers dans `models/`. Il faut réentraîner après toute modification du prétraitement ou des dépendances majeures.

## Lancement du tableau de bord

```powershell
streamlit run app.py
```

## Lancement de l'API

```powershell
uvicorn main:app --reload
```

La documentation interactive est disponible sur `/docs`. `POST /predict` accepte les champs client indiqués dans le schéma OpenAPI. `total_charges` est facultatif : s'il est absent, l'API utilise `tenure × monthly_charges` et indique cette approximation dans `total_charges_estimated`.

## Notes de lecture des résultats

Le revenu annuel exposé dans le tableau de bord est une approximation pondérée par la probabilité de churn (`facture mensuelle × 12 × probabilité`). Le coût annuel de remise et le revenu annuel après remise sont affichés séparément ; le revenu après remise suppose que le client reste abonné et ne constitue pas un bénéfice net. Les scores sont des estimations statistiques, pas des garanties de résiliation.
