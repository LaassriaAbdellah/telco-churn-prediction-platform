import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from inference import build_client_frame

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard Churn Client",
    page_icon="📊",
    layout="wide"
)

# Chargement des objets sauvegardés
@st.cache_resource
def load_assets():
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    columns = joblib.load("models/model_columns.joblib")
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        # Les variables numériques sont standardisées : zéro est leur moyenne.
        explainer = shap.LinearExplainer(model, np.zeros((1, len(columns))))
    return model, scaler, columns, explainer

try:
    model, scaler, model_columns, explainer = load_assets()
except Exception as e:
    st.error("Erreur lors du chargement des modèles. Assurez-vous d'avoir exécuté train_model.py.")
    st.stop()

# --- FONCTION DE GÉNÉRATION DU PDF ---
# --- FONCTION DE GÉNÉRATION DU PDF (CORRIGÉE) ---
def generate_pdf_report(client_info, churn_pct, revenu_expose, cout_remise, revenu_apres_remise, recommandation):
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(26, 82, 118)
    pdf.cell(0, 10, "Rapport d'Analyse Anti-Churn", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Genere le : {datetime.now().strftime('%d/%m/%Y a %H:%M')}", ln=True, align="C")
    pdf.ln(5)
    
    # Ligne de séparation
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Section 1 : Profil Client
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Profil du Client", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    # Utilisation du tiret '-' à la place du symbole puce '•'
    pdf.cell(90, 7, f"- Anciennete : {client_info['tenure']} mois", ln=False)
    pdf.cell(90, 7, f"- Facture Mensuelle : {client_info['monthly_charges']} $", ln=True)
    pdf.cell(90, 7, f"- Type de Contrat : {client_info['contract']}", ln=False)
    pdf.cell(90, 7, f"- Service Internet : {client_info['internet']}", ln=True)
    pdf.cell(90, 7, f"- Support Technique : {client_info['tech_support']}", ln=False)
    pdf.cell(90, 7, f"- Mode de Paiement : {client_info['payment']}", ln=True)
    pdf.ln(8)
    
    # Section 2 : Risque et Impact
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "2. Evaluation du Risque & ROI", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"- Probabilite de Resiliation (Churn) : {churn_pct}%", ln=True)
    pdf.cell(0, 7, f"- Revenu annuel estime expose au risque : {revenu_expose:,.2f} $", ln=True)
    pdf.cell(0, 7, f"- Cout annuel de la remise : {cout_remise:,.2f} $", ln=True)
    pdf.cell(0, 7, f"- Revenu annuel apres remise : {revenu_apres_remise:,.2f} $", ln=True)
    pdf.ln(8)
    
    # Section 3 : Recommandations Métier
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "3. Plan d'Action Recommande", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(40, 40, 40)
    
    # Nettoyage des caractères spéciaux dans la recommandation pour éviter tout blocage Unicode
    clean_reco = recommandation.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_reco)
    
    return bytes(pdf.output())

# Titre principal
st.title("📊 Prédiction du Churn & Rétention Client")
st.markdown("Utilisez cet outil pour évaluer le risque de résiliation d'un client en temps réel et adapter votre stratégie marketing.")

st.divider()

# Création des onglets
tab1, tab2, tab3 = st.tabs([
    "🔮 Évaluation Client", 
    "🔍 Explicabilité Client (SHAP)", 
    "📈 Importance Globale"
])

with tab1:
    col_input, col_result = st.columns([1, 1])

    # --- COLONNE DE GAUCHE : SAISIE DES DONNÉES ---
    with col_input:
        st.subheader("Saisie des données client")
        with st.form("client_analysis_form"):
            tenure = st.slider("Ancienneté (Mois)", 0, 72, 12)
            monthly_charges = st.slider("Facture Mensuelle ($)", 18.0, 120.0, 65.0)
            total_charges = st.number_input("Total réellement facturé à ce jour ($)", min_value=0.0, value=float(tenure * monthly_charges), step=10.0, help="La valeur proposée est une approximation ; remplacez-la par le total réel si vous l'avez.")
            contract = st.selectbox("Type de Contrat", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
            tech_support = st.selectbox("Support Technique", ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
            paperless = st.selectbox("Facturation en ligne", ["Yes", "No"])
            payment_method = st.selectbox("Méthode de paiement", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            taux_remise = st.slider(
                "Proposer une remise de fidélisation (%) :",
                min_value=5, max_value=30, value=15, step=5
            )
            analyze = st.form_submit_button("🚀 Calculer le Risque de Churn", use_container_width=True)

    # --- COLONNE DE DROITE : RÉSULTATS & CALCUL ROI ---
    with col_result:
        st.subheader("Analyse du Risque")
        
        # Après la première analyse, conserver l'affichage lors des autres reruns
        # Streamlit (changement d'onglet ou clic sur téléchargement).
        if analyze or st.session_state.get('has_predicted', False):
            # 1. Traitement des données d'entrée
            client_input = {'tenure': tenure, 'monthly_charges': monthly_charges, 'total_charges': total_charges,
                'contract': contract, 'internet_service': internet_service, 'tech_support': tech_support,
                'online_security': online_security, 'paperless_billing': paperless, 'payment_method': payment_method}
            input_df = build_client_frame(client_input, model_columns, scaler)
            
            st.session_state['current_input_df'] = input_df
            st.session_state['has_predicted'] = True
            
            # 2. Prédiction
            proba = model.predict_proba(input_df)[0][1]
            churn_pct = round(proba * 100, 1)
            
            # 3. Jauge Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_pct,
                number={'suffix': "%"},
                title={'text': "Probabilité de Résiliation"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 60], 'color': "#f1c40f"},
                        {'range': [60, 100], 'color': "#e74c3c"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. Message de recommandation
            if churn_pct >= 60:
                reco_text = f"Risque Élevé ({churn_pct}%) : Offrir immédiatement un engagement annuel avec remise."
                st.error(f"⚠️ **{reco_text}**")
            elif churn_pct >= 30:
                reco_text = f"Risque Modéré ({churn_pct}%) : Proposer une option de support technique ou surclassement."
                st.warning(f"⚡ **{reco_text}**")
            else:
                reco_text = f"Faible Risque ({churn_pct}%) : Client stable. Proposer des services additionnels."
                st.success(f"✅ **{reco_text}**")

            # 5. Impact Financier & Simulation ROI
            st.divider()
            st.markdown("### 💰 Impact Financier & Simulation de Rétention")

            revenu_expose = monthly_charges * 12 * proba
            clv_estimee = monthly_charges * tenure

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    label="Revenu annuel estimé exposé au risque",
                    value=f"{revenu_expose:,.2f} $",
                    delta=f"Probabilité de churn : {churn_pct}%",
                    delta_color="inverse"
                )

            with col_m2:
                st.metric(
                    label="Valeur Vie Client (CLV estimée)",
                    value=f"{clv_estimee:,.2f} $",
                    help="Calculé par : Charge mensuelle × Ancienneté (mois)"
                )

            st.markdown("#### 🔄 Simulateur d'Offre Anti-Churn")

            nouvelle_facture = monthly_charges * (1 - (taux_remise / 100))
            cout_promotion = (monthly_charges - nouvelle_facture) * 12
            revenu_apres_remise = nouvelle_facture * 12

            col_roi1, col_roi2 = st.columns(2)
            with col_roi1:
                st.metric(
                    label=f"Coût Annuel de la Remise ({taux_remise}%)",
                    value=f"{cout_promotion:,.2f} $",
                    delta=f"-{cout_promotion:,.2f} $",
                    delta_color="inverse"
                )

            with col_roi2:
                st.metric(
                    label="Revenu annuel après remise (si conservé)",
                    value=f"{revenu_apres_remise:,.2f} $",
                    delta=f"Remise de {taux_remise}%",
                    delta_color="normal"
                )

            # --- BOUTON DE TÉLÉCHARGEMENT PDF ---
            st.divider()
            client_data_pdf = {
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'contract': contract,
                'internet': internet_service,
                'tech_support': tech_support,
                'payment': payment_method
            }
            
            pdf_bytes = generate_pdf_report(
                client_info=client_data_pdf,
                churn_pct=churn_pct,
                revenu_expose=revenu_expose,
                cout_remise=cout_promotion,
                revenu_apres_remise=revenu_apres_remise,
                recommandation=reco_text
            )
            
            st.download_button(
                label="📄 Télécharger le Rapport PDF du Client",
                data=pdf_bytes,
                file_name=f"rapport_retention_client_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# --- ONGLET 2 : EXPLICABILITÉ INDIVIDUELLE (SHAP) ---
with tab2:
    st.subheader("🔍 Pourquoi ce client précis a-t-il ce score ? (Analyse SHAP)")
    if st.session_state.get('has_predicted', False):
        input_data = st.session_state['current_input_df']
        shap_values = explainer(input_data)
        
        fig_shap, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig_shap)
    else:
        st.info("👈 Veuillez d'abord cliquer sur **'Calculer le Risque de Churn'** dans le premier onglet.")

# --- ONGLET 3 : IMPORTANCE GLOBALE ---
with tab3:
    st.subheader("Top 10 des variables les plus utilisées par le modèle")
    st.caption("L'importance décrit le fonctionnement du modèle ; elle ne prouve pas une relation de cause à effet.")
    importances = (model.feature_importances_ if hasattr(model, 'feature_importances_')
                  else np.abs(model.coef_[0]))
    feature_imp = pd.DataFrame({
        'Feature': model_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    fig_bar = go.Figure(go.Bar(
        x=feature_imp['Importance'],
        y=feature_imp['Feature'],
        orientation='h'
    ))
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)
