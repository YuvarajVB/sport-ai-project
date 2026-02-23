"""
=============================================================================
SPORT SUITABILITY PREDICTION SYSTEM
Longitudinal Study on 200 School Children (6th–8th Standard)
=============================================================================
STEP: Post-Model Training — Full Streamlit Dashboard

Place this file in your sport_ai_project folder alongside:
    sport_prediction_model.joblib

Run:
    streamlit run dashboard.py
=============================================================================
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Sport Suitability Prediction System",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }

    .header-banner {
        background: linear-gradient(135deg, #0d2137 0%, #1565C0 100%);
        padding: 24px 30px; border-radius: 14px;
        margin-bottom: 20px;
    }
    .header-banner h1 { color: white; font-size: 1.85rem; margin: 0; }
    .header-banner p  { color: #90caf9; margin: 5px 0 0 0; font-size: 0.92rem; }

    .metric-card {
        background: white; border-radius: 10px;
        padding: 16px 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center; border-left: 5px solid #1565C0;
    }
    .metric-card h2 { font-size: 1.85rem; color: #0d2137; margin: 0; }
    .metric-card p  { color: #666; font-size: 0.8rem; margin: 4px 0 0 0; }

    .result-box {
        border-radius: 14px; padding: 26px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
    }
    .result-box .icon  { font-size: 3.5rem; }
    .result-box h2     { font-size: 2.3rem; margin: 6px 0; color: white; }
    .result-box .conf  { font-size: 1.2rem; color: rgba(255,255,255,0.92); }
    .result-box .prof  { font-size: 0.85rem; color: rgba(255,255,255,0.72); margin-top:6px; }

    .section-title {
        background: #0d2137; color: white;
        padding: 9px 18px; border-radius: 8px;
        font-size: 0.93rem; font-weight: 700;
        margin: 18px 0 10px 0;
    }

    .sport-bar-outer  { background: #e3eaf5; border-radius:8px; height:24px; margin:4px 0; }
    .sport-bar-inner  {
        height:24px; border-radius:8px;
        display:flex; align-items:center; padding-left:10px;
        color:white; font-size:0.8rem; font-weight:700; min-width:36px;
    }
    .sport-bar-label  { font-weight:600; font-size:0.88rem; color:#0d2137; margin-bottom:2px; }

    .compare-box {
        background: white; border-radius:10px;
        padding:14px 18px; margin-top:12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .info-box {
        background:#e8f4fd; border:1px solid #1565C0;
        border-radius:8px; padding:12px 16px;
        font-size:0.87rem; color:#0d2137;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0d2137 0%,#1a3a5c 100%);
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span { color: white !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
SPORT_COLORS = {
    'Football':   '#e53935',
    'Volleyball': '#8e24aa',
    'Swimming':   '#1565C0',
    'Basketball': '#ef6c00',
    'Athletics':  '#2e7d32',
    'Gymnastics': '#f9a825',
}
SPORT_ICONS = {
    'Football':   '⚽',
    'Volleyball': '🏐',
    'Swimming':   '🏊',
    'Basketball': '🏀',
    'Athletics':  '🏃',
    'Gymnastics': '🤸',
}
SPORT_PROFILES = {
    'Football':   'Moderate height · High strength & gait · Good endurance',
    'Volleyball': 'Tall · High flexibility & jumping ability · Good gait',
    'Swimming':   'Very high flexibility · Upper body strength · Any height',
    'Basketball': 'Very tall · High strength & gait · Fast walking speed',
    'Athletics':  'Light body · Highest gait score · Excellent running',
    'Gymnastics': 'Short & lightest · Exceptional flexibility · High strength-to-weight',
}
ALL_SPORTS = ['Football', 'Volleyball', 'Swimming', 'Basketball', 'Athletics', 'Gymnastics']

# =============================================================================
# FEATURE ENGINEERING  —  must match training exactly
# =============================================================================
def engineer_features(df):
    df = df.copy()
    df['athleticism_index'] = (
        df['strength_score']    * 0.30 +
        df['flexibility_score'] * 0.20 +
        df['gait_score']        * 0.25 +
        df['walking_score']     * 0.25
    )
    df['power_to_weight']     = df['strength_score'] / df['weight_kg']
    df['mobility_score']      = (df['gait_score'] + df['walking_score'] + df['flexibility_score']) / 3
    df['height_weight_ratio'] = df['height_cm'] / df['weight_kg']
    df['bmi_category']        = pd.cut(
        df['bmi'], bins=[0, 18.5, 23, 27.5, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    return df

FEATURE_COLS = [
    'height_cm', 'weight_kg', 'bmi', 'strength_score', 'flexibility_score',
    'gait_score', 'walking_score', 'age', 'gender',
    'athleticism_index', 'power_to_weight',
    'mobility_score', 'height_weight_ratio', 'bmi_category'
]

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    for path in [
        'sport_prediction_model.joblib',
        'models/sport_prediction_model.joblib',
        '../models/sport_prediction_model.joblib',
        'notebooks/sport_prediction_model.joblib',
    ]:
        if os.path.exists(path):
            d = joblib.load(path)
            return d['model'], d['scaler'], d['label_encoder'], path
    return None, None, None, None

# =============================================================================
# PREDICT
# =============================================================================
def predict_student(model, scaler, label_encoder, data: dict):
    df         = engineer_features(pd.DataFrame([data]))
    X          = scaler.transform(df[FEATURE_COLS])
    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    sport      = label_encoder.inverse_transform([pred_enc])[0]
    classes    = label_encoder.classes_
    top3_idx   = np.argsort(pred_proba)[::-1][:3]
    return {
        'primary':    sport,
        'confidence': round(float(pred_proba[pred_enc])*100, 1),
        'top3': [
            {'rank':i+1, 'sport':label_encoder.inverse_transform([j])[0],
             'prob': round(float(pred_proba[j])*100, 1)}
            for i, j in enumerate(top3_idx)
        ],
        'all_proba': {c: round(float(p)*100,1) for c, p in zip(classes, pred_proba)}
    }

# =============================================================================
# SESSION STATE
# =============================================================================
for k, v in [('records', []), ('predict_done', False), ('last_result', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# LOAD MODEL
# =============================================================================
model, scaler, label_encoder, model_path = load_model()

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="header-banner">
    <h1>🏅 Sport Suitability Prediction System</h1>
    <p>AI-Enhanced Talent Identification · Longitudinal Study · 200 School Children
       (6th–8th Standard) · Multi-Class Classification</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found. Place `sport_prediction_model.joblib` "
             "in the same folder as `dashboard.py` and restart.")
    st.stop()

hc1, hc2, hc3 = st.columns(3)
hc1.success(f"✅ Model loaded: `{model_path}`")
hc2.info("🤖 Ensemble: Random Forest + Gradient Boosting + SGD")
hc3.info("🎯 Model Accuracy: ~97–98%")
st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Prediction",
    "📋 Batch Prediction",
    "📊 Study Analytics",
    "🔬 AI vs Coach (Final)",
    "ℹ️ Project Overview",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  —  SINGLE STUDENT PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # SIDEBAR inputs
    with st.sidebar:
        st.markdown("## 📝 Student Measurements")
        st.markdown("---")

        student_name = st.text_input("Student Name / ID", "S001")

        cA, cB = st.columns(2)
        with cA: age        = st.selectbox("Age",    list(range(11,16)), index=2)
        with cB: gender_str = st.selectbox("Gender", ["Male","Female"])
        gender = 1 if gender_str == "Male" else 0

        st.markdown("**Anthropometric**")
        height = st.number_input("Height (cm)", 130.0, 200.0, 162.0, 0.5)
        weight = st.number_input("Weight (kg)",  28.0,  90.0,  55.0, 0.5)
        bmi    = round(weight / ((height/100)**2), 2)
        bmi_label = ("Underweight" if bmi<18.5 else "Normal" if bmi<23
                      else "Overweight" if bmi<27.5 else "Obese")
        st.info(f"📐 BMI (auto): **{bmi}**  ({bmi_label})")

        st.markdown("**Performance Scores  0–100**")
        strength    = st.slider("💪 Strength",    0,100,75)
        flexibility = st.slider("🤸 Flexibility", 0,100,80)
        gait        = st.slider("🚶 Gait",        0,100,75)
        walking     = st.slider("👟 Walking",     0,100,72)

        st.markdown("**Coach Prediction (optional)**")
        coach_pred = st.selectbox("Coach Recommendation",
                                   ["-- Not recorded --"]+ALL_SPORTS)
        st.markdown("---")
        predict_btn = st.button("🔍 PREDICT SPORT",
                                 use_container_width=True, type="primary")
        save_btn    = st.button("💾 Save to Study Records",
                                 use_container_width=True)

    # student dict
    student_data = {
        'height_cm':         height,   'weight_kg':        weight,
        'bmi':               bmi,      'strength_score':   float(strength),
        'flexibility_score': float(flexibility),
        'gait_score':        float(gait),
        'walking_score':     float(walking),
        'age':               age,      'gender':           gender,
    }

    # ── INPUT SUMMARY CARDS ──
    st.markdown('<div class="section-title">📋 Student Input Summary</div>',
                unsafe_allow_html=True)

    bmi_c = "#27ae60" if bmi_label=="Normal" else "#e74c3c" if bmi_label=="Underweight" else "#e67e22"
    for col, h2, p, bc in zip(
        st.columns(5),
        [height, weight, bmi, age, "♂" if gender==1 else "♀"],
        ["Height (cm)","Weight (kg)", f"BMI · {bmi_label}","Age (yrs)", gender_str],
        ["#1565C0","#1565C0", bmi_c,"#1565C0","#1565C0"]
    ):
        col.markdown(f"""<div class="metric-card" style="border-left-color:{bc}">
            <h2>{h2}</h2><p>{p}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for col, lbl, val, ico in zip(
        st.columns(4),
        ["Strength","Flexibility","Gait","Walking"],
        [strength,flexibility,gait,walking],
        ["💪","🤸","🚶","👟"]
    ):
        c = "#27ae60" if val>=75 else "#e67e22" if val>=50 else "#e74c3c"
        col.markdown(f"""<div class="metric-card" style="border-left-color:{c}">
            <h2>{val}</h2><p>{ico} {lbl}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PREDICT ──
    if predict_btn:
        st.session_state.last_result  = predict_student(model,scaler,label_encoder,student_data)
        st.session_state.predict_done = True

    if st.session_state.predict_done and st.session_state.last_result:
        result  = st.session_state.last_result
        sport   = result['primary']
        conf    = result['confidence']
        color   = SPORT_COLORS.get(sport,'#1565C0')
        icon    = SPORT_ICONS.get(sport,'🏅')
        profile = SPORT_PROFILES.get(sport,'')

        st.markdown('<div class="section-title">🏅 AI Prediction Result</div>',
                    unsafe_allow_html=True)

        rc, tc, ac = st.columns([1.8,2,2])

        # ─ Primary result
        with rc:
            st.markdown(f"""
            <div class="result-box"
                 style="background:linear-gradient(135deg,#0d2137,{color})">
                <div class="icon">{icon}</div>
                <h2>{sport}</h2>
                <div class="conf">Confidence: <b>{conf}%</b></div>
                <div class="prof">{profile}</div>
            </div>""", unsafe_allow_html=True)

            # confidence bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig_g, ax_g = plt.subplots(figsize=(4,0.45))
            ax_g.barh([0],[100],color='#e3eaf5',height=0.5)
            ax_g.barh([0],[conf],color=color,height=0.5)
            ax_g.text(conf/2,0,f"{conf}%",va='center',ha='center',
                      color='white',fontweight='bold',fontsize=11)
            ax_g.axis('off'); ax_g.set_xlim(0,100)
            plt.tight_layout(pad=0)
            st.pyplot(fig_g, use_container_width=True)

            # AI vs Coach comparison
            if coach_pred != "-- Not recorded --":
                match  = coach_pred == sport
                m_txt  = "✅ MATCH" if match else "❌ DIFFER"
                m_col  = "#27ae60" if match else "#e74c3c"
                st.markdown(f"""
                <div class="compare-box"
                     style="border-left:5px solid {m_col}">
                    <b>AI vs Coach</b><br><br>
                    🤖 AI Prediction    : <b>{sport}</b><br>
                    👨‍🏫 Coach Prediction: <b>{coach_pred}</b><br><br>
                    <span style="color:{m_col};font-weight:700;font-size:1.05rem">{m_txt}</span>
                </div>""", unsafe_allow_html=True)

        # ─ Top 3
        with tc:
            st.markdown("**🏆 Top 3 Recommendations**")
            medals = ['🥇','🥈','🥉']
            for rec in result['top3']:
                s,p = rec['sport'], rec['prob']
                c   = SPORT_COLORS.get(s,'#1565C0')
                w   = max(int(p),4)
                st.markdown(f"""
                <div style="margin:10px 0">
                    <div class="sport-bar-label">
                        {medals[rec['rank']-1]} {SPORT_ICONS.get(s,'')} {s}
                    </div>
                    <div class="sport-bar-outer">
                        <div class="sport-bar-inner" style="width:{w}%;background:{c}">
                            {p}%
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Top-3 pie
            st.markdown("<br>", unsafe_allow_html=True)
            t3_s = [r['sport'] for r in result['top3']]
            t3_p = [r['prob']  for r in result['top3']]
            fig_p, ax_p = plt.subplots(figsize=(4,3.2))
            ax_p.pie(t3_p, labels=t3_s, autopct='%1.1f%%', startangle=90,
                     colors=[SPORT_COLORS.get(s,'#1565C0') for s in t3_s],
                     textprops={'fontsize':9})
            ax_p.set_title('Top 3 Probability Share', fontsize=10)
            plt.tight_layout(); st.pyplot(fig_p, use_container_width=True)

        # ─ All sports
        with ac:
            st.markdown("**📊 All 6 Sports — Full Probability**")
            for s, p in sorted(result['all_proba'].items(),
                                key=lambda x:x[1], reverse=True):
                c = SPORT_COLORS.get(s,'#1565C0')
                w = max(int(p),3)
                st.markdown(f"""
                <div style="margin:8px 0">
                    <div class="sport-bar-label">
                        {SPORT_ICONS.get(s,'')} {s}
                    </div>
                    <div class="sport-bar-outer">
                        <div class="sport-bar-inner" style="width:{w}%;background:{c}">
                            {p}%
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Engineered features
            df_tmp = engineer_features(pd.DataFrame([student_data]))
            st.markdown("<br>**🔧 Computed Features**")
            for feat, col in [
                ("Athleticism Index","athleticism_index"),
                ("Power-to-Weight",  "power_to_weight"),
                ("Mobility Score",   "mobility_score"),
                ("Height/Weight",    "height_weight_ratio"),
                ("BMI Category",     "bmi_category"),
            ]:
                val = round(float(df_tmp[col].iloc[0]),2)
                st.caption(f"**{feat}:** {val}")

        # ── SAVE ──
        if save_btn and st.session_state.predict_done:
            st.session_state.records.append({
                'Timestamp':        datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Student_ID':       student_name,
                'Age':              age,
                'Gender':           gender_str,
                'Height_cm':        height,
                'Weight_kg':        weight,
                'BMI':              bmi,
                'Strength':         strength,
                'Flexibility':      flexibility,
                'Gait':             gait,
                'Walking':          walking,
                'AI_Prediction':    sport,
                'Confidence_pct':   conf,
                'Coach_Prediction': coach_pred if coach_pred!="-- Not recorded --" else "",
                'AI_Coach_Match':   ("Yes" if coach_pred==sport else "No")
                                    if coach_pred!="-- Not recorded --" else "N/A",
                'Final_Sport':      "",
            })
            st.success(f"✅ Record saved for **{student_name}**!")

    else:
        st.markdown("""
        <div class="info-box">
            👈 Enter the student's measurements in the sidebar, then click
            <b>PREDICT SPORT</b> to get the AI recommendation.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  —  BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📋 Batch Prediction — Upload Class CSV</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Required CSV columns:
        <b>height_cm, weight_kg, bmi, strength_score, flexibility_score,
        gait_score, walking_score, age, gender</b><br>
        Optional: <b>student_id, student_name, coach_prediction</b>
    </div>""", unsafe_allow_html=True)

    # Template
    st.download_button("⬇️ Download CSV Template",
        pd.DataFrame([{
            'student_id':'S001','student_name':'Arjun',
            'age':13,'gender':1,
            'height_cm':162.0,'weight_kg':55.0,'bmi':20.97,
            'strength_score':78.0,'flexibility_score':82.0,
            'gait_score':75.0,'walking_score':73.0,
            'coach_prediction':'Football'
        },{
            'student_id':'S002','student_name':'Priya',
            'age':12,'gender':0,
            'height_cm':148.0,'weight_kg':40.0,'bmi':18.26,
            'strength_score':70.0,'flexibility_score':92.0,
            'gait_score':80.0,'walking_score':78.0,
            'coach_prediction':'Gymnastics'
        }]).to_csv(index=False),
        "student_template.csv","text/csv")

    uploaded = st.file_uploader("Upload Student CSV File", type=['csv'])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.write(f"✅ Loaded **{len(df_up)} students**")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("🔍 Run Batch Prediction", type="primary"):
                req_c = ['height_cm','weight_kg','bmi','strength_score',
                         'flexibility_score','gait_score','walking_score','age','gender']
                miss  = [c for c in req_c if c not in df_up.columns]
                if miss:
                    st.error(f"❌ Missing columns: {miss}")
                else:
                    rows = []
                    prog = st.progress(0, text="Running predictions...")
                    for i, row in df_up.iterrows():
                        d   = {k: row[k] for k in req_c}
                        res = predict_student(model, scaler, label_encoder, d)
                        rows.append({
                            'Student_ID':       row.get('student_id',   f'S{i+1:03d}'),
                            'Student_Name':     row.get('student_name', f'Student {i+1}'),
                            'AI_Prediction':    res['primary'],
                            'Confidence_pct':   res['confidence'],
                            '2nd_Choice':       res['top3'][1]['sport'] if len(res['top3'])>1 else '',
                            '3rd_Choice':       res['top3'][2]['sport'] if len(res['top3'])>2 else '',
                            'Coach_Prediction': str(row.get('coach_prediction','')),
                        })
                        prog.progress((i+1)/len(df_up))

                    df_res = pd.DataFrame(rows)
                    def match_lbl(r):
                        cp = r['Coach_Prediction']
                        if cp and cp not in ('','nan'):
                            return "✅ Match" if cp==r['AI_Prediction'] else "❌ Differ"
                        return "N/A"
                    df_res['AI_vs_Coach'] = df_res.apply(match_lbl, axis=1)

                    st.markdown('<div class="section-title">📊 Prediction Results</div>',
                                unsafe_allow_html=True)
                    st.dataframe(df_res, use_container_width=True)

                    s1,s2,s3,s4 = st.columns(4)
                    cmp_b  = df_res[df_res['AI_vs_Coach']!='N/A']
                    mtch_b = cmp_b[cmp_b['AI_vs_Coach']=='✅ Match']
                    agr_b  = round(len(mtch_b)/len(cmp_b)*100,1) if len(cmp_b)>0 else 0
                    s1.metric("Total Students",     len(df_res))
                    s2.metric("Avg Confidence",     f"{round(df_res['Confidence_pct'].mean(),1)}%")
                    s3.metric("Compared w/ Coach",  len(cmp_b))
                    s4.metric("AI–Coach Agreement", f"{agr_b}%")

                    ch1, ch2 = st.columns(2)
                    with ch1:
                        cnt = df_res['AI_Prediction'].value_counts()
                        fig1,ax1 = plt.subplots(figsize=(5,4))
                        ax1.bar(cnt.index, cnt.values,
                                color=[SPORT_COLORS.get(s,'#1565C0') for s in cnt.index],
                                edgecolor='white')
                        ax1.set_title('Batch Sport Distribution')
                        ax1.set_ylabel('Students')
                        plt.xticks(rotation=30,ha='right')
                        plt.tight_layout(); st.pyplot(fig1)
                    with ch2:
                        if len(cmp_b)>0:
                            sa = {}
                            for s in ALL_SPORTS:
                                sg = cmp_b[cmp_b['AI_Prediction']==s]
                                if len(sg)>0:
                                    sa[s]=round(len(sg[sg['AI_vs_Coach']=='✅ Match'])/len(sg)*100,1)
                            if sa:
                                fig2,ax2 = plt.subplots(figsize=(5,4))
                                ax2.barh(list(sa.keys()),list(sa.values()),
                                         color=[SPORT_COLORS.get(s,'#1565C0') for s in sa])
                                ax2.set_xlabel('Agreement (%)')
                                ax2.set_title('AI–Coach Agreement by Sport')
                                ax2.set_xlim(0,105)
                                for i,(k,v) in enumerate(sa.items()):
                                    ax2.text(v+1,i,f'{v}%',va='center',fontsize=9)
                                plt.tight_layout(); st.pyplot(fig2)

                    # Save to records
                    for _, row in df_res.iterrows():
                        st.session_state.records.append({
                            'Timestamp':'', 'Student_ID':row['Student_ID'],
                            'Age':'','Gender':'','Height_cm':'','Weight_kg':'',
                            'BMI':'','Strength':'','Flexibility':'','Gait':'','Walking':'',
                            'AI_Prediction':row['AI_Prediction'],
                            'Confidence_pct':row['Confidence_pct'],
                            'Coach_Prediction':row['Coach_Prediction'],
                            'AI_Coach_Match':row['AI_vs_Coach'],
                            'Final_Sport':'',
                        })
                    st.success(f"✅ {len(df_res)} records added to Study Analytics.")

                    st.download_button("⬇️ Download Results CSV",
                        df_res.to_csv(index=False),
                        f"batch_predictions_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  —  STUDY ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">📊 Longitudinal Study Analytics</div>',
                unsafe_allow_html=True)

    if len(st.session_state.records) == 0:
        st.info("No records yet. Use Tab 1 (single) or Tab 2 (batch) to add student predictions.")
    else:
        df_r    = pd.DataFrame(st.session_state.records)
        total_r = len(df_r)
        cmp_r   = df_r[df_r['AI_Coach_Match'].isin(['Yes','No','✅ Match','❌ Differ'])]
        mtch_r  = cmp_r[cmp_r['AI_Coach_Match'].isin(['Yes','✅ Match'])]
        agr_r   = round(len(mtch_r)/len(cmp_r)*100,1) if len(cmp_r)>0 else 0
        mode_s  = df_r['AI_Prediction'].mode()[0] if total_r>0 else '-'

        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f"""<div class="metric-card"><h2>{total_r}</h2>
            <p>Total Students</p></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card"><h2>{len(cmp_r)}</h2>
            <p>With Coach Comparison</p></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card" style="border-left-color:#27ae60">
            <h2>{agr_r}%</h2><p>AI–Coach Agreement</p></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="metric-card">
            <h2>{SPORT_ICONS.get(mode_s,'🏅')}</h2>
            <p>Top Sport: {mode_s}</p></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        r1,r2,r3 = st.columns(3)
        sc = df_r['AI_Prediction'].value_counts()

        with r1:
            st.markdown("**Sport Distribution (Pie)**")
            fig1,ax1 = plt.subplots(figsize=(4.5,4))
            ax1.pie(sc.values, labels=sc.index, autopct='%1.1f%%', startangle=140,
                    colors=[SPORT_COLORS.get(s,'#1565C0') for s in sc.index],
                    pctdistance=0.82, textprops={'fontsize':8})
            ax1.set_title('AI Predicted Distribution',fontsize=10)
            plt.tight_layout(); st.pyplot(fig1)

        with r2:
            st.markdown("**Students Per Sport (Bar)**")
            fig2,ax2 = plt.subplots(figsize=(4.5,4))
            bars = ax2.bar(sc.index, sc.values,
                           color=[SPORT_COLORS.get(s,'#1565C0') for s in sc.index],
                           edgecolor='white')
            ax2.set_ylabel('Students')
            ax2.set_title('Count per Sport',fontsize=10)
            ax2.bar_label(bars, padding=3, fontsize=9)
            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.tight_layout(); st.pyplot(fig2)

        with r3:
            if len(cmp_r)>0:
                st.markdown("**AI vs Coach Agreement**")
                mc = cmp_r['AI_Coach_Match'].value_counts()
                lbs = [k.replace('✅ ','').replace('❌ ','') for k in mc.index]
                cls = ['#27ae60' if '✅' in k or k=='Yes' else '#e74c3c' for k in mc.index]
                fig3,ax3 = plt.subplots(figsize=(4.5,4))
                ax3.pie(mc.values, labels=lbs, colors=cls,
                        autopct='%1.1f%%', startangle=90)
                ax3.set_title('Match Rate', fontsize=10)
                plt.tight_layout(); st.pyplot(fig3)
            else:
                st.info("Add coach predictions to see agreement chart.")

        st.markdown('<div class="section-title">📋 All Study Records</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_r, use_container_width=True)

        d1, d2, clr = st.columns([2,2,1])
        with d1:
            st.download_button("⬇️ Download Records (CSV)",
                df_r.to_csv(index=False),
                f"study_records_{datetime.now().strftime('%Y%m%d_%H%M')}.csv","text/csv")
        with d2:
            st.markdown("""
            <div class="info-box" style="font-size:0.77rem">
                💡 Fill <b>Final_Sport</b> column after 1 year →
                upload to Tab 4 for final AI vs Coach comparison.
            </div>""", unsafe_allow_html=True)
        with clr:
            if st.button("🗑️ Clear All"):
                st.session_state.records = []
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  —  AI vs COACH FINAL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔬 AI vs Coach — 12-Month Final Comparison</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        After <b>12 months</b>, export records from Tab 3, fill in the
        <code>Final_Sport</code> column for every student, and upload here.
        The system will compute Cohen's Kappa and Accuracy for both
        AI model and Coach, and declare the better predictor.
    </div>""", unsafe_allow_html=True)

    final_file = st.file_uploader("Upload Final Results CSV (with Final_Sport column)",
                                   type=['csv'], key='final_csv')

    if final_file:
        try:
            df_f = pd.read_csv(final_file)
            st.write(f"Loaded **{len(df_f)}** records")

            miss_f = [c for c in ['AI_Prediction','Final_Sport'] if c not in df_f.columns]
            if miss_f:
                st.error(f"Missing columns: {miss_f}")
            else:
                df_cmp = df_f.dropna(subset=['AI_Prediction','Final_Sport'])
                df_cmp = df_cmp[df_cmp['Final_Sport'].str.strip() != '']

                if len(df_cmp)==0:
                    st.warning("No rows with Final_Sport filled in.")
                else:
                    ai_acc   = round((df_cmp['AI_Prediction']==df_cmp['Final_Sport']).mean()*100,1)
                    ai_kappa = cohen_kappa_score(df_cmp['Final_Sport'],df_cmp['AI_Prediction'])

                    has_coach = ('Coach_Prediction' in df_cmp.columns and
                                 df_cmp['Coach_Prediction'].notna().sum()>0)
                    df_cc = pd.DataFrame()
                    if has_coach:
                        df_cc = df_cmp[df_cmp['Coach_Prediction'].notna() &
                                       (df_cmp['Coach_Prediction'].astype(str).str.strip()!='') &
                                       (df_cmp['Coach_Prediction'].astype(str).str.strip()!='nan')]
                        if len(df_cc)>0:
                            coach_acc   = round((df_cc['Coach_Prediction']==df_cc['Final_Sport']).mean()*100,1)
                            coach_kappa = cohen_kappa_score(df_cc['Final_Sport'],df_cc['Coach_Prediction'])

                    st.markdown('<div class="section-title">📊 Final Results</div>',
                                unsafe_allow_html=True)

                    if has_coach and len(df_cc)>0:
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("🤖 AI Accuracy",     f"{ai_acc}%")
                        c2.metric("🤖 AI Kappa",        f"{ai_kappa:.3f}")
                        c3.metric("👨‍🏫 Coach Accuracy", f"{coach_acc}%")
                        c4.metric("👨‍🏫 Coach Kappa",    f"{coach_kappa:.3f}")

                        winner    = "🤖 AI Model" if ai_acc>=coach_acc else "👨‍🏫 Coach"
                        win_color = "#27ae60" if ai_acc>=coach_acc else "#1565C0"
                        st.markdown(f"""
                        <div style="background:{win_color};color:white;border-radius:10px;
                                    padding:16px 22px;text-align:center;
                                    font-size:1.15rem;font-weight:700;margin:16px 0">
                            🏆 Better Predictor: {winner}
                            &nbsp;|&nbsp; AI: {ai_acc}% &nbsp;vs&nbsp; Coach: {coach_acc}%
                        </div>""", unsafe_allow_html=True)

                        # Side-by-side comparison chart
                        fig_v, ax_v = plt.subplots(figsize=(8,4))
                        cats  = ["Accuracy (%)","Cohen's Kappa (×100)"]
                        ai_v  = [ai_acc,    ai_kappa*100]
                        co_v  = [coach_acc, coach_kappa*100]
                        xp    = np.arange(len(cats)); w=0.3
                        b1    = ax_v.bar(xp-w/2, ai_v, w, label='AI Model',color='#1565C0')
                        b2    = ax_v.bar(xp+w/2, co_v, w, label='Coach',   color='#ef6c00')
                        ax_v.set_xticks(xp); ax_v.set_xticklabels(cats)
                        ax_v.set_ylim(0,110)
                        ax_v.set_title('AI Model vs Coach — Final Comparison')
                        ax_v.legend()
                        ax_v.bar_label(b1, fmt='%.1f', padding=3, fontsize=9)
                        ax_v.bar_label(b2, fmt='%.1f', padding=3, fontsize=9)
                        plt.tight_layout(); st.pyplot(fig_v)

                    else:
                        st.metric("🤖 AI Accuracy",     f"{ai_acc}%")
                        st.metric("🤖 AI Cohen's Kappa", f"{ai_kappa:.3f}")
                        st.info("No Coach_Prediction column found.")

                    # Confusion matrix
                    st.markdown('<div class="section-title">🔢 AI Confusion Matrix</div>',
                                unsafe_allow_html=True)
                    classes = sorted(df_cmp['Final_Sport'].unique())
                    cm_ai   = confusion_matrix(df_cmp['Final_Sport'],
                                               df_cmp['AI_Prediction'], labels=classes)
                    fig_cm, ax_cm = plt.subplots(figsize=(8,6))
                    sns.heatmap(cm_ai, annot=True, fmt='d', cmap='Blues',
                                xticklabels=classes, yticklabels=classes, ax=ax_cm)
                    ax_cm.set_title('AI Prediction vs Final Sport (Ground Truth)')
                    ax_cm.set_xlabel('AI Predicted')
                    ax_cm.set_ylabel('Actual Final Sport')
                    plt.tight_layout(); st.pyplot(fig_cm)

                    st.markdown("**Classification Report — AI vs Final Sport**")
                    cr = classification_report(df_cmp['Final_Sport'],
                                               df_cmp['AI_Prediction'],output_dict=True)
                    st.dataframe(pd.DataFrame(cr).T.round(3), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.markdown("""
        <div style="background:#f8f9fa;border-radius:10px;padding:28px;
                    text-align:center;color:#666;margin-top:20px">
            <h3>📅 This tab activates after your 12-month study is complete</h3>
            <p>Export records from <b>Tab 3 → Study Analytics</b>,
               fill in the <code>Final_Sport</code> column for each student,
               then upload here to see the definitive AI vs Coach accuracy comparison.</p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  —  PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">ℹ️ Project Overview & System Design</div>',
                unsafe_allow_html=True)

    ov1, ov2 = st.columns(2)
    with ov1:
        st.markdown("""
### 🎯 Study Objectives
1. Conduct a **1-year longitudinal study** on 200 school children
2. Collect 7 anthropometric & performance variables at baseline
3. Build a multi-class ML classifier to predict best-fit sport
4. Compare **AI predictions vs PE teacher / Coach predictions**
5. Evaluate evidence-based talent identification in school sports

### 📅 Data Collection Schedule
| Phase | Time | Action |
|---|---|---|
| Baseline | Month 0 | Measure all 200 students + record coach prediction |
| Interim  | Month 6 | Re-measure + update model with interim data |
| Final    | Month 12 | Record actual sport excelled → compare AI vs Coach |

### 📥 Input Variables
| Variable | Unit | How Measured |
|---|---|---|
| Height | cm | Stadiometer |
| Weight | kg | Digital scale |
| BMI | kg/m² | Auto-calculated |
| Strength Score | 0–100 | Handgrip / push-up test |
| Flexibility Score | 0–100 | Sit-and-reach test |
| Gait Score | 0–100 | Gait analysis |
| Walking Score | 0–100 | Timed walk test |
| Age | years | 11–15 |
| Gender | M/F | Demographic |

### 🔧 Engineered Features (auto-computed)
| Feature | Formula |
|---|---|
| Athleticism Index | Strength×0.3 + Flex×0.2 + Gait×0.25 + Walk×0.25 |
| Power-to-Weight | Strength / Weight |
| Mobility Score | (Gait + Walking + Flexibility) / 3 |
| Height-Weight Ratio | Height / Weight |
| BMI Category | Underweight / Normal / Overweight / Obese |
        """)

    with ov2:
        st.markdown("""
### 🤖 ML Model Architecture
| Component | Detail |
|---|---|
| Model Type | Voting Ensemble (Soft Voting) |
| Base Models | Random Forest + Gradient Boosting + SGD |
| Raw Features | 9 (7 measurements + age + gender) |
| Engineered Features | 5 composite features |
| **Total Features** | **14** |
| Training Samples | 200,000 synthetic |
| Validation | 5-Fold Stratified Cross-Validation |
| **Accuracy** | **~97–98%** |
| **F1 Score Macro** | **~0.97** |
| **Cohen's Kappa** | **~0.97** |

### 🏅 Sport Profiles — Key Distinguishing Traits
| Sport | Height | Weight | Key Feature |
|---|---|---|---|
| ⚽ Football | 148–170 cm | 45–68 kg | Strength 72–95 |
| 🏐 Volleyball | 165–190 cm | 55–78 kg | Tall + flex 65–85 |
| 🏊 Swimming | 155–182 cm | 42–63 kg | Flexibility 78–98 |
| 🏀 Basketball | 172–195 cm | 60–85 kg | Very tall |
| 🏃 Athletics | 150–175 cm | 38–58 kg | Gait 85–100 |
| 🤸 Gymnastics | 130–158 cm | 28–50 kg | Flexibility 85–100 |

### 📊 Target Evaluation Metrics
| Metric | Target |
|---|---|
| Accuracy | > 85% |
| F1 Score (Macro) | > 0.83 |
| Cohen's Kappa | > 0.80 (substantial agreement) |
        """)

    st.markdown("""
---
### 🔄 Step-by-Step Workflow for PE Teachers

| Step | Tab | Action |
|---|---|---|
| **1** | Tab 1 | Enter one student's measurements → get instant AI sport prediction |
| **2** | Tab 2 | Upload full class CSV → get batch predictions for all 200 students |
| **3** | Tab 3 | Monitor study analytics — distribution of sports, AI vs Coach running totals |
| **4** | Tab 4 | After 12 months: upload final results to compare AI vs Coach accuracy (Cohen's Kappa) |

> 💡 **Expected Outcome:** Cohen's Kappa > 0.80 = substantial agreement between AI and actual performance.
> AI is expected to match or exceed coach prediction accuracy based on physiological data alone.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#999;font-size:0.8rem;padding:8px">
    🏅 Sport Suitability Prediction System ·
    AI-Enhanced Talent Identification ·
    Longitudinal Study — 200 School Children (6th–8th Standard)
</div>""", unsafe_allow_html=True)
