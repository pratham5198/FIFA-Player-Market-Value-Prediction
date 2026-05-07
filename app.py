import pandas as pd, numpy as np, streamlit as st, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

st.set_page_config(page_title="FIFA Value Predictor", layout="wide")

# ---------- STYLE ----------
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#14001f,#2a0047);color:white;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1f0036,#3a0066);}
.stButton>button{background:linear-gradient(135deg,#c084fc,#9333ea);color:white;border-radius:12px;}
.metric-box{background:linear-gradient(135deg,#2e0b4d,#4c1d95);padding:20px;border-radius:16px;text-align:center;}
header,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data
def load(): return pd.read_csv("players_21.csv")

df_all = load()
cols = ['age','overall','potential','wage_eur','international_reputation']
df = df_all[cols+['value_eur']].dropna()

X, y = df[cols], df['value_eur']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

# ---------- MODELS ----------
models = {
    "Linear Regression": Pipeline([('s',StandardScaler()),('m',LinearRegression())]),
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=100,n_jobs=1),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100,learning_rate=0.1,n_jobs=1,tree_method='hist'),
    "KNN": Pipeline([('s',StandardScaler()),('m',KNeighborsRegressor(5))])
}
params = {"Random Forest":{'max_depth':[10,None]},"XGBoost":{'learning_rate':[0.1]}}

# ---------- TRAIN ----------
@st.cache_resource
def train():
    res, best_model = [], {}
    for n,m in models.items():
        m = GridSearchCV(m, params[n], cv=3,n_jobs=1).fit(Xtr,ytr).best_estimator_ if n in params else m.fit(Xtr,ytr)
        p = m.predict(Xte)
        res.append([n,mean_absolute_error(yte,p),np.sqrt(mean_squared_error(yte,p)),r2_score(yte,p)])
        best_model[n]=m
    df_res = pd.DataFrame(res,columns=["Model","MAE","RMSE","R2"])
    best = df_res.loc[df_res.R2.idxmax(),"Model"]
    return df_res, best, best_model[best]

res_df, best_name, best_model = train()

# ---------- UI ----------
st.title("⚽ FIFA Market Value Prediction Dashboard")

c1,c2,c3 = st.columns(3)
row = res_df[res_df.Model==best_name]

c1.markdown(f"<div class='metric-box'><h3>Best Model</h3><h2>{best_name}</h2></div>",True)
c2.markdown(f"<div class='metric-box'><h3>R²</h3><h2>{row.R2.values[0]:.3f}</h2></div>",True)
c3.markdown(f"<div class='metric-box'><h3>RMSE</h3><h2>{int(row.RMSE.values[0]):,}</h2></div>",True)

st.subheader("📊 Model Performance")
st.dataframe(res_df)

# ---------- CHARTS ----------
# ---------- CHARTS ----------
c1, c2, c3 = st.columns(3)

for col, metric in zip([c1, c2, c3], ["MAE","RMSE","R2"]):
    with col:
        fig = plt.figure()
        plt.bar(res_df["Model"], res_df[metric])
        plt.xticks(rotation=45)
        plt.title(metric)
        st.pyplot(fig)

# ---------- SIDEBAR ----------
st.sidebar.markdown("### 🎯 Player Input")

names = ["Custom"] + sorted(df_all['short_name'].dropna().unique())
p = st.sidebar.selectbox("Select Player", names)

d = df_all[df_all.short_name==p].iloc[0] if p!="Custom" else None
def val(k,default): return int(d[k]) if d is not None else default

age = st.sidebar.number_input("Age", value=val('age',0))
ovr = st.sidebar.number_input("Overall", value=val('overall',0))
pot = st.sidebar.number_input("Potential", value=val('potential',0))
wag = st.sidebar.number_input("Wage", value=val('wage_eur',0))
rep = st.sidebar.number_input("Reputation", value=val('international_reputation',1))

# ---------- PREDICT ----------
st.subheader("💰 Prediction Result")
if st.sidebar.button("Predict"):
    inp = pd.DataFrame([[age,ovr,pot,wag,rep]], columns=cols)
    pred = best_model.predict(inp)[0]
    st.markdown(f"<div class='metric-box'><h2>Estimated Value</h2><h1>€{int(pred):,}</h1></div>",True)

st.markdown("---")
st.markdown("### 🚀 MSc Data Science | 25P0630009")
