import streamlit as st
import joblib
import pandas as pd

st.title("心脏病风险预测演示")
st.caption("教学演示，不构成医疗建议")

model = joblib.load("heart_model.joblib")
cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

age  = st.slider("年龄", 29, 77, 55)
sex  = st.selectbox("性别", [1, 0], format_func=lambda v: "男" if v else "女")
cp   = st.selectbox("胸痛类型", [1, 2, 3, 4],
                    format_func=lambda v: {1:"典型心绞痛",2:"非典型心绞痛",
                                           3:"非心绞痛",4:"无症状"}[v])
tbps = st.slider("静息血压 (mm Hg)", 94, 200, 130)
chol = st.slider("血清胆固醇 (mg/dl)", 126, 564, 240)
fbs  = st.selectbox("空腹血糖>120", [0, 1], format_func=lambda v: "是" if v else "否")
ecg  = st.selectbox("静息心电图", [0, 1, 2],
                    format_func=lambda v: {0:"正常",1:"ST-T波异常",2:"左心室肥大"}[v])
thal = st.slider("最大心率", 71, 202, 150)
exg  = st.selectbox("运动诱发心绞痛", [0, 1], format_func=lambda v: "是" if v else "否")
oldp = st.slider("ST段压低", 0.0, 6.2, 1.0, 0.1)
slope = st.selectbox("ST段斜率", [1, 2, 3],
                     format_func=lambda v: {1:"上升",2:"平坦",3:"下降"}[v])
ca   = st.selectbox("显影血管数", [0, 1, 2, 3])
thal_ = st.selectbox("铊试验", [3.0, 6.0, 7.0],
                     format_func=lambda v: {3.0:"正常",6.0:"固定缺损",7.0:"可逆缺损"}[v])

THRESHOLD = 0.32   # ← 换成你第 6 周选定的阈值

if st.button("预测"):
    x = pd.DataFrame([[age, sex, cp, tbps, chol, fbs, ecg, thal,
                       exg, oldp, slope, ca, thal_]], columns=cols)
    p = model.predict_proba(x)[0, 1]
    st.metric("患病概率", f"{p:.1%}")
    if p >= THRESHOLD:
        st.warning("高风险，建议进一步检查")
    else:
        st.success("低风险")