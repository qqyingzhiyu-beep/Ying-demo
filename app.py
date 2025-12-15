import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from shap import Explanation
from shap.plots import waterfall

# 加载模型
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🧠 智慧学途 - 学业风险预警系统")

col1, col2 = st.columns(2)
with col1:
    age = st.selectbox("年龄级别", [1,2,3])
    sex = st.selectbox("性别", [1,2])
    scholarship = st.slider("奖学金类型", 1,5,3)
    weekly_hours = st.slider("每周学习小时", 0,4,2)
    attendance = st.selectbox("出勤情况", [1,2,3])
with col2:
    listening = st.selectbox("听课认真度", [1,2,3])
    notes = st.selectbox("记笔记习惯", [1,2,3])
    last_cgpa = st.selectbox("上学期CGPA", [1,2,3,4,5])
    prep1 = st.selectbox("期中备考1", [1,2,3])
    prep2 = st.selectbox("期中备考2", [1,2,3])

投入 = weekly_hours*0.3 + attendance*0.3 + listening*0.2 + notes*0.2
规律 = (prep1 + prep2)/2

data = pd.DataFrame({
    'Student Age': [age], 'Sex': [sex], 'Scholarship type': [scholarship],
    'Additional work': [1], 'Weekly study hours': [weekly_hours],
    'Attendance to classes': [attendance], 'Taking notes in classes': [notes],
    'Listening in classes': [listening], 'Preparation to midterm exams 1': [prep1],
    'Preparation to midterm exams 2': [prep2],
    'Cumulative grade point average in the last semester (/4.00)': [last_cgpa],
    '学习投入指数': [投入], '期中备考规律性': [规律],
    '兴趣驱动指数': [2.0], '阅读广度指数': [4.0], '预期信心指数': [0.5]
})

data = pd.get_dummies(data, columns=['Sex', 'Scholarship type', 'Additional work'], drop_first=True)
for col in feature_names:
    if col not in data.columns:
        data[col] = 0
data = data[feature_names]
data_scaled = scaler.transform(data)

pred = model.predict(data_scaled)[0]
prob = model.predict_proba(data_scaled)[0][pred]
risk = le.inverse_transform([pred])[0]

st.markdown(f"## 预测风险：**{risk}**")
st.progress(prob)

if risk == 'High':
    st.error("高风险！建议加强监督和辅导")
elif risk == 'Medium':
    st.warning("中风险，建议考试培训")
else:
    st.success("低风险，继续保持！")

# SHAP瀑布图
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_scaled)
shap_val = shap_values[pred][0]
fig, ax = plt.subplots()
waterfall(Explanation(shap_val, explainer.expected_value[pred], data_scaled[0], feature_names=feature_names))
st.pyplot(fig)
