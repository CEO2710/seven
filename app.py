# app.py - 最终优化版本 (无运行时依赖安装)

import streamlit as st
import pandas as pd
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="Surgical Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# 尝试导入 joblib
try:
    import joblib
    st.success("✅ joblib successfully imported")
except ImportError:
    st.error("❌ joblib not found. This application cannot run without joblib.")
    st.error("Please ensure joblib is included in your requirements.txt file")
    st.stop()

# 加载模型
try:
    model = joblib.load("model.joblib")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.error("Please ensure 'model.joblib' exists and is a valid joblib file")
    st.stop()

# 变量配置
VARIABLE_CONFIG = {
    "Sex": {
        "min": 0, 
        "max": 1,
        "description": "Patient gender (0=Female, 1=Male)",
        "value": 0
    },
    "ASA scores": {
        "min": 0,
        "max": 5,
        "description": "ASA physical status classification",
        "value": 2
    },
    "tumor location": {
        "min": 1,
        "max": 4,
        "description": "Tumor location code (1-4)",
        "value": 2
    },
    "Benign or malignant": {
        "min": 0,
        "max": 1,
        "description": "Tumor nature (0=Benign, 1=Malignant)",
        "value": 0
    },
    "Admitted to NICU": {
        "min": 0,
        "max": 1,
        "description": "NICU admission status",
        "value": 0
    },
    "Duration of surgery": {
        "min": 0,
        "max": 1,
        "description": "Surgery duration category",
        "value": 0
    },
    "diabetes": {
        "min": 0,
        "max": 1,
        "description": "Diabetes mellitus status",
        "value": 0
    },
    "CHF": {
        "min": 0,
        "max": 1,
        "description": "Congestive heart failure",
        "value": 0
    },
    "Functional dependencies": {
        "min": 0,
        "max": 1,
        "description": "Functional dependencies",
        "value": 0
    },
    "mFI-5": {
        "min": 0,
        "max": 5,
        "description": "Modified Frailty Index",
        "value": 1
    },
    "Type of tumor": {
        "min": 1,
        "max": 5,
        "description": "Tumor type code (1-5)",
        "value": 2
    }
}

# 应用界面
st.title("Unplanned Reoperation Risk Prediction System")
st.markdown("---")

# 创建输入表单
st.subheader("Patient Parameters")
inputs = {}
cols = st.columns(2)
for i, (feature, config) in enumerate(VARIABLE_CONFIG.items()):
    with cols[i % 2]:
        inputs[feature] = st.number_input(
            label=f"{feature}",
            help=config["description"],
            min_value=config["min"],
            max_value=config["max"],
            value=config["value"],
            step=1
        )

# 预测按钮
if st.button("Predict Risk", type="primary"):
    try:
        # 创建输入数据框
        input_df = pd.DataFrame([inputs])
        
        # 确保特征顺序正确
        expected_columns = list(VARIABLE_CONFIG.keys())
        input_df = input_df[expected_columns]
        
        # 执行预测
        with st.spinner("Calculating risk..."):
            try:
                proba = model.predict_proba(input_df)[0][1]
            except AttributeError:
                # 如果模型没有predict_proba方法，使用其他方式
                prediction = model.predict(input_df)[0]
                proba = prediction  # 假设模型直接输出概率
                
            risk_level = "High Risk" if proba > 0.5 else "Low Risk"
            color = "#FF4B4B" if proba > 0.5 else "#00CC96"
            
            # 显示结果
            st.markdown("---")
            st.subheader("Prediction Results")
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{risk_level}</h2>", 
                        unsafe_allow_html=True)
            st.write(f"### Reoperation Probability: {proba:.1%}")
            
            # 特征重要性展示
            st.markdown("---")
            st.subheader("Key Risk Factors")
            
            try:
                # 尝试获取特征重要性
                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importance = model.coef_[0]
                else:
                    raise AttributeError("Feature importance not available")
                
                # 创建重要性数据框
                importance_df = pd.DataFrame({
                    "Feature": expected_columns,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)
                
                # 显示重要性图表
                st.bar_chart(importance_df.set_index("Feature"))
                
                # 显示重要性表格
                st.dataframe(importance_df)
                
            except Exception as e:
                st.warning(f"⚠️ Feature importance not available: {str(e)}")
                st.info("The model does not provide feature importance data")
                
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.error("Please check your input values and try again")

# 侧边栏文档
with st.sidebar:
    st.markdown("""
    ## System Documentation
    
    ### Model Information
    - Algorithm: XGBoost Classifier
    - Training Date: 2023-10-15
    
    ### Variable Codes
    - **ASA scores**: 
      0 = Healthy, 5 = Morbund
    - **Tumor location**:
      1 = Supratentorial extramedullary,  
      2 = Supratentorial intramedullary,  
      3 = Infratentorial extramedullary,  
      4 = Infratentorial intramedullary
    - **Tumor type**:
      1 = Meningioma,  
      2 = Glioma,  
      3 = Metastasis,  
      4 = Acoustic neuroma,  
      5 = Other
    """)
    
    st.markdown("---")
    st.markdown("**Technical Support**: contact@medai.org")
    st.markdown("**Version**: 1.4.0")