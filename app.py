import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model, setup

# Loading the saved model
model_file = "saved_model_1"  # didn't work when attempted with pickle. so models made with pycaret must also be loaded with pycaret itself
model = load_model(model_file)

# setting the same setting applied to the training data so we can apply same transformations to the input data for predictions
training_data = pd.read_csv("./Data/creditcard.csv")
setup(data=training_data, target="Class", session_id=123)


# Function to make predictions
def make_prediction(data):
    # Use the model to make predictions
    predictions = predict_model(model, data=data)
    return predictions


# Streamlit UI
st.title("Credit Card Fraud Detection Model")
st.markdown("<br>", unsafe_allow_html=True)


st.sidebar.header("Input Parameters")

time = st.sidebar.number_input("Time", min_value=0.0, value=80450.513742)
v1 = st.sidebar.number_input("V1", value=-4.498280)
v2 = st.sidebar.number_input("V2", value=3.405965)
v3 = st.sidebar.number_input("V3", value=-6.729599)
v4 = st.sidebar.number_input("V4", value=4.472591)
v5 = st.sidebar.number_input("V5", value=-2.957197)
v6 = st.sidebar.number_input("V6", value=-1.432518)
v7 = st.sidebar.number_input("V7", value=-5.175912)
v8 = st.sidebar.number_input("V8", value=0.953255)
v9 = st.sidebar.number_input("V9", value=-2.522124)
v10 = st.sidebar.number_input("V10", value=-5.453274)
v11 = st.sidebar.number_input("V11", value=3.716347)
v12 = st.sidebar.number_input("V12", value=-6.103254)
v13 = st.sidebar.number_input("V13", value=-0.094324)
v14 = st.sidebar.number_input("V14", value=-6.835946)
v15 = st.sidebar.number_input("V15", value=-0.072830)
v16 = st.sidebar.number_input("V16", value=-4.000956)
v17 = st.sidebar.number_input("V17", value=-6.463285)
v18 = st.sidebar.number_input("V18", value=-2.157071)
v19 = st.sidebar.number_input("V19", value=0.669143)
v20 = st.sidebar.number_input("V20", value=0.405043)
v21 = st.sidebar.number_input("V21", value=0.466550)
v22 = st.sidebar.number_input("V22", value=0.086639)
v23 = st.sidebar.number_input("V23", value=-0.096464)
v24 = st.sidebar.number_input("V24", value=-0.106643)
v25 = st.sidebar.number_input("V25", value=0.040615)
v26 = st.sidebar.number_input("V26", value=0.050456)
v27 = st.sidebar.number_input("V27", value=0.213774)
v28 = st.sidebar.number_input("V28", value=0.078270)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=123.871860)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("./Images/featureImp.png", use_container_width=True)


# parsing input data as a DataFrame
input_data = {
    "Time": [time],
    "V1": [v1],
    "V2": [v2],
    "V3": [v3],
    "V4": [v4],
    "V5": [v5],
    "V6": [v6],
    "V7": [v7],
    "V8": [v8],
    "V9": [v9],
    "V10": [v10],
    "V11": [v11],
    "V12": [v12],
    "V13": [v13],
    "V14": [v14],
    "V15": [v15],
    "V16": [v16],
    "V17": [v17],
    "V18": [v18],
    "V19": [v19],
    "V20": [v20],
    "V21": [v21],
    "V22": [v22],
    "V23": [v23],
    "V24": [v24],
    "V25": [v25],
    "V26": [v26],
    "V27": [v27],
    "V28": [v28],
    "Amount": [amount],
}

input_df = pd.DataFrame(input_data)

# Show the input data for confirmation
st.write("Input Data ->")
st.write(input_df)


if st.button("Predict Fraud or Not 💳"):
    prediction = make_prediction(input_df)
    st.markdown("#### Prediction Results")
    st.write(
        prediction[["prediction_label"]]
    )  # Adjust according to the correct column names for predictions

    if prediction["prediction_label"][0] == 1:
        st.write("🔴 FRAUDULENT Transaction.")
    else:
        st.write("🟢 VALID Transaction. ")


data = {
    "Model": [
        "Extra Trees Classifier",
        "Random Forest Classifier",
        "Dummy Classifier",
        "Decision Tree Classifier",
        "Light Gradient Boosting Machine",
        "K Neighbors Classifier",
        "Ridge Classifier",
        "Linear Discriminant Analysis",
        "Gradient Boosting Classifier",
        "SVM - Linear Kernel",
        "Logistic Regression",
        "Ada Boost Classifier",
        "Naive Bayes",
        "Quadratic Discriminant Analysis",
    ],
    "Accuracy": [
        0.9992,
        0.9987,
        0.9983,
        0.9970,
        0.9959,
        0.9957,
        0.9936,
        0.9936,
        0.9897,
        0.9855,
        0.9847,
        0.9830,
        0.9814,
        0.9709,
    ],
    "AUC": [
        0.9547,
        0.9521,
        0.5000,
        0.9021,
        0.9541,
        0.9234,
        0.9677,
        0.9677,
        0.9652,
        0.9740,
        0.9743,
        0.9585,
        0.9683,
        0.9690,
    ],
    "Recall": [
        0.8311,
        0.8282,
        0.0000,
        0.8070,
        0.8461,
        0.8492,
        0.7948,
        0.7948,
        0.8701,
        0.8883,
        0.8883,
        0.8852,
        0.8854,
        0.8855,
    ],
    "Prec.": [
        0.7288,
        0.5846,
        0.0000,
        0.3370,
        0.2689,
        0.2618,
        0.1813,
        0.1813,
        0.1279,
        0.0996,
        0.0910,
        0.0809,
        0.0749,
        0.0489,
    ],
    "F1": [
        0.7717,
        0.6816,
        0.0000,
        0.4741,
        0.4069,
        0.3991,
        0.2945,
        0.2945,
        0.2225,
        0.1781,
        0.1648,
        0.1482,
        0.1380,
        0.0927,
    ],
    "Kappa": [
        0.7713,
        0.6810,
        0.0000,
        0.4729,
        0.4054,
        0.3975,
        0.2926,
        0.2926,
        0.2202,
        0.1756,
        0.1623,
        0.1456,
        0.1354,
        0.0898,
    ],
    "MCC": [
        0.7754,
        0.6932,
        0.0000,
        0.5195,
        0.4748,
        0.4692,
        0.3770,
        0.3770,
        0.3305,
        0.2925,
        0.2810,
        0.2646,
        0.2542,
        0.2041,
    ],
    "TT (Sec)": [
        7.5530,
        35.0100,
        2.5790,
        7.9310,
        3.3330,
        4.7020,
        3.9810,
        2.3540,
        37.6960,
        5.0140,
        3.5120,
        10.9090,
        5.5360,
        3.1900,
    ],
}

df1 = pd.DataFrame(data)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Model Performance Metrics"):
    st.dataframe(df1)

st.write("**Machine Learning Model used 🧠 - Extra Trees Classifier**")
