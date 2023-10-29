import streamlit as st
import pickle
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go


with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

dataset = "Sleep_health_and_lifestyle_dataset.csv"
df = pd.read_csv(dataset)

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
object_columns = df.select_dtypes(include=["object"]).columns

st.title("Sleep Disorder Prediction App")

st.sidebar.header("Navigate")
section = st.sidebar.radio(
    "Sections",
    ["Prediction", "Graphs", "Distributions", "Violin Plots", "Line Plots"],
)


if section == "Prediction":
    st.header("Sleep Disorder Prediction")
    st.write("Please provide the following information to predict your sleep disorder.")

    # Gender Input
    gender = st.selectbox("Gender:", ["Male", "Female"])
    st.write("Select your gender.")

    # Age Input
    age = st.slider("Age:", 18, 90, 30)
    st.write("Select your age.")

    # Occupation Input
    occupation = st.selectbox(
        "Occupation:",
        [
            "Nurse",
            "Doctor",
            "Engineer",
            "Lawyer",
            "Accountant",
            "Teacher",
            "Salesperson",
            "Software Engineer",
            "Scientist",
            "Manager",
        ],
        index=2,
    )
    st.write("Choose your occupation.")

    # Sleep Duration Input
    sleep_duration = st.slider("Sleep Duration (hours):", 1.0, 24.0, 7.0, 0.1)
    st.write("Enter the average number of hours you sleep per day.")

    # Quality of Sleep Input
    quality_of_sleep = st.slider("Quality of Sleep (1-10):", 1, 10, 5)
    st.write("Rate the quality of your sleep on a scale from 1 to 10.")

    # Physical Activity Input
    physical_activity = st.slider("Physical Activity Level (minutes/day):", 0, 180, 30)
    st.write("Enter the number of minutes you engage in physical activity per day.")

    # Stress Level Input
    stress_level = st.slider("Stress Level (1-10):", 1, 10, 5)
    st.write("Rate your stress level on a scale from 1 to 10.")

    # BMI Category Input
    bmi_category = st.selectbox(
        "BMI Category:", ["Underweight", "Normal", "Overweight"], index=1
    )
    st.write("Select your BMI category.")

    # Blood Pressure Input
    blood_pressure = st.text_input("Blood Pressure (systolic/diastolic)", "120/80")
    st.write(
        "Enter your blood pressure in the format 'systolic / diastolic' (e.g., '120/80')."
    )

    # Heart Rate Input
    heart_rate = st.slider("Resting Heart Rate (bpm):", 40, 120, 70)
    st.write("Enter your resting heart rate in beats per minute (bpm).")

    # Daily Steps Input
    daily_steps = st.slider("Daily Steps:", 0, 20000, 5000)
    st.write("Enter the average number of steps you take per day.")

    user_data = pd.DataFrame(
        {
            "Age": [age],
            "Sleep Duration": [sleep_duration],
            "Quality of Sleep": [quality_of_sleep],
            "Physical Activity Level": [physical_activity],
            "Stress Level": [stress_level],
            "Heart Rate": [heart_rate],
            "Daily Steps": [daily_steps],
            "Systolic": [blood_pressure.split("/")[0]],
            "Diastolic": [blood_pressure.split("/")[1]],
            "Gender_Male": [1 if gender == "Male" else 0],
            "Occupation_Doctor": [1 if occupation == "Doctor" else 0],
            "Occupation_Engineer": [1 if occupation == "Engineer" else 0],
            "Occupation_Lawyer": [1 if occupation == "Lawyer" else 0],
            "Occupation_Manager": [1 if occupation == "Manager" else 0],
            "Occupation_Nurse": [1 if occupation == "Nurse" else 0],
            "Occupation_Salesperson": [1 if occupation == "Salesperson" else 0],
            "Occupation_Scientist": [1 if occupation == "Scientist" else 0],
            "Occupation_Software Engineer": [
                1 if occupation == "Software Engineer" else 0
            ],
            "Occupation_Teacher": [1 if occupation == "Teacher" else 0],
            "BMI Category_Overweight": [1 if bmi_category == "Overweight" else 0],
            "BMI Category_Underweight": [1 if bmi_category == "Underweight" else 0],
        }
    )

    def predict_sleep_disorder(user_data):
        prediction = model.predict(user_data)[0]
        sleep_disorder_label = label_encoder.inverse_transform([prediction])[0]

        sleep_disorder_descriptions = {
            "None": "The individual does not exhibit any specific sleep disorder.",
            "Insomnia": "The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.",
            "Sleep Apnea": "The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.",
        }

        sleep_disorder_description = sleep_disorder_descriptions.get(
            sleep_disorder_label, "Description not available."
        )

        return sleep_disorder_label, sleep_disorder_description

    st.subheader("Predicted Sleep Disorder")
    if st.button("Predict Sleep Disorder"):
        predicted_disorder, disorder_description = predict_sleep_disorder(user_data)
        st.write(predicted_disorder)
        st.subheader("Diagnosis:")
        st.write(disorder_description)

elif section == "Graphs":
    st.header("Graphs")
    st.subheader("Dataset Statistics")
    st.write("Here are some summary statistics for numerical columns:")
    st.write(df.describe())

    st.header("Data Exploration")

    st.subheader("Stress Level by Gender")
    gender_stress = df.groupby("Gender")["Stress Level"].mean()
    x = np.arange(len(gender_stress))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x, gender_stress, width, label="Stress Level", color=["blue", "green"])
    plt.xlabel("Gender")
    plt.ylabel("Average Stress Level")
    plt.title("Stress Level by Gender")
    plt.xticks(x, gender_stress.index)
    plt.legend()
    st.pyplot(plt)

    st.subheader("Occupation vs Daily Steps Taken")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Occupation", y="Daily Steps", data=df, ci=None)
    plt.title("Comparison of Average Daily Steps by Occupation")
    plt.xlabel("Occupation")
    plt.ylabel("Average Daily Steps")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader("Blood Pressure Trends by Age")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Age", y="Blood Pressure", data=df, ci=None)
    plt.title("Blood Pressure Trends by Age")
    plt.xlabel("Age")
    plt.ylabel("Blood Pressure")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader("BMI Category Distribution")
    bmi_counts = df["BMI Category"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(bmi_counts, labels=bmi_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("BMI Category Distribution")
    plt.axis("equal")
    st.pyplot(plt)

    st.subheader("Sleep Disorder Bar Chart")
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Sleep Disorder"].value_counts().index,
                y=df["Sleep Disorder"].value_counts(),
            )
        ]
    )
    fig.update_layout(
        title="Sleep Disorder", xaxis_title="Sleep Disorder", yaxis_title="Count"
    )
    st.plotly_chart(fig)

    st.subheader("Sleep Disorder Pie Chart")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["Sleep Disorder"].value_counts().index,
                values=df["Sleep Disorder"].value_counts(),
            )
        ]
    )
    fig.update_layout(title="Sleep Disorder")
    st.plotly_chart(fig)

    st.subheader("Gender Bar Chart")
    fig = go.Figure(
        data=[
            go.Bar(x=df["Gender"].value_counts().index, y=df["Gender"].value_counts())
        ]
    )
    fig.update_layout(title="Gender", xaxis_title="Gender", yaxis_title="Count")
    st.plotly_chart(fig)

    counts = df["Gender"].value_counts()
    st.subheader("Gender Pie Chart")
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
    fig.update_layout(title="Gender")
    st.plotly_chart(fig)

    st.subheader("Occupation Bar Chart")
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Occupation"].value_counts().index,
                y=df["Occupation"].value_counts(),
            )
        ]
    )
    fig.update_layout(title="Occupation", xaxis_title="Occupation", yaxis_title="Count")
    st.plotly_chart(fig)

    st.subheader("Occupation Pie Chart")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["Occupation"].value_counts().index,
                values=df["Occupation"].value_counts(),
            )
        ]
    )
    fig.update_layout(title="Occupation")
    st.plotly_chart(fig)

    st.subheader("BMI Category Bar Chart")
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["BMI Category"].value_counts().index,
                y=df["BMI Category"].value_counts(),
            )
        ]
    )
    fig.update_layout(
        title="BMI Category", xaxis_title="BMI Category", yaxis_title="Count"
    )
    st.plotly_chart(fig)

    st.subheader("BMI Category Pie Chart")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["BMI Category"].value_counts().index,
                values=df["BMI Category"].value_counts(),
            )
        ]
    )
    fig.update_layout(title="BMI Category")
    st.plotly_chart(fig)

    st.subheader("Sleep Disorder vs Age")
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Sleep Disorder"].value_counts().index,
                y=df["Sleep Disorder"].value_counts(),
            )
        ]
    )
    fig.update_layout(
        title="Sleep Disorder vs Age", xaxis_title="Sleep Disorder", yaxis_title="Age"
    )
    st.plotly_chart(fig)

elif section == "Distributions":
    st.header("Distributions")
    for column in df.select_dtypes(include=["int64", "float64"]).columns:
        if column != "Person ID":
            plt.figure(figsize=(15, 6))
            sns.histplot(df[column], kde=True, bins=20, palette="hls")
            plt.title(f"{column} Distribution")
            plt.xlabel(column)
            plt.xticks(rotation=0)
            st.pyplot(plt)

elif section == "Violin Plots":
    st.header("Violin Plots")
    for i in numerical_columns:
        for j in object_columns:
            if i != "Person ID" and j != "Blood Pressure":
                st.subheader(f"Violin Plot: {j} vs {i}")
                plt.figure(figsize=(15, 6))
                sns.violinplot(x=j, y=i, data=df, palette="hls")
                st.pyplot(plt)

elif section == "Line Plots":
    st.header("Line Plots")
    for i in numerical_columns:
        for j in numerical_columns:
            if i != "Person ID" and j != "Person ID" and i != j:
                st.subheader(f"Line Plot: {i} vs {j}")
                plt.figure(figsize=(8, 4))
                sns.lineplot(x=j, y=i, data=df, palette="hls")
                st.pyplot(plt)

st.header("About")
st.write(
    "App repository available on [GitHub](https://github.com/GopalSaraf/Sleep-Disorder-Prediction)."
)

st.header("Models Used")
st.write(
    """- Logistic Regression
- Decision Tree
- XGBoost"""
)

st.header("Results")
st.write(
    """
| Model                 | Accuracy | Precision | Recall | F1 Score |
| -------------         | -------- | --------- | ------ | -------- |
| Logistic Regression   | 0.79     | 0.80      | 0.73   | 0.75     |
| Decision Tree         | 0.96     | 0.95      | 0.95   | 0.95     |
| XGBoost               | 0.96     | 0.95      | 0.95   | 0.95     |
"""
)


st.header("Contributors")
st.write(
    "This app was created by Gopal Saraf, Riddhi Sabane, Mugdha Kulkarni, and Vaishnavi Shinde."
)

st.write(
    """- [Gopal Saraf](https://github.com/GopalSaraf)
- [Riddhi Sabane](https://github.com/sabaneriddhi)
- [Mugdha Kulkarni](https://github.com/mugdha0611)
- [Vaishnavi Shinde](https://github.com/GopalSaraf)"""
)

st.header("License")
st.write("This app is provided under the MIT License.")
