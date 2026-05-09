import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

st.title("📊 Student Performance Analyzer")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error(" Uploaded CSV is empty!")
            st.stop()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ====
    # ✅ DATA PREVIEW
    # ====
    st.write('### Dataset Preview')
    st.dataframe(df)

    # ===
    # 📊 STATISTICS
    # ===
    st.write('### Mean Values')
    st.write(df.mean(numeric_only=True))

    st.write('### Median Values')
    st.write(df.median(numeric_only=True))

    st.write('### Standard Deviation')
    st.write(df.std(numeric_only=True))

    st.write('### Correlation Matrix')
    st.write(df.corr(numeric_only=True))

    # ===
    # 📉 HEATMAP
    # ===
    st.write('### Correlation Heatmap')

    corr = df.corr(numeric_only=True)

    if not corr.empty:
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric data for heatmap")

    # =====
    # 🎯 SUBJECT DETECTION
    # =====
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if 'Study_Hours' not in numeric_cols:
        st.error(" 'Study_Hours' column is required in dataset")
        st.stop()

    subjects = [col for col in numeric_cols if col != 'Study_Hours']

    if len(subjects) == 0:
        st.error(" No subject columns found!")
        st.stop()

    df['Overall'] = df[subjects].mean(axis=1)

    # ====
    # 🎛 SIDEBAR
    # ====
    st.sidebar.title("Options")

    selected_subject = st.sidebar.selectbox(
        "Select Subject",
        options=subjects + ["Overall"]
    )

    show_all = st.sidebar.checkbox("Show All Subjects")
    show_data = st.sidebar.checkbox("Show Dataset")

    # =====
    # 📈 SCATTER PLOT
    # =====
    st.write("### Study Hours vs Performance")

    fig, ax = plt.subplots()

    if show_all:
        for subject in subjects:
            ax.scatter(df['Study_Hours'], df[subject], label=subject)
        ax.legend()
    else:
        if selected_subject == "Overall":
            ax.scatter(df['Study_Hours'], df['Overall'])
            ax.set_ylabel("Overall Marks")
        else:
            ax.scatter(df['Study_Hours'], df[selected_subject])
            ax.set_ylabel(f"{selected_subject} Marks")

    ax.set_xlabel("Study Hours")
    st.pyplot(fig)

    # ======
    # 🏆 TOP STUDENTS
    # ======
    st.write("### Top Performers")  # 

    if selected_subject == "Overall":
        threshold = df['Overall'].quantile(0.75)
        top_students = df[df['Overall'] > threshold]
    else:
        threshold = df[selected_subject].quantile(0.75)
        top_students = df[df[selected_subject] > threshold]

    st.write(f'Top 25% Students (> {threshold:.2f})')
    st.dataframe(top_students)

    # ====
    # 📋 SHOW DATA OPTION
    # ====
    if show_data:
        st.write("### Full Dataset")
        st.dataframe(df)