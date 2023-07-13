# This the main file for the project. It contains the main streamlit app.
import os

import streamlit as st
import pandas as pd
import numpy as np
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import plotly.express as px

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.anomaly import AnomalyExperiment

# make streamlit app full width
st.set_page_config(layout="wide")

with open("styles.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png",
             width=300)
    st.markdown("<h1 style='text-align: center; color: white;'>SMART-ML <p>Machine Learning Made Easy!</p></h1>",
                unsafe_allow_html=True)
    choices = st.radio("Navigation", ["Dataset", "EDA", "Modeling"])
    st.info("This application will help you to perform machine learning tasks on your dataset. You can upload your "
            "dataset,"
            " perform exploratory data analysis, and train machine learning models which can be for classification,"
            " regression, clustering, or anomaly detection.")
   
if choices == 'Dataset':
    st.title('Upload your CSV dataset: :smile:')
    file = st.file_uploader("Upload you dataset file", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choices == 'EDA':
    st.title('Exploratory Data Analysis: :bar_chart:')
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choices == 'Modeling':

    exp_choice = st.radio("Choose an Experiment", ["Classification", "Regression", "Clustering", "Anomaly Detection"])
    if exp_choice == "Classification":
        st.title('Classification Experiment: :chart_with_upwards_trend:')

        X = st.multiselect("Select Features", df.columns)
        y = st.selectbox("Select Target Variable", df.columns)

        data = df[X + [y]]

        st.title('Modeling: :rocket:')
        if st.button('Run Training: :running:'):
            cls = ClassificationExperiment()
            cls.setup(data, target=y)
            setup_df = cls.pull()

            st.subheader('Setup Info: :clipboard:')
            st.dataframe(setup_df)
            best_model = cls.compare_models()
            compare_df = cls.pull()

            st.subheader('Compare Models: :chart_with_upwards_trend:')
            st.dataframe(compare_df)

            st.subheader('Evaluate Model: :chart_with_downwards_trend:')
            col1_ts_exp, col2_ts_exp = st.columns(2)
            with col1_ts_exp:
                cls.plot_model(best_model, plot='auc', display_format="streamlit")
            with col2_ts_exp:
                cls.plot_model(best_model, plot='confusion_matrix', display_format="streamlit")

            col1_ts_exp, col2_ts_exp = st.columns(2)
            with col1_ts_exp:
                cls.plot_model(best_model, plot='boundary', display_format="streamlit")
            with col2_ts_exp:
                cls.plot_model(best_model, plot='class_report', display_format="streamlit")
            cls.save_model(best_model, 'best_model')
            if st.button('Download Model: :floppy_disk:'):
                cls.save_model(best_model, 'best_model')
    elif exp_choice == "Regression":
        st.title('Regression Experiment: :chart_with_upwards_trend:')

        X = st.multiselect("Select Features", df.columns)
        y = st.selectbox("Select Target Variable", df.columns)

        data = df[X + [y]]

        st.title('Modeling: :rocket:')
        if st.button('Run Training: :running:'):
            reg = RegressionExperiment()
            reg.setup(data, target=y)
            setup_df = reg.pull()

            st.subheader('Setup Info: :clipboard:')
            st.dataframe(setup_df)
            best_model = reg.compare_models()
            compare_df = reg.pull()

            st.subheader('Compare Models: :chart_with_upwards_trend:')
            st.dataframe(compare_df)

            st.subheader('Evaluate Model: :chart_with_downwards_trend:')
            col1_ts_exp, col2_ts_exp = st.columns(2)
            with col1_ts_exp:
                reg.plot_model(best_model, plot='residuals', display_format="streamlit")
            with col2_ts_exp:
                reg.plot_model(best_model, plot='error', display_format="streamlit")

            col1_ts_exp, col2_ts_exp = st.columns(2)
            with col1_ts_exp:
                reg.plot_model(best_model, plot='learning', display_format="streamlit")
            with col2_ts_exp:
                reg.plot_model(best_model, plot='vc', display_format="streamlit")

            if st.button('Download Model: :floppy_disk:'):
                reg.download_model(best_model, 'best_model')
    elif exp_choice == "Clustering":
        st.title('Clustering Experiment: :chart_with_upwards_trend:')

        X = st.multiselect("Select X Features", df.columns)

        data = df[X]

        st.title('Modeling: :rocket:')
        if st.button('Run Training: :running:'):
            clu = ClusteringExperiment()
            clu.setup(data)
            setup_df = clu.pull()

            st.subheader('Setup Info: :clipboard:')
            st.dataframe(setup_df)

            st.subheader('Choose a Model: :chart_with_upwards_trend:')
            st.dataframe(clu.models())

            model_choice = st.selectbox("Select Model", clu.models().index.tolist(), index=0)

            model = clu.create_model(model_choice)

            assigned_clusters = clu.assign_model(model)
            st.dataframe(assigned_clusters)

            st.subheader('Analyze Model: :chart_with_downwards_trend:')

            col1_ts_exp, col2_ts_exp = st.columns(2)
            with col1_ts_exp:
                clu.plot_model(model, plot='cluster', display_format="streamlit")
            with col2_ts_exp:
                clu.plot_model(model, plot='tsne', display_format="streamlit")

            col1_ts_exp = st.columns(1)
            with col1_ts_exp:
                clu.plot_model(model, plot='elbow', display_format="streamlit")

            if st.button('Download Model: :floppy_disk:'):
                clu.save_model(model, 'best_model')
    elif exp_choice == "Anomaly Detection":
        st.title('Anomaly detection Experiment: :chart_with_upwards_trend:')

        X = st.multiselect("Select X's", df.columns)
        data = df[X]

        st.title('Modeling: :rocket:')
        if st.button('Run Training: :running:'):
            ano = AnomalyExperiment()
            ano.setup(data)
            setup_df = ano.pull()

            st.subheader('Setup Info: :clipboard:')
            st.dataframe(setup_df)

            st.subheader('Choose a Model: :chart_with_upwards_trend:')
            st.dataframe(ano.models())

            model_choice = st.selectbox("Select Model", ano.models().index.tolist(), index=3)

            model = ano.create_model(model_choice)

            assigned_clusters = ano.assign_model(model)
            st.dataframe(assigned_clusters)

            st.subheader('Analyze Model: :chart_with_downwards_trend:')

            col1_ts_exp, col2_ts_exp = st.columns(2)
            with col1_ts_exp:
                ano.plot_model(model, plot='umap', display_format="streamlit")

            if st.button('Download Model: :floppy_disk:'):
                ano.save_model(model, 'best_model')
