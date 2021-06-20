import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.manifold import TSNE

import plotly.express as px

from algorithms import algorithms

st.title("Clustering apps")
st.text(
"""
CSVファイルをアップロードしてください。
"""
)
uploaded_file = st.file_uploader("Choose a file")
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

if df is not None:
    with st.form("説明変数"):
        ex_val = st.multiselect("使用する変数を選んでください", [str(c) for c in df.columns]
        , default=[str(c) for c in df.columns])
        exp_submitted = st.form_submit_button("confirm")

        if exp_submitted is not None:
            df = df.loc[:, ex_val]
            df

algorithm_dict = algorithms()

with st.form("algorithm"):
    algorithm = st.selectbox("alogrithm", (
        [alg for alg in algorithm_dict.keys()]
    ))
    algorithm_submit = st.form_submit_button("決定") 

if algorithm_submit is not None:
    parameters = None
    if algorithm == "KMeans":
        parameters = {}
        with st.form("kmeans_parameter"):
            n_clusters = st.slider("クラスタ数", min_value=2, max_value=100, step=1)
            parameters["n_clusters"] = n_clusters
            param_submitted = st.form_submit_button("決定")
    elif algorithm == "DBSCAN":
        parameters = {}
        with st.form("dbscan_parameter"):
            eps = st.slider("epsilon", min_value=0.1, max_value=10.0, step=0.1)
            parameters["eps"] = eps
            param_submitted = st.form_submit_button("決定")

if param_submitted is not None:
    clu = algorithm_dict[algorithm].make_instance(**parameters)

if df is not None and clu is not None:
    st.write(f"""
    - algorithm: {algorithm}
    - parameters: {parameters}
    """)
    if st.button("analyze"):
        res = [str(r) for r in clu.fit_predict(df)]
        tsne = TSNE(n_components=2)
        coor = tsne.fit_transform(df)
        fig = px.scatter(coor, color=res, hover_name=res)    
        st.plotly_chart(fig)
