import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def upload_and_display_csv(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def select_columns(df, selected_columns):
    if selected_columns:
        df_selected = df[selected_columns]
        return df_selected
    return None

def preprocess_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    encoder = None
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
        df_encoded.columns = encoder.get_feature_names_out(categorical_columns)
        df = pd.concat([df.drop(columns=categorical_columns), df_encoded], axis=1)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, encoder, scaler, categorical_columns

def determine_optimal_clusters(df_scaled):
    silhouette_scores = []
    cluster_range = range(2, 11)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))

    optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_clusters, silhouette_scores

def perform_clustering(df_selected, df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_selected['cluster'] = kmeans.fit_predict(df_scaled)
    
    # Filter numeric columns for calculating the mean
    numeric_columns = df_selected.select_dtypes(include=['number']).columns.tolist()
    cluster_means = df_selected.groupby('cluster')[numeric_columns].mean()
    
    return kmeans, df_selected, cluster_means

def classify_new_data_clustering(df, selected_columns, kmeans, encoder, scaler, categorical_columns, cluster_means, input_data):
    input_df = pd.DataFrame([input_data])

    # Ensure all categorical columns are properly encoded
    if categorical_columns:
        for col in categorical_columns:
            if col in input_df.columns:
                input_df_encoded = pd.DataFrame(encoder.transform(input_df[[col]]), columns=encoder.get_feature_names_out([col]))
                input_df = pd.concat([input_df.drop(columns=[col]), input_df_encoded], axis=1)
            else:
                # If a categorical column is missing in input_df, fill it with 0 (assuming no selection)
                input_df[col] = 0  # Or use appropriate default value

    # Ensure input_df has all columns used during training, dropping 'cluster'
    input_df = input_df.reindex(columns=df.columns.drop('cluster'), fill_value=0)

    # Scale the input data
    input_df_scaled = scaler.transform(input_df)

    # Predict the cluster
    predicted_cluster = kmeans.predict(input_df_scaled)[0]

    return predicted_cluster, cluster_means.loc[predicted_cluster]


