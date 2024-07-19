import streamlit as st
import pandas as pd
import io
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page configuration to wider layout
st.set_page_config(layout="wide")
st.cache_data()

# Set the title of the Streamlit app
st.title("🚀All-in-One EDA and Machine Learning Platform")
st.caption('''**Our All-in-One EDA (Exploratory Data Analysis) and Machine Learning Platform is a comprehensive,
            user-friendly web application designed to streamline the entire data analysis and machine learning workflow. 
            Built with modern technologies such as Streamlit, Pandas, Plotly, and Scikit-Learn, this platform 
            empowers users to easily upload datasets, explore data, perform transformations, visualize insights, 
            and train machine learning models—all within a single, integrated environment.**''')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display the filename
    st.write(f"Filename: {uploaded_file.name}")
    
    @st.cache_data
    def load_data(uploaded_file):
        # Read and display the file content based on file type
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode("utf-8")
            return pd.read_csv(io.StringIO(content), delimiter='\t')
        else:
            st.write("Unsupported file type")
            return None
    
    df = load_data(uploaded_file)

    if df is not None:
        st.write(df.head())
        st.divider()

        # Options for the multiselect
        options = st.sidebar.multiselect("Pick one or more", ["Data Info", "Data Null Values", "Data Duplicates", "Data Describe"])
        st.sidebar.divider()
        
        # Create columns for layout
        col1, col2 = st.columns(2)

        # Display selected options in columns
        with col1:
            if "Data Info" in options:
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text("DataFrame Info:")
                st.text(s)

            if "Data Duplicates" in options:
                duplicates = df[df.duplicated()]
                st.write("Duplicate Rows in DataFrame:")
                st.table(duplicates)

        with col2:
            if "Data Null Values" in options:
                null_values = df.isnull().sum().reset_index()
                null_values.columns = ['Column', 'Null Count']
                st.write("Null Values in DataFrame:")
                st.table(null_values)

            if "Data Describe" in options:
                st.write("Describe DataFrame")
                st.table(df.describe())

        missing_value_method = st.sidebar.selectbox("Select missing value handling method", ["None", "Mean", "Median", "Mode", "Drop"])
        st.sidebar.divider()
        
        @st.cache_data
        def handle_missing_values(df, method):
            if method == "Mean":
                return df.fillna(df.mean(numeric_only=True))
            elif method == "Median":
                return df.fillna(df.median(numeric_only=True))
            elif method == "Mode":
                return df.fillna(df.mode().iloc[0])
            elif method == "Drop":
                return df.dropna()
            else:
                return df

        df_filled = handle_missing_values(df, missing_value_method)

        st.write("Data after handling missing values:")
        st.write(df_filled.head())

        @st.cache_data
        def encode_data(df):
            label_encoders = {}
            one_hot_encoders = {}
            encoded_df = df.copy()

            for column in encoded_df.select_dtypes(include=['object']).columns:
                label_encoders[column] = LabelEncoder()
                encoded_df[column] = label_encoders[column].fit_transform(encoded_df[column])

                one_hot_encoders[column] = OneHotEncoder(sparse_output=False, drop='first')
                encoded_array = one_hot_encoders[column].fit_transform(encoded_df[[column]])
                encoded_df = encoded_df.drop(column, axis=1)
                encoded_df = pd.concat([encoded_df, pd.DataFrame(encoded_array, columns=one_hot_encoders[column].get_feature_names_out([column]))], axis=1)
            
            return encoded_df, label_encoders, one_hot_encoders

        encoded_df, label_encoders, one_hot_encoders = encode_data(df_filled)

        st.write("Encoded DataFrame:")
        st.write(encoded_df.head())

        # Split the data into features and target variable
        st.sidebar.write("### Split Data")
        target_column = st.sidebar.selectbox("Select Target Column", encoded_df.columns)
        X = encoded_df.drop(columns=[target_column])
        y = encoded_df[target_column]

        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.sidebar.number_input("Random State", min_value=0, value=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        st.write("Training and Testing Data Shapes")
        st.write(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        st.write(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Model selection and training
        st.sidebar.write("### Select Model")
        model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Logistic Regression", "Decision Tree Regressor", "Decision Tree Classifier", "Random Forest Regressor", "Random Forest Classifier"])

        model = None
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif model_name == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor()
        elif model_name == "Random Forest Classifier":
            model = RandomForestClassifier()

        if model is not None:
            with st.spinner("Training the model..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            if model_name in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]:
                st.write("### Regression Model Performance")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
                st.write(f"R2 Score: {r2_score(y_test, y_pred)}")
            else:
                st.write("### Classification Model Performance")
                st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
                st.write(f"Classification Report: \n{classification_report(y_test, y_pred)}")
else:
    st.write("Please upload a file to get started.")
