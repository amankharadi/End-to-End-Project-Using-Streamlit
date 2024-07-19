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
#st.subheader("**File Uploader for EDA & Machine Learning Platform**")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display the filename
    st.write(f"Filename: {uploaded_file.name}")
    
    # Read and display the file content based on file type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), delimiter='\t')
    else:
        st.write("Unsupported file type")
        df = None

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
        # Apply selected method
        if missing_value_method != "None":
            if missing_value_method == "Mean":
                df_filled = df.fillna(df.mean(numeric_only=True))
            elif missing_value_method == "Median":
                df_filled = df.fillna(df.median(numeric_only=True))
            elif missing_value_method == "Mode":
                df_filled = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype != 'datetime64[ns]' else x)
            elif missing_value_method == "Drop":
                df_filled = df.dropna()

                    # Display the filled DataFrame
            st.write("DataFrame after handling missing values:")
            st.write("### DataFrame Shape",df_filled.shape)
            st.write(df_filled.head())
        else:
            df_filled = df


        # Get column names
        column_names = df_filled.columns.tolist()

        # Multiselect widget to pick columns
        selected_columns = st.sidebar.multiselect("Pick one or more columns to drop", column_names)
        st.sidebar.divider()
        # Drop selected columns
        if selected_columns:
            df_dropped = df_filled.drop(columns=selected_columns)
            st.markdown("**Dropped Unnecessary Columns**")
            st.write("Modified DataFrame:")
            st.write(df_dropped.head())
        else:
            df_dropped = df_filled
            st.markdown("**No columns selected to drop.**")
            st.caption("**Original DataFrame:**")
            st.write(df.head())

        st.divider()
        


        # Encoding section
        label_encoders = {}
        one_hot_encoders = {}

        not_encode_columns = st.sidebar.multiselect("Pick one or more columns to Not Encode", df_dropped.select_dtypes('object').columns.tolist())
        st.sidebar.divider()
        for column in df_dropped.select_dtypes('object').columns:
            if column in not_encode_columns:
                continue
            unique_values = df_dropped[column].unique()
            num_unique = len(unique_values)
            
            if num_unique > 0 and num_unique == 2:
                label_encoders[column] = LabelEncoder()
                df_dropped[column] = label_encoders[column].fit_transform(df_dropped[column])
            elif num_unique > 2:
                one_hot_encoders[column] = OneHotEncoder(sparse_output=False, drop='first')
                encoded_column = one_hot_encoders[column].fit_transform(df_dropped[[column]])
                df_encoded = pd.DataFrame(encoded_column, columns=one_hot_encoders[column].get_feature_names_out([column]))
                df_dropped = pd.concat([df_dropped.drop(column, axis=1), df_encoded], axis=1)
                
        # Display the transformed DataFrame
        st.markdown("### Transformed DataFrame:")
        st.write(df_dropped.head())
        st.divider()

        # Options for visualization
        vis_options = st.sidebar.multiselect("Pick one or more visualizations", ["Correlation Heatmap","Histogram", "Scatter Plot", "Pie Chart", "Line Plot", "Bar Plot", "Box Plot", "Heatmap", "Area Plot", "Violin Plot"])
        st.sidebar.divider()
        if "Histogram" in vis_options:
            st.write("Histogram")
            selected_column = st.sidebar.selectbox("Select column for histogram", df_dropped.columns)
            fig = px.histogram(df_dropped, x=selected_column, color_discrete_sequence=px.colors.qualitative.Dark24, title=f"Histogram of {selected_column}")
            st.plotly_chart(fig)
            
    
        if "Scatter Plot" in vis_options:
            st.write("Scatter Plot")
            x_column = st.sidebar.selectbox("Select column for x-axis Scatter Plot", df_dropped.columns, key="scatter_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Scatter Plot", df_dropped.columns, key="scatter_y")
            color_column = st.sidebar.selectbox("Select column for color Scatter Plot", df_dropped.columns)
            fig = px.scatter(df_dropped, x=x_column, y=y_column, color=color_column, title=f"Scatter Plot: {x_column} vs {y_column}")
            st.plotly_chart(fig)

        if "Correlation Heatmap" in vis_options:
            st.write("Correlation Heatmap")
            corr = df_dropped.corr()
            fig = px.imshow(corr, title="Correlation Heatmap")
            st.plotly_chart(fig)

        if "Pie Chart" in vis_options:
            st.write("Pie Chart")
            column = st.sidebar.selectbox("Select column for pie chart", df_dropped.columns)
            fig = px.pie(df_dropped, names=column, title=f"Pie Chart of {column}")
            st.plotly_chart(fig)

        if "Line Plot" in vis_options:
            st.write("Line Plot")
            x_column = st.sidebar.selectbox("Select column for x-axis Line Plot", df_dropped.columns, key="line_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Line Plot", df_dropped.columns, key="line_y")
            fig = px.line(df_dropped, x=x_column, y=y_column, title=f"Line Plot: {x_column} vs {y_column}")
            st.plotly_chart(fig)

        if "Bar Plot" in vis_options:
            st.write("Bar Plot")
            x_column = st.sidebar.selectbox("Select column for x-axis Bar Plot", df_dropped.columns, key="bar_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Bar Plot", df_dropped.columns, key="bar_y")
            fig = px.bar(df_dropped, x=x_column, y=y_column, title=f"Bar Plot: {x_column} vs {y_column}")
            st.plotly_chart(fig)

        if "Box Plot" in vis_options:
            st.write("Box Plot")
            x_column = st.sidebar.selectbox("Select column for x-axis Box Plot", df_dropped.columns, key="box_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Box Plot", df_dropped.columns, key="box_y")
            fig = px.box(df_dropped, x=x_column, y=y_column, title=f"Box Plot: {x_column} vs {y_column}")
            st.plotly_chart(fig)

        if "Heatmap" in vis_options:
            st.write("Heatmap")
            x_column = st.sidebar.selectbox("Select column for x-axis Heatmap", df_dropped.columns, key="heatmap_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Heatmap", df_dropped.columns, key="heatmap_y")
            z_column = st.sidebar.selectbox("Select column for z-axis Heatmap", df_dropped.columns, key="heatmap_z")
            fig = px.density_heatmap(df_dropped, x=x_column, y=y_column, z=z_column, title=f"Heatmap: {x_column} vs {y_column} vs {z_column}")
            st.plotly_chart(fig)

        if "Area Plot" in vis_options:
            st.write("Area Plot")
            x_column = st.sidebar.selectbox("Select column for x-axis Area Plot", df_dropped.columns, key="area_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Area Plot", df_dropped.columns, key="area_y")
            fig = px.area(df_dropped, x=x_column, y=y_column, title=f"Area Plot: {x_column} vs {y_column}")
            st.plotly_chart(fig)

        if "Violin Plot" in vis_options:
            st.write("Violin Plot")
            x_column = st.sidebar.selectbox("Select column for x-axis Violin Plot", df_dropped.columns, key="violin_x")
            y_column = st.sidebar.selectbox("Select column for y-axis Violin Plot", df_dropped.columns, key="violin_y")
            fig = px.violin(df_dropped, x=x_column, y=y_column, title=f"Violin Plot: {x_column} vs {y_column}")
            st.plotly_chart(fig)

        st.title("Regression and Classification Model Training")

            # Problem type selection
        problem_type = st.selectbox("Select problem type", ["Regression", "Classification"])

        # Target column selection
        target_column = st.selectbox("Select target column", df_dropped.columns)

        # Model options
        if problem_type == "Regression":
            model_options = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor()
            }
        else:
            model_options = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier']
            models = {
                'Logistic Regression': LogisticRegression(max_iter=200),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier()
            }

        # Sidebar for model selection
        selected_models = st.multiselect('Pick one or more models to evaluate', model_options)

        if selected_models:
            features = df_dropped.drop(columns=[target_column])
            target = df_dropped[target_column]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            results = []

            # Train and evaluate each selected model
            for model_name in selected_models:
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "Regression":
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results.append({
                        'Model': model_name,
                        'Mean Squared Error': mse,
                        'R^2 Score': r2
                    })
                else:
                    accuracy = accuracy_score(y_test, y_pred)
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()  # Convert report dictionary to DataFrame
                    results.append({
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'Classification Report': report_df.to_html()  # Convert DataFrame to HTML for display
                    })

            # Convert results to DataFrame for regression
            if problem_type == "Regression":
                results_df = pd.DataFrame(results)
                st.write("Model Performance for Regression")
                st.table(results_df)

            # Display results for classification
            if problem_type == "Classification":
                for result in results:
                    st.write(f"Model: {result['Model']}")
                    st.write(f"Accuracy: {result['Accuracy']}")
                    st.write("Classification Report:")
                    st.write(result['Classification Report'], unsafe_allow_html=True)

            else:
                st.write("Please upload a CSV file to proceed.")
                

        # Download Processed Data
        st.markdown("### Download Processed Data")
        csv = df_dropped.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='processed_data.csv', mime='text/csv')

st.divider()
st.caption("""<div style="text-align: center;"><h3>Manage By Aman Kharadi</h3></div>""", unsafe_allow_html=True)