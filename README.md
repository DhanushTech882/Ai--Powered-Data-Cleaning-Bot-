# Ai--Powered-Data-Cleaning-Bot-
Overview
This Python script performs data cleaning and preprocessing for a customer dataset. It handles tasks like missing value imputation, duplicate removal, anomaly detection, and normalization of numerical features. The code is structured to ensure that the dataset is ready for further analysis or modeling.

Requirements
The following Python libraries are required for this script:

pandas (for data manipulation)
numpy (for numerical operations)
scikit-learn (for KNN imputation, anomaly detection, and normalization)
matplotlib (for plotting)
seaborn (for visualizations)
You can install the necessary libraries using pip:

bash
Copy
pip install pandas numpy scikit-learn matplotlib seaborn
Script Overview
Functions
load_dataset(file_path)

Loads a CSV dataset from the provided file path.
Prints the shape of the loaded dataset.
summarize_data(df)

Displays a summary of the dataset including data types, missing values, and duplicate rows.
Visualizes missing values in a heatmap.
handle_missing_values(df)

Handles missing values using KNN Imputer (K-Nearest Neighbors).
Imputes missing values in numerical columns and visualizes before and after the imputation.
remove_duplicates(df)

Removes duplicate rows from the dataset and returns the cleaned DataFrame.
detect_anomalies(df)

Detects anomalies using Isolation Forest algorithm.
Visualizes detected anomalies in a scatter plot.
normalize_data(df)

Normalizes numerical columns using StandardScaler.
Visualizes the dataset before and after normalization.
save_cleaned_data(df, output_path)

Saves the cleaned dataset to a specified output path.
main()

Main function that orchestrates the data cleaning process:
Loads the dataset.
Summarizes the dataset.
Handles missing values.
Removes duplicates.
Detects anomalies.
Normalizes the data.
Saves the cleaned dataset to the output path.
Usage
Replace "Cusromers_1.csv" with the path to your own dataset.
The output file will be saved as "outputs/cleaned_dataset.csv". Make sure the directory exists or change the path to your desired location.
Run the script by executing:
bash
Copy
python data_cleaning_script.py
This will execute the cleaning pipeline, generating visualizations for missing values, anomalies, and normalization, and finally saving the cleaned dataset.

Visualizations
The following visualizations are generated during the data cleaning process:

Missing Values Heatmap: Visual representation of missing data in the dataset.
Bar Plot of Missing Values Before/After Imputation: Shows the missing data percentage before and after KNN imputation.
Anomaly Detection Plot: Scatter plot showing detected anomalies in the dataset.
Before and After Normalization Histograms: Histograms of numerical features before and after normalization.
Notes
The script assumes the dataset is in CSV format.
The KNNImputer uses k=5 nearest neighbors by default, which can be adjusted according to your needs.
The IsolationForest is set to detect anomalies with a contamination factor of 0.1, which can be changed based on the dataset and requirements.
The dataset should ideally have some numerical features, as they are the focus of the preprocessing steps like missing value imputation and anomaly detection.
Conclusion
This pipeline provides a thorough approach to cleaning customer data, including handling missing values, removing duplicates, detecting anomalies, and normalizing numerical features. After running the script, you'll have a cleaned dataset ready for further analysis or machine learning modeling.
