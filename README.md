# SMART-ML: Machine Learning Made Easy!

SMART-ML is a user-friendly Streamlit application that simplifies machine learning tasks on datasets. With SMART-ML, you can easily upload your dataset, perform exploratory data analysis (EDA), and train machine learning models for classification, regression, clustering, or anomaly detection. The application provides a convenient interface for data scientists, researchers, and enthusiasts to quickly experiment with different machine learning techniques.

![image](https://github.com/ozzmanmuhammad/SMART-ML/assets/93766242/04228fc9-a058-42a9-af09-7e8cc083b80a)

## Live Application
You can also access the live version of the application hosted on Streamlit Sharing by visiting https://smart-ml.streamlit.app/.

## Features

- **Dataset Management**: Upload your CSV dataset and visualize its contents.
- **Exploratory Data Analysis (EDA)**: Perform comprehensive EDA using Pandas Profiling to gain insights into your dataset.
- **Modeling**: Train machine learning models for classification, regression, clustering, or anomaly detection using the PyCaret library.
- **Model Comparison**: Compare and evaluate multiple models to identify the best performing one.
- **Interactive Visualizations**: Generate interactive visualizations to analyze model performance and make informed decisions.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your_repository
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run main.py
   ```

2. Access the application in your browser at `http://localhost:8501`.

## Getting Started

1. **Upload Your Dataset**: Select the "Dataset" option in the navigation panel and upload your CSV dataset file. The application will display the contents of the dataset.

2. **Perform EDA**: Choose the "EDA" option in the navigation panel to perform exploratory data analysis on your dataset. The application utilizes Pandas Profiling to generate a comprehensive report with statistical summaries, data visualizations, and correlation analysis.

3. **Train Machine Learning Models**: Select the "Modeling" option in the navigation panel to train machine learning models. Choose the type of experiment you want to perform (classification, regression, clustering, or anomaly detection). Specify the features and target variable from your dataset.

4. **Evaluate and Compare Models**: After training the models, you can compare their performance using various evaluation metrics. The application provides visualizations such as ROC curves, confusion matrices, and class reports to facilitate model evaluation.

5. **Download Trained Models**: You have the option to download the best-performing model in each experiment type. Simply click the "Download Model" button to save the model locally.

## Contributing

Contributions to SMART-ML are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. We appreciate your feedback and contributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
