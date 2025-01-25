Stock Price Prediction Using Keras and Recurrent Neural Networks (RNN)

Overview

This project leverages Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) layers to predict stock prices based on historical data. The model provides a robust approach to time-series forecasting by capturing temporal dependencies in stock market trends. This project includes data preprocessing, feature engineering, model training, and performance visualization.

Key Features

Historical Data Analysis: Preprocesses and scales stock price data for effective input into the model.

Deep Learning Architecture: Implements a multi-layered LSTM-based RNN.

Time-Series Forecasting: Accurately predicts future stock prices based on historical trends.

Visualization: Generates insightful plots comparing real vs. predicted stock prices.

Outlier Handling: Detects and removes outliers using box plots and IQR analysis.

Customizable Framework: Adaptable for other datasets and forecasting problems.

Project Workflow

1. Data Preprocessing

Import historical stock price data from Google_Stock_Price_Train.csv.

Perform feature scaling using MinMaxScaler to normalize values between 0 and 1.

Create time-series data sequences for training with a sliding window approach.

2. Exploratory Data Analysis (EDA)

Generate histograms to understand the data distribution and identify outliers.

Use box plots to visualize and handle outliers using the IQR method.

3. Model Building

Build an RNN model with the following architecture:

LSTM layers to capture temporal dependencies.

Dropout layers for regularization and overfitting prevention.

Dense layers for output predictions.

Compile the model using the Adam optimizer and mean squared error (MSE) as the loss function.

4. Model Training

Train the model on the prepared dataset for 100 epochs with a batch size of 32.

Evaluate model performance on the test set (Google_Stock_Price_Test.csv).

5. Performance Evaluation

Visualize real vs. predicted stock prices using Matplotlib.

Assess prediction accuracy with metrics such as RMSE (Root Mean Squared Error).

File Structure

📂 Stock-Price-Prediction
├── PROJ.ipynb              # Main Jupyter Notebook for the project
├── Google_Stock_Price_Train.csv  # Training dataset
├── Google_Stock_Price_Test.csv   # Testing dataset
├── Filtered_Stock_Data.csv       # Processed dataset after outlier removal
├── requirements.txt        # List of required Python libraries
└── README.md               # Project documentation (this file)

Installation

Clone the repository:

git clone https://github.com/your-username/Stock-Price-Prediction.git
cd Stock-Price-Prediction

Install the required Python libraries:

pip install -r requirements.txt

Ensure that the dataset files (Google_Stock_Price_Train.csv and Google_Stock_Price_Test.csv) are in the project directory.

Usage

Open the PROJ.ipynb file in Jupyter Notebook:

jupyter notebook PROJ.ipynb

Run each cell in the notebook to:

Preprocess the data.

Train the LSTM model.

Evaluate the model's performance.

Visualize the predictions.

Modify the code for additional experiments, such as changing hyperparameters or using different datasets.

Results

The LSTM-based model demonstrates its ability to capture stock price trends and make accurate predictions.

Insights:

Real vs. predicted stock prices are visualized to assess performance.

Outlier handling improves model reliability.

Customization

Extend Dataset: Replace Google_Stock_Price_Train.csv and Google_Stock_Price_Test.csv with your own datasets.

Adjust Hyperparameters: Experiment with LSTM units, dropout rates, batch size, and epochs.

Feature Engineering: Add additional technical indicators or external factors like sentiment analysis.

Technologies Used

Python Libraries:

NumPy: Numerical computations

Pandas: Data manipulation and preprocessing

Matplotlib: Visualization

Keras (with TensorFlow backend): Building and training the deep learning model

Scikit-learn: Feature scaling (MinMaxScaler)

Contribution

Contributions are welcome! To contribute:

Fork the repository.

Create a new branch:

git checkout -b feature-name

Commit your changes:

git commit -m "Add a meaningful commit message"

Push the branch:

git push origin feature-name

Open a Pull Request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Dataset: Google Stock Price data.

Inspiration: Hands-on Machine Learning and Deep Learning tutorials.

