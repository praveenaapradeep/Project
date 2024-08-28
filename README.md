Stock Market Prediction Analysis


Project Overview

In this project, I conducted a comprehensive analysis to predict stock prices for Google and Facebook using advanced deep learning techniques. The goal was to build models capable of forecasting future stock prices based on historical data, providing valuable insights for in-vestment decisions. The NYSE Securities dataset includes information on securities listed on the New York Stock Exchange (NYSE), such as their symbols, names, and other identifying details, making it valuable for stock market analysis and financial research.


Objectives

•	Model Performance: Aim to develop a model that generalizes well, is robust, and operates efficiently.

•	Model Comparison: Benchmark various models, understand trade-offs, and assess their real-world applicability.

•	Model Accuracy: Maximize accuracy, understand error types, and ensure consistent performance across different scenarios.

•	Dataset Utilization: Ensure comprehensive coverage, high quality, appropriate aug-mentation, and balanced representation of the data.


Dataset Description

•	ID (integer): A unique identifier for each security, used for indexing and referenc-ing.

•	Symbol (string): The ticker symbol for the security, a unique code identifying pub-licly traded securities.

•	Name (string): The full name of the company associated with the security.

•	Market (string): The market where the security is listed (e.g., NYSE).

•	Sector (string): The sector in which the company operates (e.g., Technology, Healthcare).

•	Industry (string): The specific industry classification of the company (e.g., Biotech-nology, Software).

•	Country (string): The country where the company is based.

•	Currency (string): The currency in which the security is traded (e.g., USD for NYSE).

•	Date (date): The date when the security information was recorded or last updated.

•	Description (string): Additional information about the security, including business activities or relevant notes.


Model Description

•	Gated Recurrent Unit (GRU): GRU captures temporal dependencies in sequential data, making it effective for learning patterns from historical stock prices.

•	Convolutional Neural Network (CNN): CNN extracts features from time series da-ta, detecting local patterns and trends to enhance stock price prediction accuracy.


Evaluation Metrics

•	Root Mean Squared Error (RMSE): Measures the average magnitude of prediction errors, penalizing larger errors more than smaller ones.

•	Root Mean Absolute Error (RMAE): Provides the average absolute differences be-tween predicted and actual values, treating all errors equally.

•	Loss: Quantifies the difference between predicted values and actual values during model training, guiding the optimization process.


Prerequisites

•	Python 3.8+

•	Mathematics and Statistics: Strong foundation in linear algebra, calculus, probabil-ity, and statistics for understanding and implementing machine learning algorithms.

•	Programming Skills: Proficiency in Python or R for coding, data manipulation, and using machine learning libraries and frameworks.

•	Data Handling Skills: Experience with data manipulation, cleaning, and prepro-cessing using tools like pandas and SQL for preparing datasets.


Project Structure

•	train_model.py: Script to train the machine learning models.

•	gui.py: Script to create plots for visualizing model performance.

•	requirements.txt: List of required Python packages for the project.

•	data/: Directory containing raw and processed datasets.

•	models/: Directory containing saved model files.


RESULTS VISUALIZATION

•	The result are clearly represent using plots that compare the prediction of different company using different models.


Known Issues

•	Model performance may vary based on the quality and quantity of historical data.

•	The project currently does not handle missing values in the dataset; data prepro-cessing may need to be adjusted for different datasets.


Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have sug-gestions or improvements. Ensure that all changes are accompanied by appropriate tests and documentation.


License

This project is licensed under the MIT License. See the LICENSE file for more details.


Acknowledgement

This project is part of the MSc Data Science program at the University of Hertfordshire. Special gratitude to Niall Miller for supervising this project.



