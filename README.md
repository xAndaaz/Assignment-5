# Housing Price Prediction

This project demonstrates a complete data science workflow for predicting housing prices on the Ames Housing dataset. It showcases skills in exploratory data analysis (EDA), feature engineering, data preprocessing, and predictive modeling.

## Project Overview

The primary goal of this project is to build a model that accurately predicts the sale price of houses. The project is structured as a single Python script (`main.py`) that performs the following steps:

1.  **Exploratory Data Analysis (EDA):** The script begins by analyzing the target variable, `SalePrice`, to understand its distribution. It identifies and corrects for right skewness using a log transformation.
2.  **Feature Engineering:** New, insightful features are created from the existing data to improve model performance. These include:
    *   `TotalSF`: Total square footage of the house.
    *   `TotalBath`: Total number of bathrooms.
    *   `TotalPorchSF`: Total square footage of all porches.
3.  **Data Cleaning:** The script handles missing values by filling them with appropriate values (e.g., 'None' for categorical features, 0 for numerical features, and the median for `LotFrontage`).
4.  **Feature Selection:** A correlation matrix and heatmap are used to identify the features most correlated with `SalePrice`.
5.  **Predictive Modeling:** A Ridge regression model is trained on the processed data. The model's performance is evaluated using the Root Mean Squared Error (RMSE) on a validation set.

## Getting Started

### Prerequisites

*   Python 3.x
*   The required Python libraries are listed in the `requirements.txt` file.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/xAndaaz/Assignment-5.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Assignment-5
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the entire data analysis and modeling pipeline, execute the `main.py` script:

```bash
python main.py
```
##Additionally, if you want to test more, use jupytext to convert `main.py` to `main.ipynb`
using 
```bash 
jupytext --to notebook main.py
```

The script will generate several plots to visualize the data and the model's performance.

## Skills Demonstrated

*   **Data Preprocessing:** Handling missing data, feature scaling, and one-hot encoding.
*   **Feature Engineering:** Creating new features to improve model accuracy.
*   **Exploratory Data Analysis (EDA):** Analyzing and visualizing data to gain insights.
*   **Predictive Modeling:** Training and evaluating a Ridge regression model.
*   **Python Libraries:** Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
