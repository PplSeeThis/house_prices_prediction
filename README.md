# House Price Prediction for King County, USA

This project is a complete end-to-end machine learning pipeline for predicting house prices based on the King County House Sales dataset from Kaggle. The goal is to explore the data, engineer relevant features, and compare the performance of classical machine learning models against a simple neural network.

---
## 📜 Project Workflow

The project is divided into four main stages, each contained within its own Jupyter Notebook for clarity and modularity.

### **1. Exploratory Data Analysis (`1_EDA.ipynb`)**
* Loaded the raw data and performed an initial inspection.
* Generated a correlation heatmap to identify key relationships between features and the target variable (`price`).
* Visualized feature distributions and relationships using histograms and scatter plots.
* **Key Insight:** `sqft_living`, `grade`, and geographical coordinates (`lat`, `long`) showed the strongest initial correlation with price.

### **2. Feature Engineering & Cleaning (`2_Feature_Engineering.ipynb`)**
* **Outlier Handling:** Removed significant outliers based on the Interquartile Range (IQR) method for `sqft_living` to create a more robust dataset.
* **Feature Creation:** Engineered new, more informative features like `house_age`.
* **Categorical Encoding:** Applied One-Hot Encoding to the `zipcode` column to treat each zip code as a distinct location.
* **Data Scaling:** Standardized all numerical features using `StandardScaler` to prepare them for modeling.
* **Data Leakage Discovery:** Initially, a feature `price_per_sqft` was created, leading to unrealistically high model performance (R² ≈ 1.0). This feature was removed to fix the data leakage, which was a critical learning step in the project.

### **3. Classical Model Training (`3_Modeling.ipynb`)**
* Split the prepared data into training (80%) and testing (20%) sets.
* Trained and evaluated three different regression models:
    1.  **Linear Regression** (as a baseline)
    2.  **Random Forest Regressor**
    3.  **Gradient Boosting Regressor**
* Used 5-fold cross-validation to get a reliable estimate of each model's performance.
* Analyzed feature importances from the best-performing model.

### **4. Neural Network (`4_Neural_Network.ipynb`)**
* Built a simple feed-forward neural network (4 layers) using PyTorch.
* Prepared the data using PyTorch Tensors and DataLoaders.
* Implemented a full training loop to train the network for 50 epochs.
* Plotted learning curves to visualize the training process.
* Evaluated the final network and compared its performance to the classical models.

---
## 🏆 Results & Key Findings

* The best performing classical model was the **Random Forest Regressor**, achieving a cross-validated Mean Absolute Error (MAE) of **~$63,573**.
* The simple Neural Network performed well but did not outperform the Random Forest, yielding an MAE of **~$66,176**.
* Feature importance analysis revealed that the top 3 predictors for house price are:
    1.  **`grade`** (quality of construction)
    2.  **`lat`** (location on the North-South axis)
    3.  **`sqft_living`** (the size of the living space)

This confirms the real-world intuition that the final price is determined by **Quality > Location > Size**.

---
## 🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pplseethis/house_prices_prediction.git
    cd house_prices_prediction
    ```

2.  **Set up the environment:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal after installing the libraries).*

3.  **Set up Kaggle API:**
    * Download your `kaggle.json` API token from your Kaggle account page.
    * Place the `kaggle.json` file inside a `C:\Users\your_username\.kaggle\` folder.

4.  **Run the notebooks:**
    Execute the Jupyter Notebooks in sequential order for the full pipeline:
    1.  `1_EDA.ipynb`
    2.  `2_Feature_Engineering.ipynb`
    3.  `3_Modeling.ipynb`
    4.  `4_Neural_Network.ipynb`

---
## 🛠️ Technologies Used

* Python 3.13
* Jupyter Notebook (built in PyCharm 2025.2.0.1)
* Pandas
* NumPy
* Matplotlib & Seaborn
* Scikit-learn
* PyTorch
* Kaggle API
