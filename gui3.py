# -*- coding: utf-8 -*-
"""gui3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vwzsTcUhSYe-uD3TK0lYMpxttF-kKjKj

# IMPORTING LIBRARIESimport tkinter as tk
"""

# IMPORTING LIBRARIESimport tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

"""# HELPER FUNCTIONS"""

def go_to_check_tab():
    """
    Switches to the 'Check' tab when called.
    """
    notebook.select(check_tab)

def rmse(y_true, y_pred):
    """
    Calculates Root Mean Square Error (RMSE) between true and predicted values.

    Args:
        y_true (tf.Tensor): Actual values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: RMSE value.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def rmae(y_true, y_pred):
    """
    Calculates Root Mean Absolute Error (RMAE) between true and predicted values.

    Args:
        y_true (tf.Tensor): Actual values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: RMAE value.
    """
    return tf.sqrt(tf.reduce_mean(tf.abs(y_pred - y_true)))

def predict_next_days(model, X_test, scaler, num_days):
    """
    Predicts the next 'num_days' of closing prices using the provided model and test data.

    Args:
        model (tf.keras.Model): Trained neural network model for prediction.
        X_test (np.ndarray): Test dataset used as the input sequence for prediction.
        scaler (sklearn.preprocessing.StandardScaler): Scaler object to inverse transform predictions.
        num_days (int): Number of future days to predict.

    Returns:
        np.ndarray: Array of predicted prices for the next 'num_days'.
    """
    predicted = []

# Start prediction from the last sequence in X_test
    input_sequence = X_test[-1].reshape(1, 1, -1)

    for _ in range(num_days):
        # Predict next day's close price
        next_day_pred = model.predict(input_sequence)
        predicted.append(next_day_pred[0, 0])
        # Update input sequence with the new prediction
        input_sequence = np.append(input_sequence[:, :, 1:], next_day_pred.reshape(1, 1, 1), axis=2)

    # Inverse transform the predictions to get actual values
    predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
    return predicted_prices.flatten()

"""# LOADING MODELS AND SCALER"""

# Load pre-trained models
facebook_model = tf.keras.models.load_model("fb_model.h5", compile=False)
google_model = tf.keras.models.load_model("google_model.h5", compile=False)

# Load scaler object for inverse transformation
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

"""# GUI SETUP"""

# Initialize the main application window
top = tk.Tk()
top.geometry("900x600")
top.title("Stock Price Forecasting")

# Create Notebook for tabs
notebook = ttk.Notebook(top)
notebook.pack(pady=10, expand=True)

# Create frames for Home and Check tabs
home_tab = ttk.Frame(notebook, width=900, height=550)
check_tab = ttk.Frame(notebook, width=900, height=550)

# Add frames to notebook
notebook.add(home_tab, text='Home')
notebook.add(check_tab, text='Check')

# Set background image for the Home tab
image_path = 'stock.png'
image = Image.open(image_path)
image = image.resize((900, 550), Image.BICUBIC)
background_image = ImageTk.PhotoImage(image)

# Place background image
background_label = tk.Label(home_tab, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add 'Get Started' button to navigate to the Check tab
home_button = tk.Button(
    home_tab, text="Get Started", font=("Arial", 12, "bold"),
    background="black", width=20, fg="white", command=go_to_check_tab
)
home_button.place(relx=0.5, rely=0.6, anchor='center')

"""# FORECAST FUNCTION"""

def forecast():
    """
    Handles the forecast button click, predicting stock prices based on selected stock
    and the number of days specified by the user. Displays the predicted prices on a bar chart.
    """
    global X_test  # Define X_test globally to access it in the function

    # Determine which model to use based on selected radio button
    selected_model = None
    if var.get() == 1:  # Google Stock selected
        selected_model = google_model
        with open('google.pkl', 'rb') as f:
            X_test = pickle.load(f)
    elif var.get() == 2:  # Facebook Stock selected
        selected_model = facebook_model
        with open('facebook_X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
    else:
        messagebox.showerror("Error", "Please select a stock model")
        return

    # Perform forecasting
    num_days = int(days_var.get())
    predicted_prices = predict_next_days(selected_model, X_test, scaler, num_days)

# Plot the predicted prices with annotations
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(1, num_days + 1), predicted_prices, color='blue')
    ax.set_xlabel('Day')
    ax.set_ylabel('Predicted Close Price')
    ax.set_title('Predicted Close Prices for Next Days')

    # Add annotations on top of each bar
    for bar, price in zip(bars, predicted_prices):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval, f'${price:.2f}',
            ha='center', va='bottom', fontsize=10, color='black'
        )

    ax.set_xticks(range(1, num_days + 1))
    fig.tight_layout()

    # Display the plot
    plt.show()

"""# GUI ELEMENTS FOR CHECK TAB"""

# Create radio buttons for selecting the stock model
var = tk.IntVar()

google_radio = tk.Radiobutton(
    check_tab, font=("Arial", 16, "bold"), bg="green",
    text="Google Stock", variable=var, value=1,
    highlightthickness=0, bd=0
)
google_radio.place(relx=0.38, rely=0.3, anchor='center')

facebook_radio = tk.Radiobutton(
    check_tab, font=("Arial", 16, "bold"), bg="green",
    text="Facebook Stock", variable=var, value=2,
    highlightthickness=0, bd=0
)
facebook_radio.place(relx=0.62, rely=0.3, anchor='center')

# Dropdown menu for selecting the number of prediction days
days_var = tk.StringVar()
days_label = tk.Label(
    check_tab, text="Select number of days:", bg="light blue",
    fg="black", font=("Arial", 14, "bold")
)
days_label.place(relx=0.5, rely=0.4, anchor='center')

days_dropdown = ttk.Combobox(
    check_tab, textvariable=days_var, font=("Arial", 14, "bold"),
    values=[1, 2, 3, 4, 5], state="readonly", width=20
)
days_dropdown.place(relx=0.5, rely=0.5, anchor='center')
days_dropdown.current(0)  # Set default value

# Forecast button to trigger the prediction
predict_button = tk.Button(
    check_tab, text="Forecast", font=("Arial", 10, "bold"),
    background="black", activebackground="green",
    width=20, fg="white", command=forecast
)
predict_button.place(relx=0.5, rely=0.68, anchor='center')

# Title label on the Check tab
head_title = tk.Label(
    check_tab, text="Stock Price Forecasting", font=("Arial", 24, "bold"),
    background="light blue", fg="black"
)
head_title.place(relx=0.5, rely=0.1, anchor="center")

# Placeholder label for selected images (currently unused)
selected_image_label = tk.Label(check_tab, bg='light blue')
selected_image_label.place(relx=0.5, rely=0.6, anchor='center')

# Configure style for ttk frames
style = ttk.Style()
style.configure('TFrame', background='light blue')

"""# RUN THE APPLICATION"""

top.mainloop()