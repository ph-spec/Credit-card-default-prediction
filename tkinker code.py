#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import xgboost as xgb


model = None  


def load_model():
    global model
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Updated_xgboost.model")
    
    try:
        if os.path.exists(desktop_path):
            model = xgb.Booster()
            model.load_model(desktop_path)
            
            messagebox.showinfo("Model Loaded", "Model loaded successfully!")
    except Exception as e:
            messagebox.showerror("Loading Error", f"Error loading model: {e}")
    else:
        messagebox.showwarning("No File", "No file selected.")


def predict_default():
    try:
        if model is None:
            raise ValueError("Model is not loaded. Please upload a model file.")

        #  user inputs
        LIMIT_BAL = entry_LIMIT_BAL.get()
        PAY_AMT1 = entry_PAY_AMT1.get()
        PAY_AMT2 = entry_PAY_AMT2.get()
        PAY_AMT3 = entry_PAY_AMT3.get()
        PAY_AMT4 = entry_PAY_AMT4.get()
        PAY_AMT5 = entry_PAY_AMT5.get()
        PAY_AMT6 = entry_PAY_AMT6.get()
        PAY_0 = entry_PAY_0.get()
        PAY_2 = entry_PAY_2.get()
        PAY_3 = entry_PAY_3.get()
        PAY_4 = entry_PAY_4.get()    
        PAY_5 = entry_PAY_5.get()
        PAY_6 = entry_PAY_6.get()
        SEX = entry_SEX.get()
        MARRIAGE = entry_MARRIAGE.get()
        EDUCATION = entry_EDUCATION.get()
        BILL_AMT6 = entry_BILL_AMT6.get()
        BILL_AMT1 = entry_BILL_AMT1.get()
        AGE = entry_AGE.get()

        
        if not LIMIT_BAL or not PAY_AMT1 or not PAY_AMT2 or not PAY_AMT3 or not PAY_AMT4 or not PAY_AMT5 or not PAY_AMT6:
            raise ValueError("All fields must be filled in with numeric values.")

        LIMIT_BAL = float(LIMIT_BAL)
        PAY_AMT1 = float(PAY_AMT1)
        PAY_AMT2 = float(PAY_AMT2)
        PAY_AMT3 = float(PAY_AMT3)
        PAY_AMT4 = float(PAY_AMT4)
        PAY_AMT5 = float(PAY_AMT5)
        PAY_AMT6 = float(PAY_AMT6)
        PAY_0 = float(PAY_0)
        PAY_2 = float(PAY_2)
        PAY_3 = float(PAY_3)
        PAY_4 = float(PAY_4)
        PAY_5 = float(PAY_5)
        PAY_6 = float(PAY_6)
        SEX = float(SEX)
        MARRIAGE = float(MARRIAGE)
        EDUCATION = float(EDUCATION)
        BILL_AMT6 = float(BILL_AMT6)
        BILL_AMT1 = float(BILL_AMT1)
        AGE = float(AGE)
        
      
        features = np.array([LIMIT_BAL, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, SEX, MARRIAGE, EDUCATION, BILL_AMT1, BILL_AMT6, AGE]).reshape(1, -1)

      
        dmatrix_features = xgb.DMatrix(features)
        
       
        prediction = model.predict(dmatrix_features)
        
       
        if prediction[0] == 1:
            result = "The credit card will default."
        else:
            result = "The credit card will not default."
    
        messagebox.showinfo("Prediction Result", result)
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Error: {ve}. Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred: {e}")


root = tk.Tk()
root.title("Credit Card Default Prediction")


menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Model", command=load_model)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)


labels = ["LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
          "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "SEX", "MARRIAGE", "EDUCATION",
          "BILL_AMT6", "BILL_AMT1", "AGE"]

entries = {}

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries[label] = entry

entry_LIMIT_BAL = entries["LIMIT_BAL"]
entry_PAY_AMT1 = entries["PAY_AMT1"]
entry_PAY_AMT2 = entries["PAY_AMT2"]
entry_PAY_AMT3 = entries["PAY_AMT3"]
entry_PAY_AMT4 = entries["PAY_AMT4"]
entry_PAY_AMT5 = entries["PAY_AMT5"]
entry_PAY_AMT6 = entries["PAY_AMT6"]
entry_PAY_0 = entries["PAY_0"]
entry_PAY_2 = entries["PAY_2"]
entry_PAY_3 = entries["PAY_3"]
entry_PAY_4 = entries["PAY_4"]
entry_PAY_5 = entries["PAY_5"]
entry_PAY_6 = entries["PAY_6"]
entry_SEX = entries["SEX"]
entry_MARRIAGE = entries["MARRIAGE"]
entry_EDUCATION = entries["EDUCATION"]
entry_BILL_AMT6 = entries["BILL_AMT6"]
entry_BILL_AMT1 = entries["BILL_AMT1"]
entry_AGE = entries["AGE"]

#Add a button to trigger the prediction
tk.Button(root, text="Predict", command=predict_default).grid(row=len(labels), column=1)

# Run the Tkinter loop
root.mainloop()


# In[ ]:




