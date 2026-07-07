import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import csv

def load_data():
    with open('activities.csv', 'r') as file:
        reader = csv.reader(file)
        headers = [h.strip() for h in next(reader)]
        rows = list(reader)
        
    df = pd.DataFrame(rows, columns=headers, dtype='object')
    for col in df.columns:
        if df[col].dtype == 'object'
        
load_data()