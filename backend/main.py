from fastapi import FastAPI, File, UploadFile
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Data(BaseModel):
    data: List[List[float]]

def load_default_data():
    iris = load_iris(as_frame=True)
    data = iris.frame
    return data

def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    return accuracy

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    accuracy = train_model(df)
    return {"accuracy": accuracy}

@app.post("/data/")
async def create_data(data: Data):
    df = pd.DataFrame(data.data)
    accuracy = train_model(df)
    return {"accuracy": accuracy}

@app.get("/default/")
async def read_default():
    data = load_default_data()
    accuracy = train_model(data)
    return {"accuracy": accuracy, "data_head": data.head().to_dict()}
