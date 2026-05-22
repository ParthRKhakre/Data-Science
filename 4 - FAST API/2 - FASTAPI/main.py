from fastapi import FastAPI 
import json

app = FastAPI()

def load_data():  
    with open('patients.json','r') as f:
# Above line of code loads the data from a json file in read only mode  
        data = json.load(f)
# json.load(f)convert JSON file data into a python dictionary/list. 
    return data    

@app.get("/")
def hello():
    return {"message":"Patient Management API"}

@app.get("/about")
def about():
    return {"message" : "A fully functional API to manage your patient records"}

@app.get("/view")
def view():
    data = load_data()
    return data
    