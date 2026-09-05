from fastapi import FastAPI 

app = FastAPI()

@app.get("/")
def home():
    return {
        "message":"Hello From Home"
    }

""" 
Normal Query Parameter
/users?name=mohit -- this is called as query param 

@app.get("/users")
def users(name : str):
    return {
        "Name" : name
    }                                                           
"""
    
# Optional query param 
@app.get("/users")
def users(name : str = None):
    return {"Name":name}
# In optional query param if there is no input then null is provided as output 
# it handle the error by replacing null value to it.
