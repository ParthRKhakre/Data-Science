from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {'message':'Hello World'}

@app.GET('/about')
def about():
    return {'message':'Campusx'}