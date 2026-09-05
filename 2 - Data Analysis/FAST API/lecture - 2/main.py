from fastapi import FastAPI

app = FastAPI()

# Home Route
@app.get("/")
def home():
    return {"Home":"Welcome to FASTAPI"}

@app.get("/about")
def about():
    return {"message":"This is about page"}

@app.get("/users")
def users():
    return {
        "message":["mohit","rohit","raj"]
    }
    