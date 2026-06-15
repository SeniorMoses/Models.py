import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

sample_data = {
    "rooms":    [4, 8, 12, 16, 20, 24, 28, 32],
    "fan":      [True, False, False, True, True, True, False, False],
    "location": [5,10, 2, 5, 0, 3, 1, 2],
    "age":      [12,24,10, 2, 4,13,14,19],
    "price":    [350, 420, 500, 850, 780, 950, 900, 1050]
} 

df = pd.DataFrame(sample_data)

x = df[["rooms", "fan", "location", "age"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, "room_bookings.pkl")
 
print("model learned successfully")


model = joblib.load("room_bookings.pkl")


app = FastAPI()

class PredictModel(BaseModel):
    rooms : int
    fan : bool 
    location : int
    age : int
    
    @field_validator("rooms")
    @classmethod
    def validate_rooms(cls, value):
        if value <= 0:
            raise ValueError("invalid room info")
        return value
            
    @field_validator("location")
    @classmethod
    def validate_location(cls, value):
        if value < 0 or value > 10:
            raise ValueError("invalid location info") 
        return value
        
    @field_validator("age")
    @classmethod
    def validate_age(cls, value):
        if value < 0:
            raise ValueError("invalid age info")
        return value
        
        
@app.post("/predict") 
def predict_price(data :PredictModel):
    
    prediction = model.predict([[
    data.rooms,
    data.fan,
    data.location,
    data.age 
    ]])   
    if float(prediction) <= 0:
        raise ValueError("inaccurate infomation provided")
    return { 
        "predicted_price": float(prediction[0]) 
    } 
