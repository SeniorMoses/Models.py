import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from fastapi import FastAPI


sample_data = {
    "rooms": [4,8,12,16,20,24,28,32],
    "price": [100,200,300,400,500,600,700,800]
}

df = pd.DataFrame(sample_data)

x = df[["rooms"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, "room_bookings.pkl")

print("model learned successfully")


model = joblib.load("room_bookings.pkl")


app = FastAPI()

@app.post("/predict/{number_of_rooms}") 
def predict_price(number_of_rooms: int):
    
    prediction = model.predict([[number_of_rooms]]) 

    return {
        "number_of_rooms": number_of_rooms,
        "predicted_price": prediction[0]
    }