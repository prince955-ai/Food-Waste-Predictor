import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 1. Load Dataset
data = pd.read_csv("food_waste_data.csv")

print("\n========== FOOD WASTE PREDICTOR ==========\n")

print("Dataset Preview:")
print(data.head())


# 2. Encode Categorical Data
meal_encoder = LabelEncoder()
day_encoder = LabelEncoder()
waste_encoder = LabelEncoder()

data["meal_type"] = meal_encoder.fit_transform(data["meal_type"])
data["day_type"] = day_encoder.fit_transform(data["day_type"])
data["waste_level"] = waste_encoder.fit_transform(data["waste_level"])


# 3. Select Features
X = data[
    [
        "people",
        "food_prepared_kg",
        "meal_type",
        "day_type",
        "previous_waste_kg"
    ]
]

y = data["waste_level"]


# 4. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create Machine Learning Model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)


# 6. Train Model
model.fit(X_train, y_train)


# 7. Test Model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


# 8. Take User Input
print("\n========== MAKE A PREDICTION ==========\n")

people = int(input("Number of people: "))

food_prepared = float(
    input("Food prepared (kg): ")
)


# Meal Type
print("\nMeal Type:")
print("1. Lunch")
print("2. Dinner")

meal_choice = int(input("Enter choice: "))

if meal_choice == 1:
    meal = "Lunch"
else:
    meal = "Dinner"


# Day Type
print("\nDay Type:")
print("1. Weekday")
print("2. Weekend")

day_choice = int(input("Enter choice: "))

if day_choice == 1:
    day = "Weekday"
else:
    day = "Weekend"


previous_waste = float(
    input("Previous food waste (kg): ")
)


# 9. Encode User Input
meal_encoded = meal_encoder.transform([meal])[0]
day_encoded = day_encoder.transform([day])[0]


user_data = [[
    people,
    food_prepared,
    meal_encoded,
    day_encoded,
    previous_waste
]]


# 10. Make Prediction
prediction = model.predict(user_data)

result = waste_encoder.inverse_transform(prediction)[0]


# 11. Display Result
print("\n========================================")
print("Predicted Food Waste Level:", result)
print("========================================")


# 12. Give Suggestion
if result == "High":

    print(
        "Suggestion: Reduce food preparation "
        "or improve portion planning."
    )

elif result == "Medium":

    print(
        "Suggestion: Monitor food consumption "
        "and adjust food preparation."
    )

else:

    print(
        "Suggestion: Food preparation appears "
        "to be well managed."
    )