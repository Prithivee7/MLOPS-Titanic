import joblib
from azureml.core.model import Model
import json
import traceback


def init():
    # Initialise the model. This function is ran only once
    global model
    model_path = Model.get_model_path('Titanic')
    model = joblib.load(model_path+"/"+"titanic_model.pkl")
    print('Titanic model loaded...')


def run(raw_data):
    #  This function gets hit when the request is sent. The data is processed and the classification is performed
    try:
        data = json.loads(raw_data)
        passenger_class = data['Pclass']
        sex = data['Sex']
        age = data['Age']
        sibling_info = data['SibSp']
        parch = data['Parch']
        fare = data['Fare']
        embarked = data['Embarked']
        prediction = model.predict(
            [[passenger_class, sex, age, sibling_info, parch, fare, embarked]])[0]

        return {'prediction': prediction}
    except Exception as err:
        traceback.print_exc()


# Sample Request
# req = {"Pclass": 3, "Sex": 1, "Age": 34.5, "SibSp":0,"Parch":0,"Fare":7.8292,"Embarked":2}
