import joblib

"""
This function takes in a DataFrame with only one row of data containing the background information of a customer.

:param data: DataFrame with only one row of data containing the background information of a customer. Format: {'Year': string, 'Major': string, 'University': string, 'Time': int}

Returns a prediction of the customer's order choice (string)
"""


def predict(data):
    model = joblib.load("gradient_boost_model.pkl")
    # We need to restore the one-hot encoding
    training_cols = [
        "Time",
        "Major_Accounting",
        "Major_Anthropology",
        "Major_Astronomy",
        "Major_Biology",
        "Major_Business Administration",
        "Major_Chemistry",
        "Major_Civil Engineering",
        "Major_Economics",
        "Major_Finance",
        "Major_Fine Arts",
        "Major_International Business",
        "Major_Marketing",
        "Major_Mathematics",
        "Major_Mechanical Engineering",
        "Major_Music",
        "Major_Philosophy",
        "Major_Physics",
        "Major_Political Science",
        "Major_Psychology",
        "Major_Sociology",
        "University_Ball State University",
        "University_Butler University",
        "University_DePauw University",
        "University_Indiana State University",
        "University_Indiana University Bloomington",
        "University_Indiana University-Purdue University Indianapolis (IUPUI)",
        "University_Purdue University",
        "University_University of Evansville",
        "University_University of Notre Dame",
        "University_Valparaiso University",
        "Year_Year 1",
        "Year_Year 2",
        "Year_Year 3",
        "Year_Year 4",
    ]

    data = pd.get_dummies(data, columns=["Major", "University", "Year"])
    new_data = pd.DataFrame(data, columns=training_cols)
    # data will be having less columns than training_cols because it won't have all the majors, universities, and years
    # So those columns will be designated as NaN
    # we will fill them with False here
    new_data = new_data.fillna(False)
    return model.predict(new_data)[0]


import pandas as pd

data = pd.DataFrame(
    {
        "Year": ["Year 2"],
        "Major": ["Chemistry"],
        "University": ["Indiana State University"],
        "Time": [11],
    }
)

print("Prediction: ", predict(data))
