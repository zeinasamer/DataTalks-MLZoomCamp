import pickle

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)



costumer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


churn = pipeline.predict_proba([costumer])[0, 1]
print('prob of churn =',churn)

if churn >= 0.5:
    print('send email with promo')
else:
    print('dont do anything')





