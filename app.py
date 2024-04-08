from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained models
models = {}
categories = ['hate_speech', 'offensive_language', 'neither']
for category in categories:
    with open(f'{category}_mark6.pkl', 'rb') as f:
        models[category] = pickle.load(f)

# Load the TfidfVectorizer
with open('cMark6.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the request
    text = request.form.get('text')

    # Vectorize the text input
    text_vectorized = vectorizer.transform([text])

    # Predict the probability of each category for the input text
    predictions = {}
    for category, model in models.items():
        predictions[category] = model.predict_proba(text_vectorized)[0][1]

    result={}


    if (predictions['offensive_language']>predictions['hate_speech'] and predictions['offensive_language']>predictions['neither']):
        result['expression']='offensive'
    elif(predictions['hate_speech']>predictions['neither']):
        result['expression'] = 'hate'
    else:
        result['expression'] = 'positive'

    # Return the predicted probabilities
    return jsonify(result,predictions)

if __name__ == '__main__':
    app.run(debug=True)
