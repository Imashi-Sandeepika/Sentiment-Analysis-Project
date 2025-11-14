import os
from flask import Flask, render_template, request, redirect, url_for        
from helper import preprocessing, vectorizer, get_prediction, fallback_prediction
from logger import logging


app = Flask(__name__)

logging.info('Flask server started')
reviews = []
positive = 0 
negative = 0


def _build_page_data(**overrides):
    view_model = {
        'reviews': reviews,
        'positive': positive,
        'negative': negative,
        'error': None,
        'last_input': ''
    }
    view_model.update({k: v for k, v in overrides.items() if v is not None})
    return view_model


@app.route("/")
def index():
    logging.info('======== Open Home Page ========')
    return render_template('index.html', data=_build_page_data())

@app.route("/", methods = ['post'])
def my_post():
    text = request.form.get('text', '').strip()
    logging.info(f'Text : {text}')

    if not text:
        logging.warning('Empty review submitted.')
        return render_template('index.html', data=_build_page_data(
            error='Please enter a review before submitting.',
            last_input=''
        )), 400

    preprocessed_txt = preprocessing(text)
    logging.info(f'Preprocessed Text : {preprocessed_txt}')

    vectorized_txt = vectorizer(preprocessed_txt)
    logging.info(f'Vectorized Text : {vectorized_txt}')

    fallback_label, fallback_score = fallback_prediction(text)
    if vectorized_txt.any():
        prediction = get_prediction(vectorized_txt)
        logging.info('Prediction generated via logistic regression model.')
        if prediction == 'positive' and fallback_label == 'negative' and fallback_score <= -0.30:
            logging.info('Overriding model prediction with fallback (compound score %.3f).', fallback_score)
            prediction = fallback_label
    else:
        logging.info('Vectorizer could not map any known tokens, using fallback.')
        prediction = fallback_label

    logging.info(f'Prediction : {prediction}')

    if prediction == 'negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1
   
    reviews.insert(0, {'text': text, 'sentiment': prediction})
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)