import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer
import re
import joblib
from flask import Flask, render_template, url_for, request
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)


def convert_lower(text):
    return text.lower()

def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text])

def change_text(text):
    text = remove_tags(text)
    text = convert_lower(text)
    text = lemmatize_word(text)
    return text

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pickle', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html", title="home page")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    user = str([(x) for x in request.form.values()])
    my_prediction = change_text(user)
    my_prediction = pd.Series(my_prediction)
    new = cv.transform(my_prediction)
    prediction = model.predict(new)
    output = prediction[0]  # getting first index
    if output == 0:
        return render_template('home.html', pred="The category of this news article is Bussiness")
    elif output == 1:
        return render_template("home.html", pred="The category of this news article is tech")
    elif output == 2:
        return render_template("home.html", pred="The category of this news article is politics")
    elif output == 3:
        return render_template("home.html", pred="The category of this news article is Sports")
    else:
        return render_template("home.html", pred="The category of this news article is Entertainment")


if __name__ == "__main__":
    app.run(debug=True)
