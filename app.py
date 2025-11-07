#import os 
from flask import Flask, render_template
from helper import preprocessing, vectorizer, get_prediction

# =================================================================
#  Flask Templates සහ Static Files සඳහා නිවැරදි Paths සකස් කිරීම
#  (Setting up correct Paths for Flask Templates and Static Files)
# =================================================================

# 1. 'app.py' ෆයිල් එක තියෙන තැන සොයා ගැනීම
#BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 2. ප්‍රධාන (Root) ෆෝල්ඩරය සොයා ගැනීම (ෆෝල්ඩර් දෙකක් පිටුපසට යන්න)
# Go up two directories to find the project root (SENTIMENT-ANALYSIS-PROJECT/)
#PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..')) 

# 3. 'templates' සහ 'static' ෆෝල්ඩර් වල නිවැරදි Paths නිර්මාණය
#TEMPLATES_FOLDER = os.path.join(PROJECT_ROOT, 'templates')
#STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')


# 4. Flask App එක හදන විට නිවැරදි Paths ලබා දීම.
# Initialize Flask app with the correct template and static folder locations
#app = Flask(
 #   __name__, 
 #   template_folder=TEMPLATES_FOLDER,
 #   static_folder=STATIC_FOLDER # <--- Image එක load කිරීමට මෙය අත්‍යවශ්‍ය වේ.
#)

# =================================================================
#  Application Routes
# =================================================================
app = Flask(__name__)

data = dict()
reviews = ['Good product' , 'Bad product', 'I like it']
positive = 2  
negative = 1

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html', data = data)

@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
   

if __name__ == "__main__":
    app.run(debug=True)