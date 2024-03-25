from flask import Flask, render_template, request
import joblib
import os


app = Flask(__name__)

# Load the pre-trained sentiment classification model
file_name = 'naive_bayes.pkl'
file_path = os.path.join(os.path.dirname(__file__),'model','naive_bayes.pkl')
model = joblib.load(file_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET', 'POST'])

def analyze():
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review']
        
        # Perform sentiment analysis using the loaded model
        sentiment = model.predict([review])[0]
        
        # Map the sentiment to a human-readable label
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        
        return render_template('result.html', review=review, sentiment=sentiment_label)
    else:
        # Render the home page template for GET requests
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)