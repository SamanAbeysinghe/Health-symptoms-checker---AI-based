from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Scraping.html')

@app.route('/search')
def search():
    disease = request.args.get('disease')
    
    # Web scraping logic (replace this with your specific implementation)
    details = get_disease_details(disease)

    return render_template('Result.html', disease=disease, details=details)

def get_disease_details(disease):
    # Example web scraping logic using BeautifulSoup
    try:
        url = f"https://en.wikipedia.org/wiki/{disease}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content div that usually contains the details
        content_div = soup.find('div', {'id': 'mw-content-text'})

        if content_div:
            # Extract text from paragraphs inside the content div
            paragraphs = content_div.find_all('p')
            details = '\n'.join(paragraph.get_text() for paragraph in paragraphs)
        else:
            details = "Details not found on the page."

        return details
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (e.g., page not found)
        print(f"HTTP error: {e}")
        return f"Error fetching details for {disease}: {e}"
    except Exception as e:
        # Handle other exceptions
        print(f"Error: {e}")
        return "Error fetching details"

if __name__ == '__main__':
    app.run(debug=True)