from flask import Flask, render_template, request, jsonify
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
executor = ThreadPoolExecutor()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
@app.route('/')
def index():
    return render_template('test.html')

@app.route('/process_keywords', methods=['POST'])
def process_keywords():
    data = request.json
    keywords = data.get('keywords', [])
    result = call_symptom_predict(keywords)
    return render_template('index.html', result=result["result"], pred2_value=result["pred2_value"])

def call_symptom_predict(keywords):
    
    try:
        script_path = os.path.abspath("sympton_predict.py")
       
        result = subprocess.check_output(["python", script_path] + keywords, text=True)
        
        result_lines = result.strip().split('\n')
        
        processed_result = None
        pred2_value = None
        for line in result_lines:
            if line.startswith("Result: "):
                processed_result = line[len("Result: "):].strip()
            elif line.startswith("Pred2 Value: "):
                pred2_value = line[len("Pred2 Value: "):].strip()
        return {"result": processed_result, "pred2_value": pred2_value}
    except subprocess.CalledProcessError as e:
        return {"error": f"Error: {e}"}


if __name__ == '__main__':
    app.run(debug=True)
