from flask import Flask, jsonify, render_template
import interface  # Import your detection logic here

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serve your HTML file

@app.route('/start_detection', methods=['GET'])
def start_detection():
    detected_data = interface.predict_and_translate()  # Call your detection function
    if isinstance(detected_data, str):  # Check if it's an error message
        return jsonify({'error': detected_data}), 500  # Return an error if frame capture fails
    
    return jsonify({
        'predictedText': detected_data['predicted_letter'],
        'tamil': detected_data['tamil'],
        'hindi': detected_data['hindi']
    })

if __name__ == '__main__':
    app.run(debug=True)
