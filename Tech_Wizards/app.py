from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import tensorflow as tf
import os
import pandas as pd
import uuid
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix, graycoprops
from feature_extract import preprocess_and_extract_features
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for


#app.config['UPLOAD_FOLDER'] = 'temp_uploads'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\rohan_075b4dd\Desktop\Sitam\temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # Add more if needed

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load required assets
try:
    # Get feature order from original dataset
    dataset_columns = pd.read_csv("Dataset_ohe.csv").columns.tolist()
    feature_columns = dataset_columns[:-1]  # Exclude label column

    # Load models and scaler (ensure these files exist)
    scaler = joblib.load('scaler.pkl')  # Critical dependency!
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')

    # Load TFLite CNN model
    interpreter = tf.lite.Interpreter(model_path='cnn_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

except Exception as e:
    raise RuntimeError(f"Initialization failed: {str(e)}")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/handle_upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_page'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('display_file', filename=filename))
    
    return 'Invalid file type!'

@app.route('/display/<filename>')
def display_file(filename):
    return render_template('display.html', filename=filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Save uploaded file with unique name
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"File saved to: {filepath}")  # Debug print
        
        # Verify file exists
        if not os.path.exists(filepath):
            return jsonify({"error": "File not saved properly"}), 500

        # Process image
        features_dict = preprocess_and_extract_features(filepath)
        # print(f"Extracted features: {features_dict}")

        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame(
            [{col: features_dict.get(col, 0) for col in feature_columns}]
        )
        print(f"Features DataFrame shape: {features_df.shape}")


        # Verify scaler is fitted
        if not hasattr(scaler, 'n_features_in_'):
            raise ValueError("Scaler is not properly fitted")

        # Scale features (critical for model compatibility)
        scaled_features = scaler.transform(features_df)
        print("Features scaled successfully")

        # Get predictions from all models
        # CNN prediction
        cnn_input = scaled_features.reshape(1, -1, 1).astype(np.float32)
        interpreter.set_tensor(input_details['index'], cnn_input)
        interpreter.invoke()
        cnn_output = interpreter.get_tensor(output_details['index'])
        cnn_pred = np.argmax(cnn_output)

        # SVM prediction
        svm_pred = svm_model.predict(scaled_features)[0]

        # Random Forest prediction
        rf_pred = rf_model.predict(scaled_features)[0]

        # Hybrid prediction (simple average)
        final_prediction = int(np.round((cnn_pred + svm_pred + rf_pred) / 3))

        match final_prediction:
            case 0:
                return render_template('condition0.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition0.html', uploaded_image=f"/{filepath}")

            case 1:
                return render_template('condition1.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition1.html', uploaded_image=f"/{filepath}")


            case 2:
                return render_template('condition2.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition2.html', uploaded_image=f"/{filepath}")
            case 3:
                return render_template('condition3.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition3.html', uploaded_image=f"/{filepath}")
            case 4:
                return render_template('condition4.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition4.html', uploaded_image=f"/{filepath}")
            case 5:
                return render_template('condition5.html', uploaded_image=url_for('uploaded_file', filename=filename))

                # return render_template('condition5.html', uploaded_image=f"/{filepath}")
            case 6:
                return render_template('condition6.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition6.html', uploaded_image=f"/{filepath}")
            case 7:
                return render_template('condition7.html', uploaded_image=url_for('uploaded_file', filename=filename))
                # return render_template('condition7.html', uploaded_image=f"/{filepath}")


            case _:
                return jsonify({
                    "predictions": {
                        "cnn": int(cnn_pred),
                        "svm": int(svm_pred),
                        "rf": int(rf_pred),
                        "hybrid": final_prediction
                    }
                })

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        pass  # Keeping the cleanup commented out for debugging
        # if os.path.exists(filepath):
        #     os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
