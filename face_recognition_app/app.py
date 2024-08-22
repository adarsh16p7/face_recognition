from flask import Flask, render_template, request, redirect, url_for
from deepface import DeepFace
from collections import Counter
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to compare two images using DeepFace
def compare_faces(image1_path, image2_path, model_name='ArcFace'):
    try:
        result = DeepFace.verify(image1_path, image2_path, model_name=model_name, enforce_detection=False)
        return result['verified'], result['distance']
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return None, None

# Function to check for low-resolution images
def low_resolution_warning(image1_path, image2_path):
    try:
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        threshold = 100
        if min(image1.size) < threshold or min(image2.size) < threshold:
            return True
        return False
    except Exception as e:
        print(f"Error checking image resolution: {e}")
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image uploads
        image1 = request.files['image1']
        image2 = request.files['image2']

        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2.filename)

        image1.save(image1_path)
        image2.save(image2_path)

        # Compare images using different models
        models = ['ArcFace', 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
        results = []

        for model in models:
            is_same_person, distance = compare_faces(image1_path, image2_path, model_name=model)
            if is_same_person is not None:
                results.append(is_same_person)

        # Majority result
        majority_result = Counter(results).most_common(1)[0][0] if results else None

        # Check for low-resolution warning
        low_res_warning = low_resolution_warning(image1_path, image2_path)

        return render_template('result.html', majority_result=majority_result, low_res_warning=low_res_warning, image1=image1.filename, image2=image2.filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
