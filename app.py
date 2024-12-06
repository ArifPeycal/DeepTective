# Standard Library Imports
import os

# Flask Imports
from flask import Flask, render_template, request, redirect, flash, session, url_for

# Configuration and Models
from config import Config
from models import db, User, File, MLModel, AnalysisResult

# Flask Extensions
from flask_migrate import Migrate

# Werkzeug Utilities
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# Machine Learning and TensorFlow Imports
from keras.models import load_model
from keras.preprocessing import image
from efficientnet.tfkeras import EfficientNetB0
import numpy as np

from PIL import Image
from io import BytesIO
import requests
import cv2
from mtcnn import MTCNN                                                      

app = Flask(__name__, template_folder='templates')
app.config.from_object(Config)

db.init_app(app)
migrate = Migrate(app, db)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

result = {0 : 'deepfake', 1 : 'real'}

models = {
    "custom_cnn_model": load_model('customcnn-batch64-epoch19.h5'),
    "mesonet_deepfake_model": load_model('mesonet-batch64-epoch22.h5'),
    "mesonet_gan_model": load_model('gan-mesonet-batch64-1e-6.h5'),
}

def predict_deepfake(model, file_path):
    i = image.load_img(file_path, target_size=(256, 256))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 256, 256, 3)
    preds = model.predict(i)
    x = (preds > 0.5).astype("int32") 
    y = float(preds[0])  
    if y < 0.5:
        y = 1 - y
    confidence = "{:.02%}".format(y)  
    return result[int(x)], confidence


@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('homepage.html', username=session['username'])
    return render_template('homepage.html') 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return redirect('/login')

        # Query the user from the database
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password): 
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect('/')  
        else:
            flash('Invalid credentials, please try again.', 'error')
            return redirect('/login')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return redirect('/register')

        hashed_password = generate_password_hash(password)

        existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing_user:
            flash('Username or email already exists.', 'error')
            return redirect('/register')

        try:
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect('/login')  
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'error')
            return redirect('/register')

    return render_template('register.html')

@app.route('/upload', methods=['GET'])
def drag():
    if 'user_id' in session:
        return render_template('drag.html', username=session['username'])
    return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload_file():
    def detect_faces_and_crop(image_path, output_size=(256, 256)):
        """
        Detect faces in an image using Haar Cascade, filter out faces smaller than 100x100,
        and return cropped and resized faces.
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        cropped_faces = []
        for (x, y, w, h) in faces:
            # Filter out faces smaller than 100x100
            if w < 100 or h < 100:
                continue

            # Crop face
            face = img[y:y+h, x:x+w]

            # Resize to fixed size
            face_resized = cv2.resize(face, output_size, interpolation=cv2.INTER_AREA)
            cropped_faces.append(face_resized)

        return cropped_faces



    def save_faces_to_disk(faces, original_file_name, base_path):
        """Save cropped and resized faces with original filename."""
        face_paths = []
        base_name, ext = os.path.splitext(original_file_name)
        
        for idx, face in enumerate(faces):
            face_filename = f"{base_name}_cv2_face_{idx+1}{ext}"
            face_path = os.path.join(base_path, face_filename)
            cv2.imwrite(face_path, face)
            face_paths.append(face_path)
        
        return face_paths

    # Handle image link
    image_link = request.form.get('image_link')
    if image_link:
        try:
            response = requests.get(image_link, stream=True)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                file_name = secure_filename(os.path.basename(image_link))
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                with open(file_path, 'wb') as image_file:
                    for chunk in response.iter_content(1024):
                        image_file.write(chunk)

                # Detect and crop faces
                cropped_faces = detect_faces_and_crop(file_path, output_size=(256, 256))
                if not cropped_faces:
                    return "No faces detected in the image.", 400

                face_paths = save_faces_to_disk(cropped_faces, file_name, app.config['UPLOAD_FOLDER'])

            # Save face metadata to the database
            user_id = session.get('user_id')
            if not user_id:
                return "User not logged in!"

            file_ids = []  # List to store file_ids of cropped faces
            for face_path in face_paths:
                new_file = File(
                    file_name=os.path.basename(face_path),
                    file_type='image',
                    file_path=face_path,
                    user_id=user_id
                )
                db.session.add(new_file)
                db.session.commit()

                file_ids.append(new_file.id)  # Add file_id to the list

            session['file_ids'] = file_ids  # Store list of file_ids in session
            return redirect(url_for('choose_model'))

        except Exception as e:
            return f"Failed to process the image link: {str(e)}", 500

    # Handle file upload
    file = request.files.get('file_image')
    if file:
        file_name = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)

        # Detect and crop faces
        cropped_faces = detect_faces_and_crop(file_path, output_size=(256, 256))
        if not cropped_faces:
            return "No faces detected in the uploaded file.", 400

        # Save cropped faces with original filename
        face_paths = save_faces_to_disk(cropped_faces, file_name, app.config['UPLOAD_FOLDER'])

        # Save face metadata to the database
        user_id = session.get('user_id')
        if not user_id:
            return "User not logged in!"

        file_ids = [] 
        for face_path in face_paths:
            new_file = File(
                file_name=os.path.basename(face_path),
                file_type='image',
                file_path=face_path,
                user_id=user_id
            )
            db.session.add(new_file)
            db.session.commit()

            file_ids.append(new_file.id)  # Add file_id to the list

        session['file_ids'] = file_ids  # Store list of file_ids in session
        return redirect(url_for('choose_model'))

    return "File upload failed!"

@app.route('/choose_model')
def choose_model():
    file_ids = session.get('file_ids')  # Get list of file_ids from session
    mlmodels = MLModel.query.all() 

    if not file_ids:
        return "No files found! Please upload a file first."

    # Get uploaded files by their IDs
    uploaded_files = File.query.filter(File.id.in_(file_ids)).all()
    if not uploaded_files:
        return "Files not found in the database!"

    return render_template('choosemodel.html', mlmodels=mlmodels, uploaded_files=uploaded_files)


@app.route('/result', methods=['POST'])
def analyze_file():
    file_ids = session.get('file_ids')  # Retrieve the list of file IDs from session

    if not file_ids:
        return "No files found! Please upload files first."

    uploaded_files = File.query.filter(File.id.in_(file_ids)).all()

    if not uploaded_files:
        return "Files not found! Please try again."

    results = {}
    for uploaded_file in uploaded_files:
        file_details = {
            "file_id": uploaded_file.id,
            "file_name": uploaded_file.file_name,
            "file_path": uploaded_file.file_path,
            "file_type": uploaded_file.file_type,
            "file_url": url_for('static', filename=f'uploads/{uploaded_file.file_name}'),
            "upload_date": uploaded_file.upload_date
        }

        selected_models = request.form.getlist('detectorSelection[]')
        if not selected_models:
            return "No models selected! Please try again."

        file_results = {}
        for model_key in selected_models:
            if model_key in models:
                model = models[model_key]
                label, confidence = predict_deepfake(model, file_details['file_path'])
                file_results[model_key] = {"label": label, "confidence": confidence}

                analysis_result = AnalysisResult(
                    label=label,
                    confidence_score=confidence,
                    file_id=uploaded_file.id,
                    model_id=MLModel.query.filter_by(model_name=model_key).first().id
                )
                db.session.add(analysis_result)

        db.session.commit()
        results[uploaded_file.id] = {
            'file_details': file_details,
            'file_results': file_results
        }

    return render_template('result.html', results=results)

@app.route('/profile')
def profile():
    if 'user_id' in session:
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        if not user:
            return "User not found!"

        # Query files uploaded by the user
        user_files = File.query.filter_by(user_id=user_id).all()

        # Structure the submission data
        submission_history = []
        for file in user_files:
            submission_history.append({
                "date": file.upload_date.strftime("%Y-%m-%d %H:%M:%S"),
                "file_url": url_for('static', filename=f'uploads/{file.file_name}'),
                "result_link": f"/history/{file.id}",
                "status": "success"  
            })

        context = {
            "email": user.email,
            "username": user.username,
            "total_tasks": len(user_files),
            "submission_history": submission_history,
            "user_id": user_id  
        }

        return render_template('profile.html', **context)
    return redirect(url_for('home'))



@app.route('/history/<int:file_id>')
def file_history(file_id):
    # Ensure the logged-in user is authenticated
    user_id = session.get('user_id')
    if not user_id:
        return "User not logged in!"

    # Query the file to ensure it belongs to the logged-in user
    file = File.query.filter_by(id=file_id, user_id=user_id).first()
    if not file:
        return "File not found or unauthorized access!"

    # Fetch all analysis results for this file
    analysis_results = (
        AnalysisResult.query
        .filter_by(file_id=file_id)
        .join(MLModel, AnalysisResult.model_id == MLModel.id)
        .add_columns(MLModel.display_name, AnalysisResult.label, AnalysisResult.confidence_score)
        .all()
    )

    submission_history = []
    for result in analysis_results:
        model_display_name, label, confidence_score = result[1], result[2], result[3]
        submission_history.append({
            "detector": model_display_name,
            "result": label,
            "confidence": f"{confidence_score:.1f}%",
            "status": "Completed"
        })

    context = {
        "file": {
            "file_name": file.file_name,
            "file_url": url_for('static', filename=f'uploads/{file.file_name}')
        },
        "submission_history": submission_history
    }

    return render_template('history.html', **context)

@app.route('/update-profile', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        flash("You must be logged in to access this page.", "error")
        return redirect(url_for('login'))

    logged_in_user_id = session.get('user_id')
    user = User.query.get(logged_in_user_id)

    if not user:
        flash("User not found!", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            email = request.form.get('email')
            username = request.form.get('username')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm-password')

            print(f"Received: Email={email}, Username={username}, Password={password}, Confirm Password={confirm_password}")

            if password and password != confirm_password:
                flash("Passwords do not match!", "error")
                return render_template('update.html', user=user)

            user.email = email
            user.username = username
            if password:
                user.password = generate_password_hash(password)

            db.session.commit()
            print("Profile updated successfully!")
            return redirect(url_for('profile'))

        except Exception as e:
            print(f"Error occurred: {e}")  
            flash("An error occurred while updating your profile. Please try again.", "error")
            return render_template('update.html', user=user)

    return render_template('update.html', user=user)

@app.route('/contact')
def contact():
        return render_template('contact.html')

@app.route('/logout')
def logout():
    session.clear()  
    flash('You have been logged out successfully.', 'success')
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)