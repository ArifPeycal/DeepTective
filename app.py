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


app = Flask(__name__, template_folder='templates')
app.config.from_object(Config)

db.init_app(app)
migrate = Migrate(app, db)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

result = {0 : 'deepfake', 1 : 'real'}

models = {
    "combined_df_model": load_model('combined_df_model.h5'),
    "static_model": load_model('static.h5'),
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
    file = request.files.get('file_image')
    if file:
        file_name = secure_filename(file.filename)
        file_type = file.content_type.split('/')[0]  
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)

        user_id = session.get('user_id') 
        if not user_id:
            return "User not logged in!"

        new_file = File(
            file_name=file_name,
            file_type=file_type,
            file_path=file_path, 
            user_id=user_id
        )
        db.session.add(new_file)
        db.session.commit()

        session['file_id'] = new_file.id

        return redirect(url_for('choose_model'))
    return "File upload failed!"


@app.route('/choose_model')
def choose_model():
    file_id = session.get('file_id')  
    mlmodels = MLModel.query.all() 

    if not file_id:
        return "No file found! Please upload a file first."

    uploaded_file = File.query.get(file_id)
    if not uploaded_file:
        return "File not found in the database!"

    return render_template('choosemodel.html',mlmodels=mlmodels)


@app.route('/result', methods=['POST'])
def analyze_file():
    file_id = session.get('file_id')
    uploaded_file = File.query.get(file_id)

    if not uploaded_file:
        return "File not found! Please try again."

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

    results = {}
    for model_key in selected_models:
        if model_key in models:
            model = models[model_key]
            label, confidence = predict_deepfake(model, file_details['file_path'])
            results[model_key] = {"label": label, "confidence": confidence}

            analysis_result = AnalysisResult(
                label=label,
                confidence_score=confidence,
                file_id=file_id,
                model_id=MLModel.query.filter_by(model_name=model_key).first().id
            )
            db.session.add(analysis_result)
    db.session.commit()
    return render_template('result.html',file=file_details,results=results)

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