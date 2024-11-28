from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    
    files = db.relationship('File', backref='user', lazy=True)

    def __repr__(self):
        return f"<User {self.username}>"

class File(db.Model):
    __tablename__ = 'files'
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, default=func.now())

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    results = db.relationship('AnalysisResult', backref='file', lazy=True)

class MLModel(db.Model):
    __tablename__ = 'ml_models'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=True)

    results = db.relationship('AnalysisResult', backref='model', lazy=True)

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    id = db.Column(db.Integer, primary_key=True)
    analysis_date = db.Column(db.DateTime, default=func.now())
    label = db.Column(db.String(10), nullable=False)  
    confidence_score = db.Column(db.Float, nullable=False)

    file_id = db.Column(db.Integer, db.ForeignKey('files.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)