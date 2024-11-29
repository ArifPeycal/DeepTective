from app import app, db
from models import MLModel

def seed_ml_models():
    model1 = MLModel(
        model_name='combined_df_model',
        display_name='Combined Deepfake Model',
        accuracy=95.0,
        description='Detects deepfakes using a combination of multiple algorithms.'
    )
    
    model2 = MLModel(
        model_name='static_model',
        display_name='Static Model',
        accuracy=92.0,
        description='Specialized in detecting static image-based deepfakes.'
    )

    # Add to the session
    db.session.add(model1)
    db.session.add(model2)

    # Commit the session
    db.session.commit()
    print("ML Models seeded successfully!")

# Run the seeding function inside an app context
if __name__ == '__main__':
    with app.app_context():
        seed_ml_models()
