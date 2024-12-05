# DeepTective: Deepfake Detection Tool
DeepTective is an innovative deepfake detection tool designed to empower users to identify and mitigate the risks associated with synthetic media manipulation. Leveraging MesoNet deep learning algorithms, DeepTective offers a comprehensive solution for detecting and analyzing deepfake content.

## Features:
Deepfake Detection: Upload suspicious media files (images or videos) for analysis and receive a confidence score indicating the likelihood of it being a deepfake.

## How to Use:
1. Upload Media: Simply upload the media file you wish to analyze using the provided interface.
2. View Results: Receive instant feedback on the likelihood of the uploaded media being a deepfake, accompanied by a confidence score.
   
![alt](DeepTective%20Tuto.gif)

## Getting Started with DeepTective

To start using **DeepTective**, follow these steps:

1. **Clone the Repository**  
   Clone the repository to your local machine using Git:
   ```bash
   git clone https://github.com/ArifPeycal/DeepTective.git   
   ```

2. **Install Dependencies**  
   Navigate to the project directory and install the necessary Python dependencies:
   ```bash
   cd DeepTective
   pip3 install -r requirements.txt
   ```

3. **Create a Database**  
   Open **phpMyAdmin** and create a new database called `deeptective`. This will store the application’s data.

4. **Initialize the Database**  
   After creating the database, run the following Flask commands to set up the necessary tables:

   - Initialize the migration environment:
     ```bash
     flask db init
     ```

   - Edit the `alembic.ini` file to ensure that the database connection string points to the correct MySQL database (i.e., `deeptective`):
     ```ini
     sqlalchemy.url = mysql://username:password@localhost/deeptective
     ```
     Replace `username` and `password` with your MySQL credentials.

   - Generate the migration script:
     ```bash
     flask db migrate -m "Create tables"
     ```

   - Apply the migration to the database:
     ```bash
     flask db upgrade
     ```

5. **Run Seeder Script**  
   After setting up the database, you can run the `seeder.py` script to populate the database with initial data (e.g., ML models):
   ```bash
   python seeder.py
   ```

6. **Run the Application**  
   Finally, start the application by running:
   ```bash
   python app.py
   ```

Your application should now be running, and you can start interacting with **DeepTective**.

---

<!---
 Contributors:
Arif Peycal (github.com/ArifPeycal)
License:
This project is licensed under the MIT License. See the LICENSE file for details.

Feedback and Support:
 For feedback or support inquiries, please contact deeptective.support@example.com.
 -->
