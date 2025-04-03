**Machine Learning Model for Diabetes Prediction**

This project focuses on developing a machine learning model to predict the likelihood of diabetes in patients. Leveraging a dataset containing various health indicators such as Glucose level, Insulin, Age, and BMI, the goal is to create an accurate and reliable predictive tool. This project employs a neural network model trained on this data to classify individuals as either having diabetes or not. The resulting model can be used to identify individuals who may be at risk and could benefit from further medical evaluation.

**video presentation link**

[presentation link]




Make sure you have Python 3.9 installed on your system. You can download it from the official Python website: https://www.python.org/downloads/


Clone the Repository (if applicable):  
If your project files are in a Git repository (like on GitHub), you'll need to clone it to your local machine using the git clone command followed by the repository URL. For example:
 
git clone https://github.com/IkireziI/Diabetes_prediction.git
cd your Diabetes_prediction
 
Navigate to the Project Directory:


If you haven't cloned a repository, make sure you are in the main directory of your project where the requirements.txt file is located. You can use the cd command in your terminal to navigate.
Install Dependencies:


Once you are in the project directory, you can install all the required Python libraries using pip. Open your terminal or command prompt and run the following command:
 
pip install -r requirements.txt
 This command will read the requirements.txt file and install all the packages listed there (like TensorFlow, NumPy, Pandas, scikit-learn, and joblib).  



Usage:
To run the diabetes prediction web application, follow these steps:
Navigate to the main directory of the project in your terminal.


Run the prediction script using the following command:


python src/prediction.py


Wait for the server to start. You should see output in your terminal similar to:

 Running on http://127.0.0.1:5000


Open your web browser (e.g., Chrome, Firefox, Safari).


Go to the address provided in the terminal: In this case, it's likely http://127.0.0.1:5000.


Interact with the web application. You should see a user interface where you can input the required features (like Glucose level, Insulin, Age, BMI) to get a diabetes prediction. Follow the instructions on the webpage to provide the input and get the result.


To stop the web application, go back to your terminal where the script is running and press Ctrl + C.
