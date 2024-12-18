## Credit Card Fraud Detection Model 💳

Category: Data Analysis, Machine Learning

![model-image](Images/deployment.png)

The goal of this project was to identify fraudulent credit card transactions using a popular dataset in machine learning.  With 30 features of over 200,000 transactions, the dataset was heavily skewed as expected. 
Regular pandas, matplotlib and seaborn was used for EDA and the duplicates were handled according to the results. A major observation was how transaction amounts were dispersed in a heavily right skewed pattern that distorted the true mean representations. Over 20,000 rows were removed in the outlier analysis with a distance of 3 * IQR above and below. (complete analysis steps are detailed in the notebook) 

Another objective of mine was to try building a model with Auto ML libraries and see how these perform instead of using regular scikit-learn models. After doing some research, I have choosen pycaret as a customizable Auto ML library. <br> The binary classifier was to build several models and find the optimal one and Extra Trees classifier had the best overall performance.

Then I used Streamlit to deploy the model and to build an interactive interface where we can insert our values and get predictions as to whether the transaction is a valid one or fraudulent. 

**Tech Stack Used:**

![Python](https://img.shields.io/badge/Python-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-1E2952?style=for-the-badge&logo=data:image/svg+xml;base64,PHRleHQgaGVyZT0nUGlXb3JrJyB2aWV3Qm94PSIwIDAgMzAgMzAiPjxwYXRoIGQ9Ik0xNS4wNy4wMjdDLjAxNy4wMzkuMDAxIDAuMDY1LjAwNSAwLjEzOE0uMTMgMTAuMDA3TDQuNTkgOS45OWw0LjQ1LTMuNTM3TDExLjc4IDguMDdsLTEuNTYgNC43NmwxLjYgNC43NiA1LjE5IDMuNTQgMi41OC0xLjkwOCAyLjU4LTEuOTA4LTguMTMgNi4xOEwxNS41MyAyNnoiIHN0eWxlPSJmaWxsOiNmZmZmZmYiLz48L3RleHQ+)
![Seaborn](https://img.shields.io/badge/Seaborn-3775A9?style=for-the-badge&logo=databricks&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)


![model-image](Images/deployment_valid.png)

---
#### **set up instructions**

The dataset has not been uploaded to avoid exceeding github lfs resource limits. download and extract the dataset inside the Data folder.

download all dependencies and make sure u're using a compatible python version.
I'm using python kernel 3.11 here as the versions above it are not supported by pycaret.

note: it's better to install pycaret first before other modules if we are using the same virtual env. cuz installing it later will replace them with the versions used by pycaret. why? PyCaret has a number of dependencies (like scikit-learn, pandas, matplotlib, etc.), and sometimes these dependencies have specific version requirements. When you install PyCaret after other libraries, it might cause some of your previously installed packages to be downgraded or upgraded to versions that are compatible with PyCaret.


run the streamlit
`streamlit run app.py`

navigate in your browser:
http://localhost:8501/

input your params and a prediction would be made by the model to the screen.


---
Made with ❤️ by [Subhanu](https://github.com/subhanu-dev)

