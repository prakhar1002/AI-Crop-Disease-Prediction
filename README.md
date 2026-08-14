\# AI-Based Crop Disease Prediction System



\## About the Project



This project is an AI-based crop disease prediction system developed as part of my IBM Generative AI course.



The main idea behind the project is to use a plant leaf image as input and predict whether the plant is affected by a particular disease. After making the prediction, the application also uses Generative AI to provide a simple explanation of the predicted condition, including its symptoms and general management practices.



I used the PlantVillage dataset for training the image classification model. For the classification part, I used MobileNetV2 with transfer learning because it provides a good pretrained image feature extractor without requiring the model to be trained completely from scratch.



The final application is built using Streamlit, which provides a simple web interface where a user can upload a leaf image and see the prediction.



\## What the System Does



The application follows these steps:



1\. The user uploads an image of a crop leaf.

2\. The image is resized to 224 × 224 pixels.

3\. The trained MobileNetV2 model processes the image.

4\. The model predicts one of the 15 supported classes.

5\. The application displays the predicted class and confidence score.

6\. Gemini Generative AI generates an explanation of the predicted disease.



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- MobileNetV2

\- Streamlit

\- NumPy

\- Pillow

\- Google Gemini API

\- PlantVillage Dataset

\- Git and GitHub



\## Dataset



The model was trained using the PlantVillage dataset.



The project currently uses 15 classes:



1\. Tomato Septoria Leaf Spot

2\. Potato Early Blight

3\. Tomato Mosaic Virus

4\. Potato Healthy

5\. Tomato Early Blight

6\. Pepper Bell Healthy

7\. Tomato Target Spot

8\. Tomato Bacterial Spot

9\. Pepper Bell Bacterial Spot

10\. Tomato Leaf Mold

11\. Tomato Late Blight

12\. Tomato Spider Mites

13\. Tomato Healthy

14\. Potato Late Blight

15\. Tomato Yellow Leaf Curl Virus



\## Model



For image classification, I used MobileNetV2 with ImageNet pretrained weights.



The pretrained layers were frozen and a new classification layer was added for the 15 classes used in this project.



The model was trained using TensorFlow and saved in Keras format.



Model file:



`crop\_disease\_model.keras`



During training, the model achieved approximately 92.66% validation accuracy.



\## Generative AI



Generative AI is used after the disease prediction stage.



The predicted disease and model confidence are sent to the Gemini API along with a prompt asking for a simple explanation. The response can include information about what the condition is, common symptoms, and general prevention or management practices.



The Gemini API key is stored as an environment variable instead of being written directly inside the source code.



\## Application



The application is built with Streamlit.



The main interface allows the user to upload a leaf image and then displays:



\- Predicted disease

\- Model confidence

\- GenAI-generated explanation



The basic workflow is:



```text

Leaf Image

&#x20;   |

&#x20;   v

Image Preprocessing

&#x20;   |

&#x20;   v

MobileNetV2 Model

&#x20;   |

&#x20;   v

Disease Prediction

&#x20;   |

&#x20;   +----> Confidence Score

&#x20;   |

&#x20;   v

Gemini Generative AI

&#x20;   |

&#x20;   v

Disease Explanation



Project Files

AI-Crop-Disease-Prediction/

│

├── app.py

├── class\_names.json

├── crop\_disease\_model.keras

├── disease\_info.py

├── requirements.txt

├── README.md

└── .gitignore



**app.py** contains the Streamlit application and prediction logic.



**crop\_disease\_model.keras** is the trained image classification model.



**class\_names.json** contains the class names used by the model.



**disease\_info.py** contains the basic disease information used by the application.



**requirements.txt** contains the Python packages required to run the project.



**How to Run the Project**



First, clone the repository:



git clone https://github.com/prakhar1002/AI-Crop-Disease-Prediction.git



Move into the project folder:



cd AI-Crop-Disease-Prediction



Create a virtual environment:



python -m venv venv



Activate it on Windows:



venv\\Scripts\\activate



Install the required packages:



pip install -r requirements.txt



Set the Gemini API key as an environment variable:



setx GEMINI\_API\_KEY "YOUR\_API\_KEY"



After setting the key, open a new terminal and activate the virtual environment again.



Then start the application:



streamlit run app.py



The Streamlit application will open in the browser.



**Limitations**



This project is mainly intended for educational purposes and demonstration of AI and Generative AI concepts.



The model was trained on the classes available in the selected PlantVillage dataset, so it cannot identify every possible crop or plant disease.



The prediction confidence is also not a guarantee that the diagnosis is correct. For real agricultural decisions, the result should be verified by a qualified agricultural professional.



**Future Improvements**



Some possible improvements for the project are:



Add more crop and disease classes.

Use a larger and more diverse dataset.

Improve the model through fine-tuning.

Add prediction history.

Add support for multiple images.

Improve the user interface.

Provide more detailed disease information through Generative AI.

Deploy the application so it can be accessed online.



**Author**



Prakhar Sharma



GitHub: https://github.com/prakhar1002



