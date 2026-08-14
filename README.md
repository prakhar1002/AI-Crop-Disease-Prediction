AI-Based Crop Disease Prediction System



About the Project



This project is an AI-based crop disease prediction system developed as part of my IBM Generative AI course.



The main purpose of this project is to predict crop diseases from leaf images using a deep learning model. The user can upload an image of a plant leaf, and the system predicts the most likely disease along with the confidence of the prediction.



After the prediction, Generative AI is used to provide a simple explanation of the predicted condition, including information about symptoms and general management practices.



For the image classification part, I used the PlantVillage dataset and MobileNetV2 with transfer learning. The final application was developed using Streamlit.



How the System Works



The application follows these steps:



1\. The user uploads an image of a crop leaf.

2\. The image is resized to 224 × 224 pixels.

3\. The trained MobileNetV2 model processes the image.

4\. The model predicts one of the supported crop/disease classes.

5\. The prediction and confidence score are displayed.

6\. The predicted disease is passed to Gemini Generative AI.

7\. Gemini generates a simple explanation of the predicted condition.



The overall workflow is:



&#x20; text

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



Technologies Used



\* Python

\* TensorFlow

\* Keras

\* MobileNetV2

\* Streamlit

\* NumPy

\* Pillow

\* Google Gemini API

\* PlantVillage Dataset

\* Git

\* GitHub



Dataset



The project uses the PlantVillage dataset for training the image classification model.



The current model supports 15 classes:



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



Model



MobileNetV2 was used as the base model for image classification.



The model uses transfer learning with pretrained ImageNet weights. The pretrained layers were frozen and a new classification layer was added for the 15 classes used in this project.



The trained model is saved as:



crop\_disease\_model.keras



During training, the model achieved approximately 92.66% validation accuracy.



Generative AI



Generative AI is used after the image classification stage.



Once the disease has been predicted, the disease name and confidence score are provided to the Gemini API. Gemini then generates a simple explanation that can include:



\* What the condition means

\* Common symptoms

\* General prevention and management practices

\* A reminder that AI predictions should be verified by an agricultural professional



The Gemini API key is stored as an environment variable and is not included directly in the source code.



Application



The application is built using Streamlit.



The user can upload a leaf image through the web interface and receive the prediction and GenAI explanation.



The application displays:



\* Uploaded leaf image

\* Predicted disease

\* Confidence score

\* Generative AI explanation



Project Structure



AI-Crop-Disease-Prediction/

│

├── app.py

├── class\_names.json

├── crop\_disease\_model.keras

├── disease\_info.py

├── requirements.txt

├── README.md

├── .gitignore

│

└── venv/





File Description



app.py

Contains the Streamlit application, image processing, disease prediction and GenAI integration.



class\_names.json

Contains the names of the 15 classes used by the model.



crop\_disease\_model.keras

The trained MobileNetV2-based crop disease classification model.



disease\_info.py

Contains disease information used by the application.



requirements.txt

Contains the Python packages required to run the project.



.gitignore

Contains files and folders that should not be uploaded to GitHub, such as the virtual environment and secret files.



How to Run the Project



1\. Clone the repository



git clone https://github.com/prakhar1002/AI-Crop-Disease-Prediction.git



2\. Open the project folder



cd AI-Crop-Disease-Prediction



3\. Create a virtual environment



On Windows:



python -m venv venv



4\. Activate the virtual environment



venv\\Scripts\\activate



5\. Install the required packages



pip install -r requirements.txt



6\. Configure the Gemini API



Create a Gemini API key and store it as an environment variable.



On Windows:



setx GEMINI\_API\_KEY "YOUR\_API\_KEY"



After setting the environment variable, open a new terminal.



Do not put the actual API key directly inside `app.py` or upload it to GitHub.



7\. Run the application



streamlit run app.py



The application will open in the browser at the local Streamlit address.



Model Performance



The model achieved approximately:



Validation Accuracy: 92.66%



The actual prediction confidence can vary depending on the image provided to the application.



For example, during testing, the model correctly predicted a Tomato Early Blight image with a confidence of approximately 99%.



Limitations



This project is primarily developed for educational purposes and demonstration of AI and Generative AI concepts.



The model is limited to the 15 classes included in the selected dataset. It cannot identify every possible crop or plant disease.



The confidence score is also not a guarantee that the prediction is correct. Real agricultural decisions should be made with appropriate professional guidance.



Future Improvements



Some possible improvements for this project include:



\* Adding more crop and disease classes

\* Training with a larger and more diverse dataset

\* Fine-tuning the MobileNetV2 model

\* Improving the user interface

\* Adding prediction history

\* Allowing multiple images to be analyzed

\* Adding more detailed GenAI-based information

\* Deploying the application online

\* Adding support for additional crops



Author



Prakhar Sharma



GitHub:

\[https://github.com/prakhar1002](https://github.com/prakhar1002)



Project Repository



https://github.com/prakhar1002/AI-Crop-Disease-Prediction

