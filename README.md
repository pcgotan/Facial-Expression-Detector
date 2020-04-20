# Facial-Expression-Recognition
## Demo
![Emotions](https://user-images.githubusercontent.com/45638058/79741542-10f5f200-831f-11ea-8059-8594d8a4b0dc.gif)   
## Description
This project, which constructs and trains Convolutional Neural Networks (CNNs) in Python with the Keras API on FER dataset,  presents the real time facial expression recognition of seven most basic human expressions: ANGER, DISGUST, FEAR, HAPPY, NEUTRAL SAD, SURPRISE.
In the later part of the project I deployed the trained model to a web interface with Flask and applied the model to real-time video streams and image data.   

## Results
I used 4 convolutional layers and 2 fully connected layers and sequential model and achieved the accuracy = 0.6748 and loss = 0.8659 which can be further improved by increasing the number of layers and dimensions of layers and also by changing the models and increasing the training iterations.       

## Workflow
First construct and then train the model on FER dataset. Then save the trained weights and serialize the model architecture as JSON string. Create a Flask app to serve the model's prediction images directly to a web interface.   
Run the `main.py` script to create a Flask app and access it on `0.0.0.0:5000/`     
Run the `SimpleRun.py` script to run the program without Flask app.    

## Refrerence
- [CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn)   
- [Video streaming with Flask](https://blog.miguelgrinberg.com/post/video-streaming-with-flask)  
- [FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
