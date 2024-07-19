from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.graphics.texture import Texture

import cv2
import os
import numpy as np
import tensorflow as tf
from layers import L1Dist

class CamApp(App):
    def build(self):

        # Main Layout Components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify",on_press=self.verify, size_hint=(1,.1))
        self.verification = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # Load tensorflow/keras model 
        model_path = 'C:\\Users\\lkmah\\FaceID\\siamesemodel.h5'
        # Check if the model file exists
        if os.path.exists(model_path):
            # Load the model with custom objects
            self.model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist})

        # Setup Video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/22.0)

        return layout

    def update(self,*args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr') 
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    # Load Image from file and convert into 100x100px
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img,(100,100))
        img = img/255.0
        return img
    
    # Verification Function
    def verify(self,*args):
        detection_threshold = 0.99
        verification_threshold = 0.8
        SAVE_PATH = os.path.join('application_data','input_image','input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH, frame)


        results=[]
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data','verification_images', image))
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
        verified = verification > verification_threshold

        self.verification.text = "Verified" if verification == True else "Unverifed"

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.6))
        Logger.info(np.sum(np.array(results)>0.2))

        return results, verified
    

if __name__ == '__main__':
    CamApp().run()