# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model("converted_keras (1)")
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    resizeImg=cv2.resize(frame,(224,224))
    numpyFormat=np.array(resizeImg,dtype=np.float32)
    expanded=np.expand_dims(numpyFormat,axis=0)
    normalisedImg = expanded/255.0

    #predicting result 
    prediction = model.predict(normalisedImg)
    print(prediction)

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()