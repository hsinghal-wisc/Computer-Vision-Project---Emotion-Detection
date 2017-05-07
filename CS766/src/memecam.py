import cv2
import sys
import json
import numpy as np
from keras.models import model_from_json


emotions = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
cascPath = sys.argv[1]

faceCascade = cv2.CascadeClassifier(cascPath)
noseCascade = cv2.CascadeClassifier(cascPath)


json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model.h5')


def overlay(probs):
    if max(probs) > 0.8:
        emotion = emotions[np.argmax(probs)]
        return 'meme_faces/{}-{}.png'.format(emotion, emotion)
    else:
        index1, index2 = np.argsort(probs)[::-1][:2]
        emotion1 = emotions[index1]
        emotion2 = emotions[index2]
        return 'meme_faces/{}-{}.png'.format(emotion1, emotion2)

def predict_meme(face_image_gray):
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

video_capture = cv2.VideoCapture(0)
while True:
    
    ret, frame = video_capture.read()
    both = np.hstack((frame,frame))
    l = both.shape[1]/2
    without_meme = both[:,l+1:,:]

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)


    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )


    for (x, y, w, h) in faces:

        face_image_gray = img_gray[y:y+h, x:x+w]
        filename = overlay(predict_meme(face_image_gray))

        print filename
        meme = cv2.imread(filename,-1)
        
        try:
            meme.shape[2]
        except:
            meme = meme.reshape(meme.shape[0], meme.shape[1], 1)
       
        orig_mask = meme[:,:,3]
       
        ret1, orig_mask = cv2.threshold(orig_mask, 10, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(orig_mask)
        meme = meme[:,:,0:3]
        origMustacheHeight, origMustacheWidth = meme.shape[:2]

        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx,ny,nw,nh) in nose:
           

            mustacheWidth =  20 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

            x1 = nx - (mustacheWidth/4)
            x2 = nx + nw + (mustacheWidth/4)
            y1 = ny + nh - (mustacheHeight/2)
            y2 = ny + nh + (mustacheHeight/2)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h


            mustacheWidth = (x2 - x1)
            mustacheHeight = (y2 - y1)

            mustache = cv2.resize(meme, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)

            roi = roi_color[y1:y2, x1:x2]

            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)

            dst = cv2.add(roi_bg,roi_fg)

            roi_color[y1:y2, x1:x2] = dst

            break

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            angry, fear, happy, sad, surprise, neutral = predict_meme(face_image_gray)


    combined_frame = np.hstack((frame, without_meme))
    cv2.imshow('Video', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
