import face_recognition
import cv2
import os
import numpy as np

#defining paths to image directories
path_known_dir = "known_faces"
path_unknown_dir = "unknown_faces"

#creating empty list for encoded faces to be populated
known_face_lst = []
known_name = 'Big Ed'

#changing directory to known image directory
os.chdir(path_known_dir)
#looping through directory to process images
for filename in os.listdir():
    if filename.endswith('.jpg'):
        image = face_recognition.load_image_file(filename)
        img_encode = face_recognition.face_encodings(image)
        known_face_lst.append(img_encode)

    else:
        pass



#changing directory to unknown images
os.chdir('../' + path_unknown_dir)
#loop through all images in directory
for unknownimg in os.listdir():
    if unknownimg.endswith('.jpg'):
        #loading image
        image = face_recognition.load_image_file(unknownimg)
        #finding faces
        face_locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=0, model='cnn')
        #encoding each face in respective locations
        img_encoded = face_recognition.face_encodings(image, face_locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #iterate over encoded images and respective locations
        for face_encoding, face_loc in zip(img_encoded, face_locations):
            #comparing the encoded faces found in the unknown image directory to the known encoded images 
            results = face_recognition.compare_faces(
                known_face_lst, face_encoding, tolerance=0.2)
            match = None
            #if any image is a match then draw a rectangle around the face and the name of the match under 
            if np.all(results):
                match = known_name
                top_left = (face_loc[3], face_loc[0])
                bottom_right = (face_loc[1], face_loc[2])
                cv2.rectangle(image, top_left, bottom_right, [255, 0, 0], 2)

                top_left = (face_loc[3], face_loc[2])
                bottom_right = (face_loc[1], face_loc[2])
                cv2.rectangle(image, top_left, bottom_right,
                              [255, 0, 0], cv2.FILLED)
                cv2.putText(image, match, (face_loc[3] + 10, face_loc[2] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        #displaying images 
        cv2.imshow("matching images", image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)
