import face_recognition
import cv2
from PIL import Image, ImageDraw
import numpy as np



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hat_ori = cv2.imread('cap1.png',-1)
 
cap = cv2.VideoCapture(0) #webcame video
cap.set(3,640)
cap.set(4,480)
cap.set(15, 0.1)


face_locations = []
face_encodings = []
process_this_frame = True


   
def eye_size(eye):  
	eyeWidth = dist.euclidean(eye[0], eye[3])  
	hull = ConvexHull(eye)  
	eyeCenter = np.mean(eye[hull.vertices, :], axis=0)  
	eyeCenter = eyeCenter.astype(int)  
   
	return int(eyeWidth), eyeCenter  

def place_eye(frame, eyeCenter, eyeSize):  
	eyeSize = int(eyeSize * 1.5)  
	x1 = int(eyeCenter[0,0] - (eyeSize/2))  
	x2 = int(eyeCenter[0,0] + (eyeSize/2))  
	y1 = int(eyeCenter[0,1] - (eyeSize/2))  
	y2 = int(eyeCenter[0,1] + (eyeSize/2))  
	h, w = frame.shape[:2]

	# check for clipping  
	if x1 < 0:  
		x1 = 0  
	if y1 < 0:  
		y1 = 0  
	if x2 > w:  
		x2 = w  
	if y2 > h:  
		y2 = h  
   
   # re-calculate the size to avoid clipping  
	eyeOverlayWidth = x2 - x1  
	eyeOverlayHeight = y2 - y1  
   
   # calculate the masks for the overlay  
	eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   
   # take ROI for the verlay from background, equal to size of the overlay image  
	roi = frame[y1:y2, x1:x2]  
   
   # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.  
	roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
   # roi_fg contains the image pixels of the overlay only where the overlay should be  
	roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)  
   
   # join the roi_bg and roi_fg  
	dst = cv2.add(roi_bg,roi_fg)  
   
   # place the joined image, saved to dst back over the original image  
	frame[y1:y2, x1:x2] = dst  

	


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

while True:
        ret_val, img = cap.read()
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]



        rgb_frame = img[:,:,::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))
        for (x, y, w, h) in faces:
                if h > 0 and w > 0:
 
 
                        hat_symin = int(y - 5 * h / 12)
                        hat_symax = int(y + 3 * h / 12)
                        sh_hat = hat_symax - hat_symin

                        face_hat_roi_color = img[hat_symin:hat_symax, x:x+w]
 
                       
                        hat = cv2.resize(hat_ori, (w, sh_hat),interpolation=cv2.INTER_CUBIC)
                        transparentOverlay(face_hat_roi_color,hat,(0,0),0.6)

                


        
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            

        # Display the results
        for (top, right, bottom, left) in (face_locations):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

           

            # Find all facial features in all the faces in the image
            face_landmarks_list = face_recognition.face_landmarks(img)
           
            for face_landmarks in face_landmarks_list:
                pil_image = Image.fromarray(img)
                
                d = ImageDraw.Draw(pil_image, 'RGBA')

                
                d.polygon(face_landmarks['left_eyebrow'],fill=(0,0,0))
                d.polygon(face_landmarks['right_eyebrow'],fill=(0,0,0))
                d.line(face_landmarks['chin'],fill=(0,0,0),width=25)

                img = np.asarray(pil_image)

 
        img = cv2.flip(img, 1)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release handle to the webcam
cap.release()
cv2.destroyAllWindows()


