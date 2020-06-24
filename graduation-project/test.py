import cv2, dlib
import numpy as np
import time
import winsound as ws
from imutils import face_utils
from keras.models import load_model

IMG_SIZE = (34, 26)
Stack_l = 0
Stack_r = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('models/2018_12_17_22_58_35.h5')
model.summary()

#눈만 이용하기 위해
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

#경고음 출력
def beepsound():
  freq = 2000
  dur = 1000
  ws.Beep(freq, dur)

# main
cap = cv2.VideoCapture(0) # 0으로 지정시 캠으로 실시간 촬영

while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    #cv2.imshow('l', eye_img_l)
    #cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

    # visualize
    state_l = ' %.1f'
    state_r = ' %.1f'

    # 눈을 감을 시 스택을 쌓아 경고를 주는
    if pred_l < 0.1:
        Stack_l += 1
        time.sleep(0.05)
    else:
        Stack_l = 0
    if pred_r < 0.1:
        Stack_r += 1
        time.sleep(0.05)
    else:
        Stack_r = 0
    # 경고로 소리가 나오게
    # 화면이랑 실제랑 반대임. 실제에선 상관 없음
    if Stack_r > 40:
        #print("Warning! left ") # 테스트용. 실제에선 출력 할 일이 없음
        print(beepsound())
        Stack_r = 0
    if Stack_l > 40:
        #print("Warning! right")
        print(beepsound())
        Stack_l = 0

    state_l = (state_l % pred_l)
    state_r = (state_r % pred_r)
    #네모 박스를 쳐서 편의성을
    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

  cv2.imshow('result', img)

  if cv2.waitKey(1) == ord('q'):
    break
