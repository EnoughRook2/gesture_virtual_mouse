
import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pgui
import time 

pgui.PAUSE=0
pgui.FAILSAFE=True
screen_w, screen_h= pgui.size()

sensitivity=0.08
deadzone=10
prev_raw_x,prev_raw_y=0,0
smooth_dx, smooth_dy=0,0
smoothing=8

is_first_frame=True
is_victory_held=False
last_right_click_time=0
right_click_cooldown=0.8

is_Open_Palm_held=False
last_Open_Palm_click_time=0
Open_Palm_click_cooldown=0.8


def result_callback(result, output_image, timestamp_ms):
    global prev_raw_x, prev_raw_y,smooth_dx,smooth_dy, is_first_frame
    global is_victory_held, last_right_click_time
    global is_Open_Palm_held, last_Open_Palm_click_time

    if result.hand_landmarks:
        landmarks= result.hand_landmarks[0]
        index_tip= landmarks[0]

        w= output_image.width
        h=output_image.height

        curs_raw_x= index_tip.x*w
        curs_raw_y=index_tip.y*h

        if is_first_frame:
            prev_raw_x,prev_raw_y= curs_raw_x,curs_raw_y
            smooth_dx, smooth_dy=0,0
            is_first_frame=False
            return

        raw_dx= (curs_raw_x- prev_raw_x)*sensitivity
        raw_dy= (curs_raw_y- prev_raw_y)*sensitivity

        smooth_dx += (raw_dx-smooth_dx)/smoothing
        smooth_dy += (raw_dy-smooth_dy)/smoothing
        if abs(smooth_dx)>0.05 or abs(smooth_dy>0.05):
            current_mouse_x, current_mouse_y= pgui.position()

            target_x= int(np.clip(current_mouse_x+smooth_dx,0,screen_w-1))
            target_y= int(np.clip(current_mouse_y+smooth_dy,0,screen_h-1))
            pgui.moveTo(target_x,target_y)


        if result.gestures and result.gestures[0]:
            top_gesture= result.gestures[0][0]
            gesture_name= top_gesture.category_name
            score= top_gesture.score

            if score >0.6:
                now=time.time()
                if gesture_name=="Closed_Fist":
                    if abs(raw_dy/sensitivity)>deadzone:
                        scroll_amount= int(-raw_dy)
                        if scroll_amount !=0:
                            pgui.scroll(scroll_amount)
                    is_victory_held=False
                    is_Open_Palm_held=False
                if gesture_name =="Victory":
                    if (not is_victory_held and (now- last_right_click_time)>right_click_cooldown):
                        pgui.leftClick()
                        is_victory_held=True
                        last_right_click_time=now
                else:
                    is_victory_held=False
                    
                if gesture_name =="Open_Palm":
                    if (not is_Open_Palm_held and (now- last_Open_Palm_click_time)>Open_Palm_click_cooldown):
                        pgui.rightClick()
                        is_Open_Palm_held=True
                        last_Open_Palm_click_time=now
                else:
                    is_Open_Palm_held=False
    else:
        is_first_frame=True
        smooth_dx,smooth_dy=0,0

BaseOptions= mp.tasks.BaseOptions
GestureRecognizer= mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions= mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult= mp.tasks.vision.GestureRecognizerResult
VisionRunningMode= mp.tasks.vision.RunningMode

options=GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=1,
)
cap= cv2.VideoCapture(0)
last_timestamp_ms=0
with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, frame= cap.read()
        if not success:
            break
        frame= cv2.flip(frame,1)

        rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image= mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)

        frame_timestamp_ms= int(time.time()*1000)
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms= last_timestamp_ms+1
        last_timestamp_ms=frame_timestamp_ms

        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        cv2.imshow("MediaPipe Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
