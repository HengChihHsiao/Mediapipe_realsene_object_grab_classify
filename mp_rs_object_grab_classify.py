# ====== Sample Code for Smart Design Technology Blog ======

# Intel Realsense D435 cam has RGB camera with 1920×1080 resolution
# Depth camera is 1280x720
# FOV is limited to 69deg x 42deg (H x V) - the RGB camera FOV

# If you run this on a non-Intel CPU, explore other options for rs.align
    # On the NVIDIA Jetson AGX we build the pyrealsense lib with CUDA

import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import math

from pprint import pprint

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 20)
fontScale = 0.6
color = (255, 0, 0) # bgr
thickness = 1

# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print('detected_camera: {}'.format(detected_camera))
    connected_devices.append(detected_camera)
device = connected_devices[0] # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey

# ====== Mediapipe ======
mpHands = mp.solutions.hands
mp_objectron = mp.solutions.objectron    # mediapipe objectron
# hands = mpHands.Hands()
hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5)

objectron = mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.2,
                            min_tracking_confidence=0.5,
                            model_name='Cup')

mpDraw = mp.solutions.drawing_utils


def get_every_finger_knuckle_distance(
        mp_hand_results: mp.solutions.hands.Hands, 
        image_w: int, 
        image_h: int, 
        depth_image_flipped: np.ndarray
        )-> list:
    
    if mp_hand_results.multi_hand_landmarks:
        for hand_landmarks in mp_hand_results.multi_hand_landmarks:
            knuckle_coordinates = []                   # 記錄手指節點座標的串列
            for knuckle in hand_landmarks.landmark:
                # print(knuckle)
                # 將 21 個節點換算成座標，記錄到 finger_points
                # x = knuckle.x * image_w
                # y = knuckle.y * image_h

                x = int(knuckle.x * image_w)
                y = int(knuckle.y * image_h)

                if x >= len(depth_image_flipped[0]):
                    x = len(depth_image_flipped[0]) - 1
                if y >= len(depth_image_flipped):
                    y = len(depth_image_flipped) - 1
                    
                mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
                mfk_distance_feet = mfk_distance * 3.281 # feet
                knuckle_coordinates.append((x, y, mfk_distance))

            # pprint(finger_points)
    return knuckle_coordinates

def get_obj_every_landmarks_distance(
        mp_obj_results: mp.solutions.objectron.Objectron,
        image_w: int,
        image_h: int,
        depth_image_flipped: np.ndarray
        )-> list:
    if mp_obj_results.detected_objects:
        for detected_object in mp_obj_results.detected_objects:
            # print('detected_object:', detected_object)
            # print('detected_object.landmarks:', detected_object.landmarks_2d.landmark)
            # print('detected_object.landmarks[0].x:', detected_object.landmarks_3d.landmark[0].x)
            # print('detected_object.landmarks[0].y:', detected_object.landmarks_3d.landmark[0].y)
            # print('detected_object.landmarks[0].z:', detected_object.landmarks_3d.landmark[0].z)

            obj_landmarks_coordinates = []
            for landmark in detected_object.landmarks_2d.landmark:

                # x = landmark.x * image_w
                # y = landmark.y * image_h

                x = int(landmark.x * image_w)
                y = int(landmark.y * image_h)

                if x >= len(depth_image_flipped[0]):
                    x = len(depth_image_flipped[0]) - 1
                if y >= len(depth_image_flipped):
                    y = len(depth_image_flipped) - 1

                obj_landmarks_distance = depth_image_flipped[y,x] * depth_scale # meters
                obj_landmarks_distance_feet = obj_landmarks_distance * 3.281 # feet
                obj_landmarks_coordinates.append((x, y, obj_landmarks_distance))

    return obj_landmarks_coordinates
            

def judge_hand_grab(
        knuckle_coordinates: list,
        obj_landmarks_coordinates: list,
        depth_scale: float,
        finger_threshold: int = 10,
        distance_threshold_cm: float = 20
        ) -> bool:
    # print('knuckle_coordinates:', knuckle_coordinates)
    # print('obj_landmarks_coordinates:', obj_landmarks_coordinates)
    # print('len(knuckle_coordinates):', len(knuckle_coordinates))
    # print('len(obj_landmarks_coordinates):', len(obj_landmarks_coordinates))

    if len(knuckle_coordinates) == 0 or len(obj_landmarks_coordinates) == 0:
        return False
    else:
        # center_obj_landmarks_coordinates = obj_landmarks_coordinates[0]
        knuckle_obj_match_count = 0
        for knuckle_coordinate in knuckle_coordinates:
            match_flag = False
            for obj_landmarks_coordinate in obj_landmarks_coordinates:
                delta_x = knuckle_coordinate[0] - obj_landmarks_coordinate[0]
                delta_y = knuckle_coordinate[1] - obj_landmarks_coordinate[1]

                delta_hypotenuse = math.sqrt(delta_x**2 + delta_y**2)
                delta_hypotenuse_cm = delta_hypotenuse * depth_scale * 100

                delta_distance_cm = knuckle_coordinate[2] - obj_landmarks_coordinate[2] * 100

                if delta_hypotenuse_cm < distance_threshold_cm and delta_distance_cm < distance_threshold_cm:
                    match_flag = True
                    break
            if match_flag:
                knuckle_obj_match_count += 1
        
        if knuckle_obj_match_count >= finger_threshold:
            return True
            
def judge_hand_grab_by_center(
        knuckle_coordinates: list,
        obj_landmarks_coordinates: list,
        depth_scale: float,
        finger_threshold: int = 10,
        distance_threshold_cm: float = 20
        ) -> bool:
    if len(knuckle_coordinates) == 0 or len(obj_landmarks_coordinates) == 0:
        return False
    else:
        center_obj_landmarks_coordinates = obj_landmarks_coordinates[0]
        knuckle_obj_match_count = 0
        for knuckle_coordinate in knuckle_coordinates:
            delta_x = knuckle_coordinate[0] - center_obj_landmarks_coordinates[0]
            delta_y = knuckle_coordinate[1] - center_obj_landmarks_coordinates[1]

            delta_hypotenuse = math.sqrt(delta_x**2 + delta_y**2)
            delta_hypotenuse_cm = delta_hypotenuse * depth_scale * 100

            delta_distance_cm = knuckle_coordinate[2] - center_obj_landmarks_coordinates[2] * 100

            if delta_hypotenuse_cm < distance_threshold_cm and delta_distance_cm < distance_threshold_cm:
                knuckle_obj_match_count += 1

        if knuckle_obj_match_count >= finger_threshold:
            return True
    


if __name__ == "__main__":
    # ====== Enable Streams ======
    config.enable_device(device)

    # # For worse FPS, but better resolution:
    # stream_res_x = 1280
    # stream_res_y = 720
    # # For better FPS. but worse resolution:
    stream_res_x = 640
    stream_res_y = 480

    object_distance_threshold = 5 # cm
    finger_grab_threshold = 12

    stream_fps = 30

    config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)  # depth
    config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps) # color
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # ====== Get depth Scale ======
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

    # ====== Set clipping distance ======
    clipping_distance_in_meters = 2
    clipping_distance = clipping_distance_in_meters / depth_scale
    print(f"\tConfiguration Successful")

    # ====== Get and process images ====== 
    print(f"Starting to capture images")

    while True:
        start_time = dt.datetime.today().timestamp()

        # Get and align frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue

        # Process images
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image, 1)
        color_image = np.asanyarray(color_frame.get_data())

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
        background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = cv2.flip(background_removed,1)
        color_image = cv2.flip(color_image,1)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        image_w = len(depth_image_flipped[0])
        image_h = len(depth_image_flipped)

        # Process hands
        mp_hand_results = hands.process(color_images_rgb)
        mp_obj_results = objectron.process(color_images_rgb)    # get obj detection results


        if mp_hand_results.multi_hand_landmarks:
            number_of_hands = len(mp_hand_results.multi_hand_landmarks)
            i = 0
            for hand_landmarks in mp_hand_results.multi_hand_landmarks:
                mpDraw.draw_landmarks(color_images_rgb, hand_landmarks, mpHands.HAND_CONNECTIONS)
                org2 = (20, org[1] + (20*(i+1)))

                # print(results.multi_hand_landmarks[i])

                hand_side_classification_list = mp_hand_results.multi_handedness[i]
                hand_side = hand_side_classification_list.classification[0].label
                # middle_finger_knuckle = mp_hand_results.multi_hand_landmarks[i].landmark[9]


                knuckle_coordinates = get_every_finger_knuckle_distance(mp_hand_results, image_w, image_h, depth_image_flipped)

                x, y, mfk_distance = knuckle_coordinates[9]

                # x = int(middle_finger_knuckle.x * image_w)
                # y = int(middle_finger_knuckle.y * image_h)

                # if x >= len(depth_image_flipped[0]):
                #     x = len(depth_image_flipped[0]) - 1
                # if y >= len(depth_image_flipped):
                #     y = len(depth_image_flipped) - 1
                    
                # mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
                # mfk_distance_feet = mfk_distance * 3.281 # feet

                cv2.putText(color_images_rgb, '{} Hand, Hand Distance: {:0.3} m, (x: {}, y: {})'.format(hand_side, mfk_distance, x, y), org2, font, fontScale, color, thickness, cv2.LINE_AA)
                
                
                i+=1
            cv2.putText(color_images_rgb, 'Hands: {}'.format(number_of_hands), org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(color_images_rgb,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)

        if mp_obj_results.detected_objects:
            for detected_object in mp_obj_results.detected_objects:
                mpDraw.draw_landmarks(
                  color_images_rgb, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mpDraw.draw_axis(color_images_rgb, detected_object.rotation,
                                    detected_object.translation)
                
                # print(mp_obj_results.detected_objects[0].landmarks_3d)
                obj_landmarks_coordinates = get_obj_every_landmarks_distance(mp_obj_results, image_w, image_h, depth_image_flipped)

                org3 = (20, org[1] + 60)

                x, y, obj_distance = obj_landmarks_coordinates[0]

                cv2.putText(color_images_rgb, 'Object Distance: {:0.3} m, (x: {}, y: {})'.format(obj_distance, x, y), org3, font, fontScale, color, thickness, cv2.LINE_AA)
        
        if mp_hand_results.multi_hand_landmarks and mp_obj_results.detected_objects:
            org5 = (20, org[1] + 100)
            # hand_obj_match_flag = judge_hand_grab(knuckle_coordinates, obj_landmarks_coordinates, depth_scale, finger_threshold=8, distance_threshold_cm=5)
            hand_obj_match_flag = judge_hand_grab_by_center(knuckle_coordinates, 
                                                            obj_landmarks_coordinates, 
                                                            depth_scale, 
                                                            finger_threshold=finger_grab_threshold, 
                                                            distance_threshold_cm=object_distance_threshold)

            if hand_obj_match_flag:
                cv2.putText(color_images_rgb, 'Grab', org5, font, fontScale, color, thickness, cv2.LINE_AA)

        # Display FPS
        time_diff = dt.datetime.today().timestamp() - start_time
        fps = int(1 / time_diff)
        org4 = (20, org[1] + 80)
        cv2.putText(color_images_rgb, f"FPS: {fps}", org4, font, fontScale, color, thickness, cv2.LINE_AA)

        windows_name = 'Hand_coor_w_mp_rs'

        # Display images 
        # cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(windows_name, 1280, 720)
        cv2.namedWindow(windows_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(windows_name, cv2.cvtColor(color_images_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the window
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {device}")
            break

    print(f"Application Closing")
    pipeline.stop()
    print(f"Application Closed.")