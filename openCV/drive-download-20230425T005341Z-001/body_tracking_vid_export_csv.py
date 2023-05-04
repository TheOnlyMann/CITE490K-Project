import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import datetime
import pandas as pd

# Body keypoint names for 18 keypoints (YOU SHOULD CHAGE IT IF YOU WANT TO USE 32 LKEYPOINTS)
body_keypoints_name = {
            0: "NOSE",
            1: "NECK",
            2: "RIGHT_SHOULDER",
            3: "RIGHT_ELBOW",
            4: "RIGHT_WRIST",
            5: "LEFT_SHOULDER",
            6: "LEFT_ELBOW",
            7: "LEFT_WRIST",
            8: "RIGHT_HIP",
            9: "RIGHT_KNEE",
            10: "RIGHT_ANKLE",
            11: "LEFT_HIP",
            12: "LEFT_KNEE",
            13: "LEFT_ANKLE",
            14: "RIGHT_EYE",
            15: "LEFT_EYE",
            16: "RIGHT_EAR",
            17: "LEFT_EAR"
}

def export_as_csv(obj): # Save per frame

    # Body ID, Date, and Time columns
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S.%f")

    detection_results = {"Body_ID" : obj.id, "Date" : date_string, "Time" : time_string}

    # Keypoint columns
    for i , keypnt in enumerate(obj.keypoint):
        keypnt_str = ','.join(str(x) for x in keypnt)
        detection_results[str(i)+"_"+body_keypoints_name[i]] = keypnt_str

    # Convert the detection results to a DataFrame
    df = pd.DataFrame(detection_results,index=[0])
        
    # Append the DataFrame to the CSV file
    file_name = 'detection_results.csv'
    with open(file_name, 'a', newline='') as f:
        df.to_csv(f, header=f.tell()==0, index=False, mode='a', lineterminator="")               
        


if __name__ == "__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # Read SVO given 
    filepath = "svo_files/ZED2_HD2K_Runners_H264.svo"
    print("Using SVO file: {0}".format(filepath))
    init_params.svo_real_time_mode = True
    init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST 
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()

    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)

            # Update GL view
            viewer.update_view(image, bodies) 

            # Update OCV view
            image_left_ocv = image.get_data()

            for obj in bodies.object_list:
                export_as_csv(obj)

            cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)

    viewer.exit()

    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()