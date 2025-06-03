# ComfyUI_pose_inter

Interpolates two DWpose or Openpose by a set number of frames.

![image](https://github.com/user-attachments/assets/d6ca6492-63a0-4906-8bc3-19ece844e842)


https://github.com/user-attachments/assets/9e2a47c8-2d87-4792-94b6-b6ae5b83b5df

I recommend using this node in conjunction with openpose-editor.
https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor

# update

2025/06/03 : Added POSE_KEYPOINT to Coord_str node. It is useful when using Openpose with Wan 2.1 ATI. 

![image](https://github.com/user-attachments/assets/5484c587-5cdb-40cc-909b-538b98feec66)


https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_ATI_testing_01.json

Added join_pose node. Creates a list of two Pose_keypoints.

![image](https://github.com/user-attachments/assets/f2aa8ad7-8b52-4c2c-b382-24b2ef86437c)




Added Pick_frame. 
If you enter a number greater than or equal to 0, you will get the pose for that number of frames instead of the entire frame. When using a Pose Sequence, if pose_from and pose_to have the same frame number, you can enter a list of integers with the same number of frames in pick_frame to move the point halfway between the two Pose Sequences.

https://github.com/user-attachments/assets/62f0958d-b262-491e-9777-e2c53b5e1672

![image](https://github.com/user-attachments/assets/1e2153cc-24d8-4736-8b07-a892bb29f929)


![image](https://github.com/user-attachments/assets/c3ae9695-6724-4dba-8992-665bd0a21033)

Interpolation_frames: Number of interpolation frames to generate.

Interpolation: Interpolation method. 

Confidence_threshold: Threshold at which joints appear in Pose_to if they are not present in pose_from. The lower it is, the faster it will appear. 

adjust_body_shape: Adjust the body proportions of Pose_to to that of Pose_from. This doesn't work well if the camera is moving a lot or the poses are too different. 

LandmarkType: DWpose or Openpose

Include_face/hands: Determines whether to draw faces and hands.




