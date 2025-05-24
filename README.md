# ComfyUI_pose_inter

Interpolates two DWpose or Openpose by a set number of frames.

![image](https://github.com/user-attachments/assets/d6ca6492-63a0-4906-8bc3-19ece844e842)


https://github.com/user-attachments/assets/9e2a47c8-2d87-4792-94b6-b6ae5b83b5df


![image](https://github.com/user-attachments/assets/c3ae9695-6724-4dba-8992-665bd0a21033)

Interpolation_frames: Number of interpolation frames to generate.

Interpolation: Interpolation method. 

Confidence_threshold: Threshold at which joints appear in Pose_to if they are not present in pose_from. The lower it is, the faster it will appear. 

adjust_body_shape: Adjust the body proportions of Pose_to to that of Pose_from. This doesn't work well if the camera is moving a lot or the poses are too different. 

LandmarkType: DWpose or Openpose

Include_face/hands: Determines whether to draw faces and hands.




