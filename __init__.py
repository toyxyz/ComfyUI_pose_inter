import torch
# import comfy.utils # Assuming this is part of a larger system, commented out if not strictly needed for this snippet
import cv2
import numpy as np
# import folder_paths # Assuming this is part of a larger system, commented out if not strictly needed for this snippet
# import os # Not used in this snippet

# OpenPose standard colors from open_pose/util.py (BGR format)
# Body colors (18 colors for 18 keypoints and 17 limbs)
# These are used for both keypoints and limbs, indexed accordingly.
body_colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

# Face keypoints color (white, no connections)
face_color = [255, 255, 255]  # White for face landmarks

# Hand keypoint color (Red)
hand_keypoint_color = [0, 0, 255] # Red (BGR) for hand keypoints

# Hand limb colors (20 distinct colors for 20 connections)
# Approximates the gradient effect from util.py without matplotlib
hand_limb_colors = [
    [255,0,0],[255,60,0],[255,120,0],[255,180,0],   # Reds/Oranges
    [180,255,0],[120,255,0],[60,255,0],[0,255,0],     # Yellows/Greens
    [0,255,60],[0,255,120],[0,255,180],[0,180,255],   # Greens/Cyans
    [0,120,255],[0,60,255],[0,0,255],[60,0,255],     # Blues
    [120,0,255],[180,0,255],[255,0,180],[255,0,120]  # Purples/Magentas
]


# Body skeleton connections (derived from open_pose/util.py limbSeq, 0-indexed)
# 17 connections, matching the first 17 colors in body_colors for limbs
body_skeleton = [
    [1, 2],   # Neck to RShoulder
    [1, 5],   # Neck to LShoulder
    [2, 3],   # RShoulder to RElbow
    [3, 4],   # RElbow to RWrist
    [5, 6],   # LShoulder to LElbow
    [6, 7],   # LElbow to LWrist
    [1, 8],   # Neck to RHip
    [8, 9],   # RHip to RKnee
    [9, 10],  # RKnee to RAnkle
    [1, 11],  # Neck to LHip
    [11, 12], # LHip to LKnee
    [12, 13], # LKnee to LAnkle
    [1, 0],   # Neck to Nose
    [0, 14],  # Nose to REye
    [14, 16], # REye to REar
    [0, 15],  # Nose to LEye
    [15, 17]  # LEye to LEar
]

# Face keypoints - no skeleton connections
face_skeleton = []

# Hand skeleton connections (20 connections)
hand_skeleton = [
    # Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    # Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],
    # Middle finger
    [0, 9], [9, 10], [10, 11], [11, 12],
    # Ring finger
    [0, 13], [13, 14], [14, 15], [15, 16],
    # Pinky
    [0, 17], [17, 18], [18, 19], [19, 20]
]

def draw_keypoints_and_skeleton(image, keypoints_data, skeleton_connections, colors_config, point_radius=3, line_thickness=2, is_face=False, is_hand=False):
    """
    키포인트와 스켈레톤을 이미지에 그리는 함수
    colors_config:
        - For body: list of colors (e.g., body_colors)
        - For face: single color list (e.g., face_color)
        - For hand: dict {'limbs': list_of_limb_colors, 'points': single_point_color}
    """
    if not keypoints_data:
        return

    # Group keypoints into (x, y, confidence) tuples
    tri_tuples = [keypoints_data[i:i + 3] for i in range(0, len(keypoints_data), 3)]

    # Draw skeleton connections
    if skeleton_connections: # Face skeleton is empty, so no lines will be drawn for face
        for i, (joint_idx_a, joint_idx_b) in enumerate(skeleton_connections):
            if joint_idx_a >= len(tri_tuples) or joint_idx_b >= len(tri_tuples):
                continue

            a_x, a_y, a_confidence = tri_tuples[joint_idx_a]
            b_x, b_y, b_confidence = tri_tuples[joint_idx_b]

            if a_confidence > 0 and b_confidence > 0:
                limb_color = None
                if is_hand:
                    limb_color_list = colors_config['limbs']
                    limb_color = limb_color_list[i % len(limb_color_list)]
                elif is_face: # Should technically not be called with non-empty skeleton for face
                    limb_color = colors_config # Expected to be a single color like face_color
                else: # Body
                    limb_color_list = colors_config # Expected to be body_colors
                    limb_color = limb_color_list[i % len(limb_color_list)]

                if limb_color is not None:
                    cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), limb_color, line_thickness)

    # Draw keypoints
    for i, (x, y, confidence) in enumerate(tri_tuples):
        if confidence > 0:
            point_color = None
            current_radius = point_radius
            if is_hand:
                point_color = colors_config['points']
            elif is_face:
                point_color = colors_config # Expected to be face_color
                current_radius = 2 # Face keypoints are smaller
            else: # Body
                point_color_list = colors_config # Expected to be body_colors
                point_color = point_color_list[i % len(point_color_list)]

            if point_color is not None:
                cv2.circle(image, (int(x), int(y)), current_radius, point_color, -1)

def gen_skeleton_with_face_hands(pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d,
                                 canvas_width, canvas_height, landmarkType):
    """
    얼굴과 손을 포함한 전체 스켈레톤 이미지 생성
    """
    # Create canvas
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # DWPose might use normalized coordinates differently, this scaling might need adjustment
    # For OpenPose, coordinates are typically already in image space or normalized 0-1.
    # The original code had a special case for DWPose setting canvas to 1x1,
    # which seems like it would make scaling problematic if keypoints are not already normalized.
    # Assuming keypoints are normalized 0-1 if landmarkType is DWPose, otherwise pixel coords.
    # For this example, we'll assume keypoints are consistently scaled or normalized before this function.

    def scale_keypoints(keypoints, target_w, target_h, input_is_normalized):
        if not keypoints:
            return []
        scaled = []
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            if input_is_normalized: # If input is 0-1, scale to canvas
                scaled.extend([x * target_w, y * target_h, conf])
            else: # If input is already in some pixel space, ensure it's for the current canvas
                  # This part might need more context on how DWPose vs OpenPose keypoints are provided
                scaled.extend([x, y, conf]) # Assuming they are already scaled for this canvas_width/height
        return scaled

    # Determine if input keypoints are normalized (heuristic, might need explicit flag)
    # For OpenPose from the util.py, they are pixel coordinates before normalization for drawing.
    # If this function receives normalized data (0-1), then scaling is needed.
    # The original code scaled them unconditionally.
    input_normalized = (landmarkType == "OpenPose") # Assumption based on original code's scaling logic

    scaled_pose = scale_keypoints(pose_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_face = scale_keypoints(face_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_left = scale_keypoints(hand_left_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_right = scale_keypoints(hand_right_keypoints_2d, canvas_width, canvas_height, input_normalized)

    # Draw body skeleton
    draw_keypoints_and_skeleton(image, scaled_pose, body_skeleton, body_colors, point_radius=6, line_thickness=4)

    # Draw face keypoints (no skeleton)
    if scaled_face:
        draw_keypoints_and_skeleton(image, scaled_face, face_skeleton, face_color, point_radius=2, line_thickness=1, is_face=True)

    # Hand colors configuration
    hand_colors_config = {'limbs': hand_limb_colors, 'points': hand_keypoint_color}

    # Draw hand skeletons
    if scaled_hand_left:
        draw_keypoints_and_skeleton(image, scaled_hand_left, hand_skeleton, hand_colors_config, point_radius=3, line_thickness=2, is_hand=True)

    if scaled_hand_right:
        draw_keypoints_and_skeleton(image, scaled_hand_right, hand_skeleton, hand_colors_config, point_radius=3, line_thickness=2, is_hand=True)

    return image

def transform_all_keypoints(keypoints_1, keypoints_2, frames):
    """
    모든 키포인트(body, face, hands)를 보간하는 함수
    """
    def interpolate_keypoint_set(kp1, kp2, num_frames):
        if not kp1 or not kp2:
            # If one set is empty, and we want to interpolate,
            # we might want to return the non-empty set for all frames, or empty.
            # For now, returning empty if either is missing for simplicity of interpolation.
            # Or, if one is empty, treat its keypoints as (0,0,0) for interpolation.
            # The original code returns [[] for _ in range(num_frames)] if len(kp1) != len(kp2)
            # or if one is empty. Let's refine this.

            if not kp1 and not kp2: return [[] for _ in range(num_frames)]
            if not kp1: kp1 = [0.0] * len(kp2) # Treat missing as all zeros
            if not kp2: kp2 = [0.0] * len(kp1)


        if len(kp1) != len(kp2):
            # This is a critical issue. For robust interpolation, they must match.
            # Fallback: return the first pose for all frames or empty.
            # For now, let's log an error or return empty sequences.
            print(f"Warning: Keypoint sets have different lengths ({len(kp1)} vs {len(kp2)}). Interpolation might be incorrect.")
            # To prevent crashes, we can try to use the shorter length, but this is not ideal.
            # A better approach would be to ensure data integrity before this step.
            # Fallback to returning empty for now if lengths mismatch significantly.
            # If lengths are similar, one might be padded. Assuming they should be identical.
            return [[] for _ in range(num_frames)]


        tri_tuples_1 = [kp1[i:i + 3] for i in range(0, len(kp1), 3)]
        tri_tuples_2 = [kp2[i:i + 3] for i in range(0, len(kp2), 3)]

        if not tri_tuples_1: # if kp1 was empty and became list of 0.0, tri_tuples_1 could be empty
             return [[] for _ in range(num_frames)]


        keypoints_sequence = []

        for j in range(num_frames):
            interpolated_kps_for_frame = []
            interp_factor = j / float(num_frames -1) if num_frames > 1 else 0.0 # Ensure factor is 0 for first, 1 for last

            for i in range(len(tri_tuples_1)):
                x1, y1, c1 = tri_tuples_1[i]
                x2, y2, c2 = tri_tuples_2[i]

                new_x, new_y, new_c = 0.0, 0.0, 0.0

                # Handle cases where one of the keypoints is not detected (confidence is 0)
                if c1 > 0 and c2 > 0: # Both detected, interpolate
                    new_x = x1 + (x2 - x1) * interp_factor
                    new_y = y1 + (y2 - y1) * interp_factor
                    new_c = c1 + (c2 - c1) * interp_factor # Interpolate confidence as well
                elif c1 > 0 and c2 == 0: # Only in first pose
                    new_x, new_y, new_c = x1, y1, c1 * (1.0 - interp_factor) # Fade out
                elif c1 == 0 and c2 > 0: # Only in second pose
                    new_x, new_y, new_c = x2, y2, c2 * interp_factor # Fade in
                else: # Not detected in either
                    new_x, new_y, new_c = 0.0, 0.0, 0.0

                interpolated_kps_for_frame.extend([new_x, new_y, new_c])
            keypoints_sequence.append(interpolated_kps_for_frame)
        return keypoints_sequence

    # Extract keypoints from both poses
    pose_1 = keypoints_1.get('pose_keypoints_2d', [])
    face_1 = keypoints_1.get('face_keypoints_2d', [])
    hand_left_1 = keypoints_1.get('hand_left_keypoints_2d', [])
    hand_right_1 = keypoints_1.get('hand_right_keypoints_2d', [])

    pose_2 = keypoints_2.get('pose_keypoints_2d', [])
    face_2 = keypoints_2.get('face_keypoints_2d', [])
    hand_left_2 = keypoints_2.get('hand_left_keypoints_2d', [])
    hand_right_2 = keypoints_2.get('hand_right_keypoints_2d', [])

    # Interpolate each keypoint set
    pose_sequence = interpolate_keypoint_set(pose_1, pose_2, frames)
    face_sequence = interpolate_keypoint_set(face_1, face_2, frames)
    hand_left_sequence = interpolate_keypoint_set(hand_left_1, hand_left_2, frames)
    hand_right_sequence = interpolate_keypoint_set(hand_right_1, hand_right_2, frames)

    # Combine all sequences
    combined_sequence = []
    for i in range(frames):
        combined_frame = {
            'pose_keypoints_2d': pose_sequence[i] if i < len(pose_sequence) else [],
            'face_keypoints_2d': face_sequence[i] if i < len(face_sequence) else [],
            'hand_left_keypoints_2d': hand_left_sequence[i] if i < len(hand_left_sequence) else [],
            'hand_right_keypoints_2d': hand_right_sequence[i] if i < len(hand_right_sequence) else []
        }
        combined_sequence.append(combined_frame)

    return combined_sequence

class Pose_Inter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_from": ("POSE_KEYPOINT", ),
                "pose_to": ("POSE_KEYPOINT", ),
                "interpolate_frames": ("INT", {"default": 10, "min": 2, "max": 100, "step": 1}),
                "landmarkType": (["OpenPose", "DWPose"], ), # DWPose might have different coord systems
                "include_face": ("BOOLEAN", {"default": True}),
                "include_hands": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Pose Interpolation"

    def run(self, pose_from, pose_to, interpolate_frames, landmarkType, include_face, include_hands):
        # Extract pose data (assuming pose_from and pose_to are lists of dicts)
        if not pose_from or not pose_to:
            raise ValueError("Input poses cannot be empty.")

        openpose_dict_from = pose_from[0] # Takes the first detected person/pose
        openpose_dict_to = pose_to[0]     # Takes the first detected person/pose

        if "people" not in openpose_dict_from or not openpose_dict_from["people"]:
            raise ValueError("pose_from does not contain 'people' data.")
        if "people" not in openpose_dict_to or not openpose_dict_to["people"]:
            raise ValueError("pose_to does not contain 'people' data.")

        person_from = openpose_dict_from["people"][0]
        person_to = openpose_dict_to["people"][0]

        # Prepare keypoint data for interpolation
        keypoints_from = {
            'pose_keypoints_2d': person_from.get("pose_keypoints_2d", []),
            'face_keypoints_2d': person_from.get("face_keypoints_2d", []) if include_face else [],
            'hand_left_keypoints_2d': person_from.get("hand_left_keypoints_2d", []) if include_hands else [],
            'hand_right_keypoints_2d': person_from.get("hand_right_keypoints_2d", []) if include_hands else []
        }

        keypoints_to = {
            'pose_keypoints_2d': person_to.get("pose_keypoints_2d", []),
            'face_keypoints_2d': person_to.get("face_keypoints_2d", []) if include_face else [],
            'hand_left_keypoints_2d': person_to.get("hand_left_keypoints_2d", []) if include_hands else [],
            'hand_right_keypoints_2d': person_to.get("hand_right_keypoints_2d", []) if include_hands else []
        }

        # Generate interpolated sequence
        interpolated_sequence = transform_all_keypoints(
            keypoints_from,
            keypoints_to,
            interpolate_frames
        )

        output_images = []
        # Assuming canvas_width and canvas_height are present in the pose dictionary
        canvas_width = openpose_dict_from.get("canvas_width", 512) # Default if not found
        canvas_height = openpose_dict_from.get("canvas_height", 512) # Default if not found


        # Generate images for each frame
        for frame_data in interpolated_sequence:
            image_np = gen_skeleton_with_face_hands(
                frame_data['pose_keypoints_2d'],
                frame_data['face_keypoints_2d'],
                frame_data['hand_left_keypoints_2d'],
                frame_data['hand_right_keypoints_2d'],
                canvas_width,
                canvas_height,
                landmarkType
            )

            # Convert to tensor for ComfyUI
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0) # HWC
            output_images.append(image_tensor)

        if not output_images:
             # Fallback: return a black image tensor if no frames were generated
            black_image_np = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            return (torch.from_numpy(black_image_np).unsqueeze(0),)


        tensor_stacked = torch.stack(output_images) # BHWC
        return (tensor_stacked,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Enhanced_Pose_Inter": Pose_Inter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhanced_Pose_Inter": "Pose Interpolation"
}
