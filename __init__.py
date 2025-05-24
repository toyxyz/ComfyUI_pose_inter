import torch
import cv2
import numpy as np

# --- 스켈레톤 및 색상 정의 (이전과 동일) ---
body_colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]
face_color = [255, 255, 255]
hand_keypoint_color = [0, 0, 255]
hand_limb_colors = [
    [255,0,0],[255,60,0],[255,120,0],[255,180,0], [180,255,0],[120,255,0],[60,255,0],[0,255,0],
    [0,255,60],[0,255,120],[0,255,180],[0,180,255], [0,120,255],[0,60,255],[0,0,255],[60,0,255],
    [120,0,255],[180,0,255],[255,0,180],[255,0,120]
]
body_skeleton = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]
face_skeleton = []
hand_skeleton = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
]

# --- 그리기 함수 수정 ---
def draw_keypoints_and_skeleton(image, keypoints_data, skeleton_connections, colors_config, confidence_threshold=0.1, point_radius=3, line_thickness=2, is_face=False, is_hand=False):
    """ confidence_threshold 인자 추가 """
    if not keypoints_data:
        return
    tri_tuples = [keypoints_data[i:i + 3] for i in range(0, len(keypoints_data), 3)]
    
    # 뼈대 그리기
    if skeleton_connections:
        for i, (joint_idx_a, joint_idx_b) in enumerate(skeleton_connections):
            if joint_idx_a >= len(tri_tuples) or joint_idx_b >= len(tri_tuples):
                continue
            a_x, a_y, a_confidence = tri_tuples[joint_idx_a]
            b_x, b_y, b_confidence = tri_tuples[joint_idx_b]
            
            # 변경된 조건: 두 관절의 신뢰도가 모두 임계값 이상일 때만 선을 그림
            if a_confidence >= confidence_threshold and b_confidence >= confidence_threshold:
                limb_color = None
                if is_hand:
                    limb_color_list = colors_config['limbs']
                    limb_color = limb_color_list[i % len(limb_color_list)]
                else:
                    limb_color_list = colors_config
                    limb_color = limb_color_list[i % len(limb_color_list)]
                if limb_color is not None:
                    cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), limb_color, line_thickness)

    # 관절 그리기
    for i, (x, y, confidence) in enumerate(tri_tuples):
        # 변경된 조건: 신뢰도가 임계값 이상일 때만 점을 그림
        if confidence >= confidence_threshold:
            point_color = None
            current_radius = point_radius
            if is_hand:
                point_color = colors_config['points']
            elif is_face:
                point_color = colors_config
                current_radius = 2
            else:
                point_color_list = colors_config
                point_color = point_color_list[i % len(point_color_list)]
            if point_color is not None:
                cv2.circle(image, (int(x), int(y)), current_radius, point_color, -1)

# --- 이미지 생성 함수 수정 ---
def gen_skeleton_with_face_hands(pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d,
                                 canvas_width, canvas_height, landmarkType, confidence_threshold=0.1):
    """ confidence_threshold 인자 추가 """
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    def scale_keypoints(keypoints, target_w, target_h, input_is_normalized):
        if not keypoints: return []
        scaled = []
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            if input_is_normalized: scaled.extend([x * target_w, y * target_h, conf])
            else: scaled.extend([x, y, conf])
        return scaled
    
    input_normalized = (landmarkType == "OpenPose")
    scaled_pose = scale_keypoints(pose_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_face = scale_keypoints(face_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_left = scale_keypoints(hand_left_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_right = scale_keypoints(hand_right_keypoints_2d, canvas_width, canvas_height, input_normalized)
    
    # 그리기 함수에 임계값 전달
    draw_keypoints_and_skeleton(image, scaled_pose, body_skeleton, body_colors, confidence_threshold, point_radius=6, line_thickness=4)
    if scaled_face:
        draw_keypoints_and_skeleton(image, scaled_face, face_skeleton, face_color, confidence_threshold, point_radius=2, line_thickness=1, is_face=True)
    hand_colors_config = {'limbs': hand_limb_colors, 'points': hand_keypoint_color}
    if scaled_hand_left:
        draw_keypoints_and_skeleton(image, scaled_hand_left, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    if scaled_hand_right:
        draw_keypoints_and_skeleton(image, scaled_hand_right, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    return image

# --- 보간 함수 (이전과 동일) ---
def transform_all_keypoints(keypoints_1, keypoints_2, frames, interpolation="linear"):
    # ... (내부 로직은 이전 버전과 동일)
    def interpolate_keypoint_set(kp1, kp2, num_frames, interp_method):
        if not kp1 and not kp2: return [[] for _ in range(num_frames)]
        if not kp1: kp1 = [0.0] * len(kp2)
        if not kp2: kp2 = [0.0] * len(kp1)
        if len(kp1) != len(kp2):
            return [[] for _ in range(num_frames)]
        
        tri_tuples_1 = [kp1[i:i + 3] for i in range(0, len(kp1), 3)]
        tri_tuples_2 = [kp2[i:i + 3] for i in range(0, len(kp2), 3)]
        if not tri_tuples_1:
            return [[] for _ in range(num_frames)]

        keypoints_sequence = []
        for j in range(num_frames):
            interpolated_kps_for_frame = []
            t = j / float(num_frames - 1) if num_frames > 1 else 0.0
            if interp_method == "ease-in": interp_factor = t * t
            elif interp_method == "ease-out": interp_factor = 1 - (1 - t) * (1 - t)
            elif interp_method == "ease-in-out":
                if t < 0.5: interp_factor = 4 * t * t * t
                else: p = 2 * t - 2; interp_factor = 0.5 * p * p * p + 1
            else: interp_factor = t

            for i in range(len(tri_tuples_1)):
                x1, y1, c1 = tri_tuples_1[i]
                x2, y2, c2 = tri_tuples_2[i]
                new_x, new_y, new_c = 0.0, 0.0, 0.0
                if c1 > 0 and c2 > 0:
                    new_x = x1 + (x2 - x1) * interp_factor
                    new_y = y1 + (y2 - y1) * interp_factor
                    new_c = c1 + (c2 - c1) * interp_factor
                elif c1 > 0 and c2 == 0:
                    new_x, new_y = x1, y1
                    new_c = c1 * (1.0 - interp_factor)
                elif c1 == 0 and c2 > 0:
                    new_x, new_y = x2, y2
                    new_c = c2 * interp_factor
                interpolated_kps_for_frame.extend([new_x, new_y, new_c])
            keypoints_sequence.append(interpolated_kps_for_frame)
        return keypoints_sequence

    pose_1 = keypoints_1.get('pose_keypoints_2d', [])
    face_1 = keypoints_1.get('face_keypoints_2d', [])
    hand_left_1 = keypoints_1.get('hand_left_keypoints_2d', [])
    hand_right_1 = keypoints_1.get('hand_right_keypoints_2d', [])
    pose_2 = keypoints_2.get('pose_keypoints_2d', [])
    face_2 = keypoints_2.get('face_keypoints_2d', [])
    hand_left_2 = keypoints_2.get('hand_left_keypoints_2d', [])
    hand_right_2 = keypoints_2.get('hand_right_keypoints_2d', [])
    
    pose_sequence = interpolate_keypoint_set(pose_1, pose_2, frames, interpolation)
    face_sequence = interpolate_keypoint_set(face_1, face_2, frames, interpolation)
    hand_left_sequence = interpolate_keypoint_set(hand_left_1, hand_left_2, frames, interpolation)
    hand_right_sequence = interpolate_keypoint_set(hand_right_1, hand_right_2, frames, interpolation)

    combined_sequence = []
    for i in range(frames):
        combined_frame = {
            'pose_keypoints_2d': pose_sequence[i], 'face_keypoints_2d': face_sequence[i],
            'hand_left_keypoints_2d': hand_left_sequence[i], 'hand_right_keypoints_2d': hand_right_sequence[i]
        }
        combined_sequence.append(combined_frame)
    return combined_sequence

# --- ComfyUI 클래스 수정 ---
class Pose_Inter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        interpolation_methods = ["linear", "ease-in", "ease-out", "ease-in-out"]
        return {
            "required": {
                "pose_from": ("POSE_KEYPOINT", ),
                "pose_to": ("POSE_KEYPOINT", ),
                "interpolate_frames": ("INT", {"default": 12, "min": 2, "max": 240, "step": 1}),
                "interpolation": (interpolation_methods, {"default": "ease-in-out"}),
                # 신뢰도 임계값 슬라이더 추가
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "landmarkType": (["OpenPose", "DWPose"], ),
                "include_face": ("BOOLEAN", {"default": True}),
                "include_hands": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Pose Interpolation"

    def run(self, pose_from, pose_to, interpolate_frames, interpolation, confidence_threshold, landmarkType, include_face, include_hands):
        # ... (이전과 동일한 입력 데이터 처리 부분)
        if not pose_from or not pose_to: raise ValueError("Input poses cannot be empty.")
        openpose_dict_from = pose_from[0]
        openpose_dict_to = pose_to[0]
        if "people" not in openpose_dict_from or not openpose_dict_from["people"]: raise ValueError("pose_from does not contain 'people' data.")
        if "people" not in openpose_dict_to or not openpose_dict_to["people"]: raise ValueError("pose_to does not contain 'people' data.")
        person_from = openpose_dict_from["people"][0]
        person_to = openpose_dict_to["people"][0]
        keypoints_from = {'pose_keypoints_2d': person_from.get("pose_keypoints_2d", []), 'face_keypoints_2d': person_from.get("face_keypoints_2d", []) if include_face else [], 'hand_left_keypoints_2d': person_from.get("hand_left_keypoints_2d", []) if include_hands else [], 'hand_right_keypoints_2d': person_from.get("hand_right_keypoints_2d", []) if include_hands else []}
        keypoints_to = {'pose_keypoints_2d': person_to.get("pose_keypoints_2d", []), 'face_keypoints_2d': person_to.get("face_keypoints_2d", []) if include_face else [], 'hand_left_keypoints_2d': person_to.get("hand_left_keypoints_2d", []) if include_hands else [], 'hand_right_keypoints_2d': person_to.get("hand_right_keypoints_2d", []) if include_hands else []}
        
        interpolated_sequence = transform_all_keypoints(keypoints_from, keypoints_to, interpolate_frames, interpolation)

        output_images = []
        canvas_width = openpose_dict_from.get("canvas_width", 512)
        canvas_height = openpose_dict_from.get("canvas_height", 512)

        for frame_data in interpolated_sequence:
            # 이미지 생성 함수에 임계값 전달
            image_np = gen_skeleton_with_face_hands(
                frame_data['pose_keypoints_2d'],
                frame_data['face_keypoints_2d'],
                frame_data['hand_left_keypoints_2d'],
                frame_data['hand_right_keypoints_2d'],
                canvas_width,
                canvas_height,
                landmarkType,
                confidence_threshold
            )
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
            output_images.append(image_tensor)

        if not output_images:
            black_image_np = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            return (torch.from_numpy(black_image_np).unsqueeze(0),)

        tensor_stacked = torch.stack(output_images)
        return (tensor_stacked,)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "Enhanced_Pose_Inter": Pose_Inter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhanced_Pose_Inter": "Pose Interpolation"
}
