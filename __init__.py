import torch
import cv2
import numpy as np
import copy

# --- 스켈레톤, 색상, KP 딕셔너리 정의 ---
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

KP = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

def calculate_bone_length(kps, p1_idx, p2_idx):
    if kps.shape[0] <= max(p1_idx, p2_idx): return 0.0
    if kps[p1_idx, 2] == 0 or kps[p2_idx, 2] == 0: return 0.0
    p1 = kps[p1_idx, :2]
    p2 = kps[p2_idx, :2]
    return np.linalg.norm(p1 - p2)

def get_valid_kps_coords(kps_np, confidence_threshold=0.1):
    if kps_np is None or kps_np.ndim != 2 or kps_np.shape[1] != 3 or kps_np.size == 0: return None
    valid_points = kps_np[kps_np[:, 2] > confidence_threshold][:, :2]
    return valid_points if valid_points.shape[0] > 0 else None

def get_bounding_box_area_and_center(kps_np, confidence_threshold=0.1):
    if kps_np is None or kps_np.ndim != 2 or kps_np.shape[1] != 3 or kps_np.size == 0:
        return 0.0, None
    valid_points_xy = []
    for i in range(kps_np.shape[0]):
        if kps_np[i, 2] > confidence_threshold:
            valid_points_xy.append(kps_np[i, :2])
    if not valid_points_xy or len(valid_points_xy) < 1:
        return 0.0, None
    valid_points_xy = np.array(valid_points_xy)
    min_x, min_y = np.min(valid_points_xy, axis=0)
    max_x, max_y = np.max(valid_points_xy, axis=0)
    width, height = max_x - min_x, max_y - min_y
    area = 0.0
    if valid_points_xy.shape[0] >=2 :
        area = width * height if width > 1e-6 and height > 1e-6 else 0.0
    center = (np.mean(valid_points_xy[:, 0]), np.mean(valid_points_xy[:, 1]))
    return area, center

def adjust_pose_to_reference_size(source_kps, ref_kps, confidence_threshold=0.1):
    if source_kps.size == 0 or ref_kps.size == 0:
        return source_kps
    
    adjusted_kps = source_kps.copy()
    
    print("Stage 1: Global Scaling based on torso.")
    RShoulder, LShoulder, RHip, LHip, Neck = KP["RShoulder"], KP["LShoulder"], KP["RHip"], KP["LHip"], KP["Neck"]
    required_indices = [RShoulder, LShoulder, RHip, LHip, Neck]
    if not all(idx < source_kps.shape[0] and source_kps[idx, 2] > confidence_threshold for idx in required_indices) or \
       not all(idx < ref_kps.shape[0] and ref_kps[idx, 2] > confidence_threshold for idx in required_indices):
        print("Skipping global scaling: Missing critical torso keypoints.")
    else:
        src_shoulder_width = np.linalg.norm(source_kps[LShoulder, :2] - source_kps[RShoulder, :2])
        src_shoulder_center = 0.5 * (source_kps[LShoulder, :2] + source_kps[RShoulder, :2])
        src_hip_center = 0.5 * (source_kps[LHip, :2] + source_kps[RHip, :2])
        src_torso_height = np.linalg.norm(src_shoulder_center - src_hip_center)
        ref_shoulder_width = np.linalg.norm(ref_kps[LShoulder, :2] - ref_kps[RShoulder, :2])
        ref_shoulder_center = 0.5 * (ref_kps[LShoulder, :2] + ref_kps[RShoulder, :2])
        ref_hip_center = 0.5 * (ref_kps[LHip, :2] + ref_kps[RHip, :2])
        ref_torso_height = np.linalg.norm(ref_shoulder_center - ref_hip_center)
        x_ratio = ref_shoulder_width / src_shoulder_width if src_shoulder_width > 1e-6 else 1.0
        y_ratio = ref_torso_height / src_torso_height if src_torso_height > 1e-6 else 1.0
        print(f"Global scaling ratios -> x: {x_ratio:.2f}, y: {y_ratio:.2f}")
        neck_pos = adjusted_kps[Neck, :2].copy()
        for i in range(adjusted_kps.shape[0]):
            if adjusted_kps[i, 2] > 0:
                vec_from_neck = adjusted_kps[i, :2] - neck_pos
                vec_from_neck[0] *= x_ratio
                vec_from_neck[1] *= y_ratio
                adjusted_kps[i, :2] = neck_pos + vec_from_neck
    
    print("Stage 2: Local refinement for each bone.")
    bones_to_adjust = [
        (KP["Neck"], KP["Nose"], [KP["Nose"], KP["REye"], KP["LEye"], KP["REar"], KP["LEar"]]),
        (KP["RShoulder"], KP["RElbow"], [KP["RElbow"], KP["RWrist"]]),
        (KP["RElbow"], KP["RWrist"], [KP["RWrist"]]),
        (KP["LShoulder"], KP["LElbow"], [KP["LElbow"], KP["LWrist"]]),
        (KP["LElbow"], KP["LWrist"], [KP["LWrist"]]),
        (KP["RHip"], KP["RKnee"], [KP["RKnee"], KP["RAnkle"], KP["RBigToe"], KP["RSmallToe"], KP["RHeel"]]),
        (KP["RKnee"], KP["RAnkle"], [KP["RAnkle"], KP["RBigToe"], KP["RSmallToe"], KP["RHeel"]]),
        (KP["LHip"], KP["LKnee"], [KP["LKnee"], KP["LAnkle"], KP["LBigToe"], KP["LSmallToe"], KP["LHeel"]]),
        (KP["LKnee"], KP["LAnkle"], [KP["LAnkle"], KP["LBigToe"], KP["LSmallToe"], KP["LHeel"]]),
    ]
    for parent_idx, child_idx, children_indices in bones_to_adjust:
        if max(parent_idx, child_idx) >= adjusted_kps.shape[0] or max(parent_idx, child_idx) >= ref_kps.shape[0]: continue
        len_source = calculate_bone_length(adjusted_kps, parent_idx, child_idx)
        len_ref = calculate_bone_length(ref_kps, parent_idx, child_idx)
        if len_source == 0 or len_ref == 0: continue
        ratio = len_ref / len_source
        if abs(1.0 - ratio) < 0.01: continue
        parent_pos = adjusted_kps[parent_idx, :2]
        child_pos_old = adjusted_kps[child_idx, :2]
        vector = child_pos_old - parent_pos
        vector_new = vector * ratio
        child_pos_new = parent_pos + vector_new
        offset = child_pos_new - child_pos_old
        for idx in children_indices:
            if idx < adjusted_kps.shape[0] and adjusted_kps[idx, 2] > 0:
                adjusted_kps[idx, :2] += offset
    
    print("Stage 3: Final alignment based on Neck position.")
    if Neck < adjusted_kps.shape[0] and Neck < ref_kps.shape[0] and \
       adjusted_kps[Neck, 2] > confidence_threshold and ref_kps[Neck, 2] > confidence_threshold:
        final_offset = ref_kps[Neck, :2] - adjusted_kps[Neck, :2]
        for i in range(adjusted_kps.shape[0]):
            if adjusted_kps[i, 2] > 0:
                adjusted_kps[i, :2] += final_offset

    head_kp_indices = [KP["Nose"], KP["REye"], KP["LEye"], KP["REar"], KP["LEar"]]
    ref_head_points = np.array([ref_kps[i] for i in head_kp_indices if i < ref_kps.shape[0] and ref_kps[i, 2] > confidence_threshold])
    adj_head_points = np.array([adjusted_kps[i] for i in head_kp_indices if i < adjusted_kps.shape[0] and adjusted_kps[i, 2] > confidence_threshold])
    if ref_head_points.shape[0] >= 2 and adj_head_points.shape[0] >= 2:
        ref_head_area, _ = get_bounding_box_area_and_center(ref_head_points, confidence_threshold)
        adj_head_area, adj_head_center = get_bounding_box_area_and_center(adj_head_points, confidence_threshold)
        if adj_head_area > 1e-6 and ref_head_area > 1e-6 and adj_head_center is not None:
            scale_factor_head = np.sqrt(ref_head_area / adj_head_area)
            if abs(1.0 - scale_factor_head) > 0.01:
                print(f"Scaling head cluster (Nose, Eyes, Ears) by: {scale_factor_head:.2f}")
                center_x, center_y = adj_head_center
                for kp_idx in head_kp_indices:
                    if kp_idx < adjusted_kps.shape[0] and adjusted_kps[kp_idx, 2] > confidence_threshold:
                        x, y, _ = adjusted_kps[kp_idx]
                        adjusted_kps[kp_idx, 0] = center_x + (x - center_x) * scale_factor_head
                        adjusted_kps[kp_idx, 1] = center_y + (y - center_y) * scale_factor_head
    return adjusted_kps

def adjust_face_keypoints_size(full_face_kps_np, scale_factor, center_xy, confidence_threshold=0.1):
    adjusted_kps = full_face_kps_np.copy()
    if center_xy is None : return adjusted_kps 
    center_x, center_y = center_xy
    for i in range(adjusted_kps.shape[0]):
        x, y, conf = adjusted_kps[i]
        if conf > confidence_threshold:
            adjusted_kps[i, :2] = (center_x + (x - center_x) * scale_factor, center_y + (y - center_y) * scale_factor)
    return adjusted_kps

def post_adjust_face_by_chin_distance(ref_body_kps, ref_face_kps, adjusted_body_kps, adjusted_face_kps, confidence_threshold=0.1):
    CHIN_INDEX = 30
    if ref_body_kps.shape[0] <= KP["Nose"] or ref_body_kps[KP["Nose"], 2] < confidence_threshold: return adjusted_face_kps
    if ref_face_kps.shape[0] <= CHIN_INDEX or ref_face_kps[CHIN_INDEX, 2] < confidence_threshold: return adjusted_face_kps
    ref_nose_pos = ref_body_kps[KP["Nose"], :2]
    ref_chin_pos = ref_face_kps[CHIN_INDEX, :2]
    ref_dist = np.linalg.norm(ref_nose_pos - ref_chin_pos)
    if adjusted_body_kps.shape[0] <= KP["Nose"] or adjusted_body_kps[KP["Nose"], 2] < confidence_threshold: return adjusted_face_kps
    if adjusted_face_kps.shape[0] <= CHIN_INDEX or adjusted_face_kps[CHIN_INDEX, 2] < confidence_threshold: return adjusted_face_kps
    adj_nose_pos = adjusted_body_kps[KP["Nose"], :2]
    current_chin_pos = adjusted_face_kps[CHIN_INDEX, :2]
    vec_nose_to_current_chin = current_chin_pos - adj_nose_pos
    current_dist = np.linalg.norm(vec_nose_to_current_chin)
    if current_dist < 1e-6: return adjusted_face_kps
    direction_vec = vec_nose_to_current_chin / current_dist
    desired_chin_pos = adj_nose_pos + direction_vec * ref_dist
    translation_vector = desired_chin_pos - current_chin_pos
    final_face_kps = adjusted_face_kps.copy()
    for i in range(final_face_kps.shape[0]):
        if final_face_kps[i, 2] > confidence_threshold:
            final_face_kps[i, :2] += translation_vector
    print(f"Face translated to match Nose-to-Chin(30) distance. Offset: ({translation_vector[0]:.2f}, {translation_vector[1]:.2f})")
    return final_face_kps

def calculate_hand_intrinsic_properties(hand_kps_np, confidence_threshold=0.1):
    if hand_kps_np is None or hand_kps_np.size == 0: return None
    hand_kp0_abs = None
    if hand_kps_np.shape[0] > 0 and hand_kps_np[0, 2] > confidence_threshold:
        hand_kp0_abs = hand_kps_np[0, :2].copy()
    valid_hand_coords_xy = get_valid_kps_coords(hand_kps_np, confidence_threshold)
    scale = 0.0
    if valid_hand_coords_xy is not None and valid_hand_coords_xy.shape[0] >= 2:
        min_x, min_y = np.min(valid_hand_coords_xy, axis=0)
        max_x, max_y = np.max(valid_hand_coords_xy, axis=0)
        width, height = max_x - min_x, max_y - min_y
        scale = np.sqrt(width**2 + height**2) if width > 1e-6 and height > 1e-6 else 0.0
    return {'scale': scale, 'kp0_abs': hand_kp0_abs}

def transform_hand_final(target_hand_kps_np_orig, target_body_wrist_pos_xy_adj, ref_hand_kps_np_orig, ref_body_wrist_pos_xy_orig, confidence_threshold=0.1):
    ref_props = calculate_hand_intrinsic_properties(ref_hand_kps_np_orig, confidence_threshold)
    target_orig_props = calculate_hand_intrinsic_properties(target_hand_kps_np_orig, confidence_threshold)
    if not all([ref_props, target_orig_props, ref_props.get('kp0_abs') is not None, target_orig_props.get('kp0_abs') is not None, ref_body_wrist_pos_xy_orig is not None, target_body_wrist_pos_xy_adj is not None]):
        return target_hand_kps_np_orig.flatten().tolist() if target_hand_kps_np_orig is not None and target_hand_kps_np_orig.size > 0 else []
    ref_hand_kp0_pos, ref_scale = ref_props['kp0_abs'], ref_props['scale']
    target_orig_hand_kp0_pos, target_orig_scale = target_orig_props['kp0_abs'], target_orig_props['scale']
    ref_offset_bodywrist_to_handkp0 = ref_hand_kp0_pos - ref_body_wrist_pos_xy_orig
    scale_factor = ref_scale / target_orig_scale if target_orig_scale > 1e-6 else 1.0
    scaled_target_hand_kps = target_hand_kps_np_orig.copy()
    pivot_for_scaling = target_orig_hand_kp0_pos.copy() 
    for i in range(scaled_target_hand_kps.shape[0]):
        if scaled_target_hand_kps[i, 2] > confidence_threshold:
            vec_from_pivot = scaled_target_hand_kps[i, :2] - pivot_for_scaling
            scaled_target_hand_kps[i, :2] = pivot_for_scaling + (vec_from_pivot * scale_factor)
    if scaled_target_hand_kps.shape[0] == 0 or scaled_target_hand_kps[0, 2] <= confidence_threshold:
        return scaled_target_hand_kps.flatten().tolist()
    current_abs_pos_of_scaled_target_hand_kp0 = scaled_target_hand_kps[0, :2]
    desired_abs_pos_for_target_hand_kp0 = target_body_wrist_pos_xy_adj + ref_offset_bodywrist_to_handkp0
    translation_vector = desired_abs_pos_for_target_hand_kp0 - current_abs_pos_of_scaled_target_hand_kp0
    for i in range(scaled_target_hand_kps.shape[0]):
        if scaled_target_hand_kps[i, 2] > confidence_threshold:
            scaled_target_hand_kps[i, :2] += translation_vector
    return scaled_target_hand_kps.flatten().tolist()

def draw_keypoints_and_skeleton(image, keypoints_data, skeleton_connections, colors_config, confidence_threshold=0.1, point_radius=3, line_thickness=2, is_face=False, is_hand=False):
    if not keypoints_data: return
    tri_tuples = [keypoints_data[i:i + 3] for i in range(0, len(keypoints_data), 3)]
    if skeleton_connections:
        for i, (joint_idx_a, joint_idx_b) in enumerate(skeleton_connections):
            if joint_idx_a >= len(tri_tuples) or joint_idx_b >= len(tri_tuples): continue
            a_x, a_y, a_confidence = tri_tuples[joint_idx_a]
            b_x, b_y, b_confidence = tri_tuples[joint_idx_b]
            if a_confidence >= confidence_threshold and b_confidence >= confidence_threshold:
                limb_color = colors_config['limbs'][i % len(colors_config['limbs'])] if is_hand else colors_config[i % len(colors_config)]
                if limb_color is not None: cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), limb_color, line_thickness)
    for i, (x, y, confidence) in enumerate(tri_tuples):
        if confidence >= confidence_threshold:
            point_color, current_radius = (colors_config['points'], point_radius) if is_hand else (colors_config, 2) if is_face else (colors_config[i % len(colors_config)], point_radius)
            if point_color is not None: cv2.circle(image, (int(x), int(y)), current_radius, point_color, -1)

def gen_skeleton_with_face_hands(pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d, canvas_width, canvas_height, landmarkType, confidence_threshold=0.1):
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    def scale_keypoints(keypoints, target_w, target_h, input_is_normalized):
        if not keypoints: return []
        scaled = []
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            scaled.extend([x * target_w, y * target_h, conf] if input_is_normalized else [x, y, conf])
        return scaled
    input_normalized = (landmarkType == "OpenPose")
    scaled_pose = scale_keypoints(pose_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_face = scale_keypoints(face_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_left = scale_keypoints(hand_left_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_right = scale_keypoints(hand_right_keypoints_2d, canvas_width, canvas_height, input_normalized)
    draw_keypoints_and_skeleton(image, scaled_pose, body_skeleton, body_colors, confidence_threshold, point_radius=6, line_thickness=4)
    if scaled_face: draw_keypoints_and_skeleton(image, scaled_face, face_skeleton, face_color, confidence_threshold, point_radius=2, line_thickness=1, is_face=True)
    hand_colors_config = {'limbs': hand_limb_colors, 'points': hand_keypoint_color}
    if scaled_hand_left: draw_keypoints_and_skeleton(image, scaled_hand_left, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    if scaled_hand_right: draw_keypoints_and_skeleton(image, scaled_hand_right, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    return image

def transform_all_keypoints(keypoints_1, keypoints_2, frames, interpolation="linear"):
    def interpolate_keypoint_set(kp1, kp2, num_frames, interp_method):
        if not kp1 and not kp2: return [[] for _ in range(num_frames)]
        if not kp1: kp1 = [0.0] * len(kp2) if kp2 else [] 
        if not kp2: kp2 = [0.0] * len(kp1) if kp1 else [] 
        if not kp1 and not kp2: return [[] for _ in range(num_frames)] 
        if len(kp1) != len(kp2):
            print(f"Warning: Keypoint sets have different lengths ({len(kp1)} vs {len(kp2)}). Returning empty sequence.")
            return [[] for _ in range(num_frames)]
        tri_tuples_1 = [kp1[i:i + 3] for i in range(0, len(kp1), 3)]
        tri_tuples_2 = [kp2[i:i + 3] for i in range(0, len(kp2), 3)]
        if not tri_tuples_1 and not tri_tuples_2 : return [[] for _ in range(num_frames)]
        if not tri_tuples_1: tri_tuples_1 = [[0,0,0]] * len(tri_tuples_2) 
        if not tri_tuples_2: tri_tuples_2 = [[0,0,0]] * len(tri_tuples_1)
        keypoints_sequence = []
        for j in range(num_frames):
            interpolated_kps_for_frame = []
            t = j / float(num_frames - 1) if num_frames > 1 else 0.0
            if interp_method == "ease-in": interp_factor = t * t
            elif interp_method == "ease-out": interp_factor = 1 - (1 - t) * (1 - t)
            elif interp_method == "ease-in-out":
                interp_factor = 4 * t * t * t if t < 0.5 else 0.5 * (2 * t - 2)**3 + 1
            else: interp_factor = t
            for i in range(len(tri_tuples_1)):
                x1, y1, c1 = tri_tuples_1[i]; x2, y2, c2 = tri_tuples_2[i]
                new_x, new_y, new_c = 0.0, 0.0, 0.0
                if c1 > 0 and c2 > 0:
                    new_x, new_y = x1 + (x2 - x1) * interp_factor, y1 + (y2 - y1) * interp_factor
                    new_c = c1 + (c2 - c1) * interp_factor
                elif c1 > 0:
                    new_x, new_y, new_c = x1, y1, c1 * (1.0 - interp_factor)
                elif c2 > 0:
                    new_x, new_y, new_c = x2, y2, c2 * interp_factor
                interpolated_kps_for_frame.extend([new_x, new_y, new_c])
            keypoints_sequence.append(interpolated_kps_for_frame)
        return keypoints_sequence
    parts = ['pose', 'face', 'hand_left', 'hand_right']
    sequences = {part: interpolate_keypoint_set(
                    keypoints_1.get(f'{part}_keypoints_2d', []),
                    keypoints_2.get(f'{part}_keypoints_2d', []),
                    frames, interpolation) for part in parts}
    combined_sequence = []
    for i in range(frames):
        combined_frame = {f'{part}_keypoints_2d': sequences[part][i] if i < len(sequences[part]) else [] for part in parts}
        combined_sequence.append(combined_frame)
    return combined_sequence

class Pose_Inter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        interpolation_methods = ["linear", "ease-in", "ease-out", "ease-in-out"]
        return {
            "required": {
                "pose_from": ("POSE_KEYPOINT", ), "pose_to": ("POSE_KEYPOINT", ),
                "interpolate_frames": ("INT", {"default": 12, "min": 2, "max": 240, "step": 1}),
                "interpolation": (interpolation_methods, {"default": "linear"}),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adjust_body_shape": ("BOOLEAN", {"default": True}),
                "landmarkType": (["OpenPose", "DWPose"], {"default": "DWPose"}),
                "include_face": ("BOOLEAN", {"default": True}),
                "include_hands": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Pose Interpolation"

    def run(self, pose_from, pose_to, interpolate_frames, interpolation, confidence_threshold, landmarkType, include_face, include_hands, adjust_body_shape):
        if not pose_from or not pose_to: raise ValueError("Input pose data is empty.")
        
        openpose_dict_from = pose_from[0] if isinstance(pose_from, list) and pose_from else pose_from
        openpose_dict_to = pose_to[0] if isinstance(pose_to, list) and pose_to else pose_to
        
        if "people" not in openpose_dict_from or not openpose_dict_from["people"]: raise ValueError("No people found in 'pose_from'.")
        if "people" not in openpose_dict_to or not openpose_dict_to["people"]: raise ValueError("No people found in 'pose_to'.")
        
        person_from = openpose_dict_from["people"][0]
        person_to = openpose_dict_to["people"][0] 
        
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
        
        original_person_to_face_kps = person_to.get("face_keypoints_2d", []) if include_face else []
        original_person_to_hand_left_kps = person_to.get("hand_left_keypoints_2d", []) if include_hands else []
        original_person_to_hand_right_kps = person_to.get("hand_right_keypoints_2d", []) if include_hands else []
        
        kps_from_np_body = np.array(keypoints_from['pose_keypoints_2d']).reshape(-1, 3) if keypoints_from['pose_keypoints_2d'] else np.array([])
        kps_to_np_body_for_adjustment = np.array(keypoints_to['pose_keypoints_2d']).reshape(-1, 3) if keypoints_to['pose_keypoints_2d'] else np.array([])
        kps_to_np_body_adjusted_for_face_hands = kps_to_np_body_for_adjustment.copy()
        
        if adjust_body_shape:
            print("Adjusting 'pose_to' BODY shape using improved 3-stage method...")
            if kps_from_np_body.size > 0 and kps_to_np_body_for_adjustment.size > 0:
                adjusted_body_kps = adjust_pose_to_reference_size(kps_to_np_body_for_adjustment, kps_from_np_body, confidence_threshold)
                keypoints_to['pose_keypoints_2d'] = adjusted_body_kps.flatten().tolist()
                kps_to_np_body_adjusted_for_face_hands = adjusted_body_kps 
                print("Body shape adjustment complete.")
            else:
                print("Skipping body adjustment (empty or malformed keypoints).")

            if include_face:
                print("Adjusting 'pose_to' FACE pose...")
                face_kps_from_np = np.array(keypoints_from['face_keypoints_2d']).reshape(-1, 3) if keypoints_from['face_keypoints_2d'] else np.array([])
                face_kps_to_np_orig = np.array(original_person_to_face_kps).reshape(-1, 3) if original_person_to_face_kps else np.array([])
                final_adjusted_face_kps_np = face_kps_to_np_orig.copy()
                if face_kps_from_np.size > 0 and face_kps_to_np_orig.size > 0:
                    area_from, _ = get_bounding_box_area_and_center(face_kps_from_np, confidence_threshold)
                    area_to_orig, center_to_face_orig = get_bounding_box_area_and_center(face_kps_to_np_orig, confidence_threshold)
                    if area_to_orig > 1e-6 and area_from > 1e-6 and center_to_face_orig is not None:
                        scale_factor = np.sqrt(area_from / area_to_orig)
                        final_adjusted_face_kps_np = adjust_face_keypoints_size(face_kps_to_np_orig, scale_factor, center_to_face_orig, confidence_threshold)
                    final_adjusted_face_kps_np = post_adjust_face_by_chin_distance(kps_from_np_body, face_kps_from_np, kps_to_np_body_adjusted_for_face_hands, final_adjusted_face_kps_np, confidence_threshold)
                    keypoints_to['face_keypoints_2d'] = final_adjusted_face_kps_np.flatten().tolist()

            if include_hands:
                print("Adjusting 'pose_to' HAND pose...")
                for hand_type in ["left", "right"]:
                    wrist_kp_name = "LWrist" if hand_type == "left" else "RWrist"
                    ref_body_wrist_pos_xy_orig = None
                    if kps_from_np_body.size > 0 and KP[wrist_kp_name] < kps_from_np_body.shape[0] and kps_from_np_body[KP[wrist_kp_name], 2] > confidence_threshold:
                        ref_body_wrist_pos_xy_orig = kps_from_np_body[KP[wrist_kp_name], :2]
                    target_body_wrist_pos_xy_adj = None
                    if kps_to_np_body_adjusted_for_face_hands.size > 0 and KP[wrist_kp_name] < kps_to_np_body_adjusted_for_face_hands.shape[0] and kps_to_np_body_adjusted_for_face_hands[KP[wrist_kp_name], 2] > confidence_threshold:
                        target_body_wrist_pos_xy_adj = kps_to_np_body_adjusted_for_face_hands[KP[wrist_kp_name], :2]
                    
                    ref_hand_kps_list = keypoints_from.get(f'hand_{hand_type}_keypoints_2d', [])
                    if ref_hand_kps_list and len(ref_hand_kps_list) % 3 == 0:
                        ref_hand_kps_np_orig = np.array(ref_hand_kps_list).reshape(-1, 3)
                    else:
                        ref_hand_kps_np_orig = np.array([])

                    target_hand_kps_list_orig = original_person_to_hand_left_kps if hand_type == "left" else original_person_to_hand_right_kps
                    if target_hand_kps_list_orig and len(target_hand_kps_list_orig) % 3 == 0:
                        target_hand_kps_np_orig = np.array(target_hand_kps_list_orig).reshape(-1, 3)
                    else:
                        target_hand_kps_np_orig = np.array([])

                    if ref_body_wrist_pos_xy_orig is None or target_body_wrist_pos_xy_adj is None or ref_hand_kps_np_orig.size == 0 or target_hand_kps_np_orig.size == 0:
                        print(f"Skipping {hand_type} hand adjustment: Missing critical keypoints.")
                        continue

                    adjusted_hand_kps_list = transform_hand_final(target_hand_kps_np_orig, target_body_wrist_pos_xy_adj, ref_hand_kps_np_orig, ref_body_wrist_pos_xy_orig, confidence_threshold)
                    keypoints_to[f'hand_{hand_type}_keypoints_2d'] = adjusted_hand_kps_list
                    print(f"{hand_type.capitalize()} hand pose adjustment complete.")

        interpolated_sequence = transform_all_keypoints(keypoints_from, keypoints_to, interpolate_frames, interpolation)
        
        output_images = []
        canvas_width = openpose_dict_from.get("canvas_width", 512)
        canvas_height = openpose_dict_from.get("canvas_height", 512)

        for frame_data in interpolated_sequence:
            image_np = gen_skeleton_with_face_hands(
                frame_data['pose_keypoints_2d'], 
                frame_data['face_keypoints_2d'],
                frame_data['hand_left_keypoints_2d'], 
                frame_data['hand_right_keypoints_2d'],
                canvas_width, canvas_height, landmarkType, confidence_threshold
            )
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
            output_images.append(image_tensor)
        
        if not output_images:
            black_image_np = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            return (torch.from_numpy(black_image_np).unsqueeze(0),)
            
        return (torch.stack(output_images),)

NODE_CLASS_MAPPINGS = {
    "Pose_Inter": Pose_Inter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Pose_Inter": "Pose Interpolation"
}
