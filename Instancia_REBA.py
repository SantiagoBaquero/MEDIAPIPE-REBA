import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


reba_table_a = np.array([
    [
        [1, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8]
    ],
    [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9]
    ],
    [
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 9]
    ],
    [
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 9],
        [8, 9, 9, 9]
    ]
])

reba_table_b = np.array([
    [
        [1, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8]
    ],
    [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9]
    ],
    [
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 9]
    ],
    [
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 9],
        [9, 9]
    ],
    [
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 9],
        [9, 9],
        [9, 9]
    ],
    [
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 9],
        [9, 9],
        [9, 9],
        [9, 9]
    ]
])

reba_final_table = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [4, 5, 6, 7],
    [5, 6, 7, 8],
    [5, 6, 7, 8],
    [6, 7, 8, 9],
    [7, 8, 9, 10]
])

reba_table_c_matrix = np.array([
    [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
    [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    [3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
    [4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
    [4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    [5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9],
    [5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 9]
])

def calculate_angle_with_vectors(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    vec_ba = a - b
    vec_bc = c - b
    if np.linalg.norm(vec_ba) == 0 or np.linalg.norm(vec_bc) == 0:
        return 0.0, 0.0, 0.0
    angle_vec_ba_rad = np.arctan2(vec_ba[1], vec_ba[0])
    angle_vec_bc_rad = np.arctan2(vec_bc[1], vec_bc[0])
    angle_rad = np.abs(angle_vec_bc_rad - angle_vec_ba_rad)
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 180.0:
        angle_deg = 360 - angle_deg
    return angle_deg, angle_vec_ba_rad, angle_vec_bc_rad

def draw_interior_arc(frame, center_px, angle_deg, vec1_rad, vec2_rad, arc_radius, color, thickness):
    arc_radius_int = int(arc_radius)
    start_angle_deg = np.degrees(vec1_rad)
    end_angle_deg = np.degrees(vec2_rad)
    diff_angle = (end_angle_deg - start_angle_deg + 360) % 360
    if start_angle_deg > end_angle_deg:
        start_angle_deg, end_angle_deg = end_angle_deg, start_angle_deg
    if diff_angle > 180:
        temp_start = start_angle_deg
        start_angle_deg = end_angle_deg
        end_angle_deg = temp_start + 360 if temp_start < end_angle_deg else temp_start - 360
    cv2.ellipse(frame, center_px, (arc_radius_int, arc_radius_int), 0, start_angle_deg, end_angle_deg, color, thickness)

def get_neck_score_reba(neck_angle_deg, is_twisted=False, is_laterally_flexed=False):
    score = 0
    if 0 <= neck_angle_deg <= 10:
        score = 1
    elif 10 < neck_angle_deg <= 20:
        score = 2
    elif neck_angle_deg > 20:
        score = 3
    if is_twisted or is_laterally_flexed:
        score += 1
    return score

def get_trunk_score_reba(trunk_angle_deg, is_twisted=False, is_laterally_flexed=False):
    score = 0
    if 0 <= trunk_angle_deg <= 5:
        score = 1
    elif 5 < trunk_angle_deg <= 20:
        score = 2
    elif 20 < trunk_angle_deg <= 60:
        score = 3
    elif trunk_angle_deg > 60:
        score = 4
    if is_twisted or is_laterally_flexed:
        score += 1
    return score

def get_leg_score_reba(hip_angle_deg, is_bilateral_support=True, is_sitting=False, is_unstable_support=False,
                       visibility_ok=True):
    if not visibility_ok:
        return 4
    score = 1
    if hip_angle_deg > 20:
        score += 1
    if is_bilateral_support and not is_sitting:
        pass
    elif is_sitting:
        score += 1
    elif not is_bilateral_support or is_unstable_support:
        score += 2
    return min(score, 4)

def get_upper_arm_score_reba(arm_angle_deg, is_shoulder_raised=False, is_arm_abducted=False, is_arm_supported=False):
    score = 0
    if 0 <= arm_angle_deg <= 20:
        score = 1
    elif 20 < arm_angle_deg <= 45:
        score = 2
    elif 45 < arm_angle_deg <= 90:
        score = 3
    elif arm_angle_deg > 90:
        score = 4
    if is_shoulder_raised:
        score += 1
    if is_arm_abducted:
        score += 1
    if is_arm_supported:
        score -= 1
    return max(1, score)

def get_forearm_score_reba(forearm_angle_deg):
    if 60 <= forearm_angle_deg <= 100:
        return 1
    else:
        return 2

def get_wrist_score_reba(wrist_angle_deg, is_deviated=False, is_twisted=False):
    score = 0
    if 0 <= wrist_angle_deg < 10:
        score = 1
    elif wrist_angle_deg >= 10:
        score = 2
    if is_deviated:
        score += 1
    if is_twisted:
        score += 1
    return score

def get_score_a(trunk_score, neck_score, leg_score):
    trunk_idx = max(0, min(trunk_score - 1, reba_table_a.shape[0] - 1))
    neck_idx = max(0, min(neck_score - 1, reba_table_a.shape[2] - 1))
    leg_idx = max(0, min(leg_score - 1, reba_table_a.shape[1] - 1))
    return reba_table_a[trunk_idx, leg_idx, neck_idx]

def get_score_b(upper_arm_score, forearm_score, wrist_score):
    upper_arm_idx = max(0, min(upper_arm_score - 1, reba_table_b.shape[0] - 1))
    forearm_idx = max(0, min(forearm_score - 1, reba_table_b.shape[2] - 1))
    wrist_idx = max(0, min(wrist_score - 1, reba_table_b.shape[1] - 1))
    return reba_table_b[upper_arm_idx, wrist_idx, forearm_idx]

def get_score_c(score_a, score_b, load_score, coupling_score):
    score_a_idx = max(0, min(score_a - 1, reba_table_c_matrix.shape[1] - 1))
    score_b_idx = max(0, min(score_b - 1, reba_table_c_matrix.shape[0] - 1))
    score_c_base = reba_table_c_matrix[score_b_idx, score_a_idx]
    return score_c_base + load_score + coupling_score

def get_reba_final_score(score_c_final, activity_score):
    score_c_idx = max(0, min(score_c_final - 1, reba_final_table.shape[0] - 1))
    activity_idx = max(0, min(activity_score - 1, reba_final_table.shape[1] - 1))
    return reba_final_table[score_c_idx, activity_idx]

def get_reba_risk_level(final_reba_score):
    if final_reba_score == 1:
        return "Riesgo Insignificante"
    elif 2 <= final_reba_score <= 3:
        return "Riesgo Bajo"
    elif 4 <= final_reba_score <= 7:
        return "Riesgo Medio"
    elif 8 <= final_reba_score <= 10:
        return "Riesgo Alto"
    elif final_reba_score >= 11:
        return "Riesgo Muy Alto"
    return "N/A"

video_path = "Video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: No se pudo abrir el archivo de video '{video_path}'. Intentando abrir la webcam (índice 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam. Saliendo del programa.")
        exit()
    else:
        print("Webcam abierta exitosamente.")
else:
    print(f"Video '{video_path}' abierto exitosamente.")

prev_frame_time = 0
new_frame_time = 0

original_stderr = sys.stderr
devnull = open(os.devnull, 'w')
sys.stderr = devnull

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o no se pudo leer el fotograma. Saliendo.")
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = pose.process(frame_rgb)
        frame.flags.writeable = True

        reba_score_display = "N/A"
        reba_risk_display = "N/A"
        pose_detected_status = "No Pose Detectada"
        debug_info = ""
        angle_info = []

        if results.pose_landmarks:
            pose_detected_status = "Pose Detectada!"
            landmarks = results.pose_landmarks.landmark

            try:
                mid_shoulders_norm_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2
                mid_shoulders_norm_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
                mid_shoulders_norm = [mid_shoulders_norm_x, mid_shoulders_norm_y]
                mid_shoulders_px = (int(mid_shoulders_norm_x * width), int(mid_shoulders_norm_y * height))

                left_ear_norm = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                right_ear_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                if landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility:
                    ear_norm = left_ear_norm
                    ear_px = (int(ear_norm[0] * width), int(ear_norm[1] * height))
                else:
                    ear_norm = right_ear_norm
                    ear_px = (int(ear_norm[0] * width), int(ear_norm[1] * height))

                left_hip_norm = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                mid_hips_norm_x = (left_hip_norm[0] + right_hip_norm[0]) / 2
                mid_hips_norm_y = (left_hip_norm[1] + right_hip_norm[1]) / 2
                mid_hips_norm = [mid_hips_norm_x, mid_hips_norm_y]
                mid_hips_px = (int(mid_hips_norm_x * width), int(mid_hips_norm_y * height))

                left_shoulder_norm = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow_norm = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_wrist_norm = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_shoulder_px = (int(left_shoulder_norm[0] * width), int(left_shoulder_norm[1] * height))
                right_shoulder_px = (int(right_shoulder_norm[0] * width), int(right_shoulder_norm[1] * height))
                left_elbow_px = (int(left_elbow_norm[0] * width), int(left_elbow_norm[1] * height))
                right_elbow_px = (int(right_elbow_norm[0] * width), int(right_elbow_norm[1] * height))
                left_wrist_px = (int(left_wrist_norm[0] * width), int(left_wrist_norm[1] * height))
                right_wrist_px = (int(right_wrist_norm[0] * width), int(right_wrist_norm[1] * height))

                left_knee_norm = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_ankle_norm = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_hip_px = (int(left_hip_norm[0] * width), int(left_hip_norm[1] * height))
                right_hip_px = (int(right_hip_norm[0] * width), int(right_hip_norm[1] * height))
                left_knee_px = (int(left_knee_norm[0] * width), int(left_knee_norm[1] * height))
                right_knee_px = (int(right_knee_norm[0] * width), int(right_knee_norm[1] * height))
                left_ankle_px = (int(left_ankle_norm[0] * width), int(left_ankle_norm[1] * height))
                right_ankle_px = (int(right_ankle_norm[0] * width), int(right_ankle_norm[1] * height))

                left_index_norm = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                right_index_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                left_index_px = (int(left_index_norm[0] * width), int(left_index_norm[1] * height))
                right_index_px = (int(right_index_norm[0] * width), int(right_index_norm[1] * height))

                left_heel_norm = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                right_heel_norm = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

                vertical_up_from_shoulders_norm = [mid_shoulders_norm[0],
                                                   mid_shoulders_norm[1] - 0.1]

                angle_neck, neck_vec1_rad, neck_vec2_rad = calculate_angle_with_vectors(ear_norm, mid_shoulders_norm,
                                                                                        vertical_up_from_shoulders_norm)
                is_neck_twisted = abs(left_ear_norm[0] - right_ear_norm[0]) > 0.03
                is_neck_laterally_flexed = abs(left_ear_norm[1] - right_ear_norm[1]) > 0.03
                angle_info.append(f"Cuello: {angle_neck:.1f}")

                angle_trunk, trunk_vec1_rad, trunk_vec2_rad = calculate_angle_with_vectors(
                    vertical_up_from_shoulders_norm, mid_shoulders_norm, mid_hips_norm)
                is_trunk_twisted = abs(left_shoulder_norm[0] - right_shoulder_norm[0]) > 0.05
                is_trunk_laterally_flexed = abs(left_shoulder_norm[1] - right_shoulder_norm[1]) > 0.05
                angle_info.append(f"Tronco: {angle_trunk:.1f}")

                angle_upper_arm_left, ul_arm_l_vec1_rad, ul_arm_l_vec2_rad = calculate_angle_with_vectors(left_hip_norm,
                                                                                                          left_shoulder_norm,
                                                                                                          left_elbow_norm)
                angle_upper_arm_right, ul_arm_r_vec1_rad, ul_arm_r_vec2_rad = calculate_angle_with_vectors(
                    right_hip_norm, right_shoulder_norm, right_elbow_norm)
                is_left_shoulder_raised = left_shoulder_norm[1] < mid_shoulders_norm_y - 0.05
                is_right_shoulder_raised = right_shoulder_norm[1] < mid_shoulders_norm_y - 0.05
                is_left_arm_abducted = abs(left_shoulder_norm[0] - left_hip_norm[0]) > 0.08
                is_right_arm_abducted = abs(right_shoulder_norm[0] - right_hip_norm[0]) > 0.08
                is_left_arm_supported = False
                is_right_arm_supported = False
                angle_info.append(f"Brazo Sup. Izq: {angle_upper_arm_left:.1f}")
                angle_info.append(f"Brazo Sup. Der: {angle_upper_arm_right:.1f}")

                angle_forearm_left, fa_l_vec1_rad, fa_l_vec2_rad = calculate_angle_with_vectors(left_shoulder_norm,
                                                                                                left_elbow_norm,
                                                                                                left_wrist_norm)
                angle_forearm_right, fa_r_vec1_rad, fa_r_vec2_rad = calculate_angle_with_vectors(right_shoulder_norm,
                                                                                                 right_elbow_norm,
                                                                                                 right_wrist_norm)
                angle_info.append(f"Antebrazo Izq: {angle_forearm_left:.1f}")
                angle_info.append(f"Antebrazo Der: {angle_forearm_right:.1f}")

                angle_wrist_left, w_l_vec1_rad, w_l_vec2_rad = calculate_angle_with_vectors(left_elbow_norm,
                                                                                            left_wrist_norm,
                                                                                            left_index_norm)
                angle_wrist_right, w_r_vec1_rad, w_r_vec2_rad = calculate_angle_with_vectors(right_elbow_norm,
                                                                                             right_wrist_norm,
                                                                                             right_index_norm)
                is_left_wrist_deviated = False
                is_right_wrist_deviated = False
                is_left_wrist_twisted = False
                is_right_wrist_twisted = False
                angle_info.append(f"Muneca Izq: {angle_wrist_left:.1f} ")
                angle_info.append(f"Muneca Der: {angle_wrist_right:.1f}")

                min_vis_threshold = 0.2

                angle_hip_left, hip_l_vec1_rad, hip_l_vec2_rad = 0.0, 0.0, 0.0
                angle_hip_right, hip_r_vec1_rad, hip_r_vec2_rad = 0.0, 0.0, 0.0
                angle_knee_left, knee_l_vec1_rad, knee_l_vec2_rad = 0.0, 0.0, 0.0
                angle_knee_right, knee_r_vec1_rad, knee_r_vec2_rad = 0.0, 0.0, 0.0

                leg_left_visible_ok = False
                leg_right_visible_ok = False

                if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > min_vis_threshold):
                    angle_hip_left, hip_l_vec1_rad, hip_l_vec2_rad = calculate_angle_with_vectors(left_shoulder_norm,
                                                                                                  left_hip_norm,
                                                                                                  left_knee_norm)
                    leg_left_visible_ok = True

                if (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > min_vis_threshold):
                    angle_hip_right, hip_r_vec1_rad, hip_r_vec2_rad = calculate_angle_with_vectors(right_shoulder_norm,
                                                                                                   right_hip_norm,
                                                                                                   right_knee_norm)
                    leg_right_visible_ok = True

                if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > min_vis_threshold):
                    angle_knee_left, knee_l_vec1_rad, knee_l_vec2_rad = calculate_angle_with_vectors(left_hip_norm,
                                                                                                     left_knee_norm,
                                                                                                     left_ankle_norm)
                    leg_left_visible_ok = leg_left_visible_ok and True
                else:
                    leg_left_visible_ok = False

                if (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > min_vis_threshold and
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > min_vis_threshold):
                    angle_knee_right, knee_r_vec1_rad, knee_r_vec2_rad = calculate_angle_with_vectors(right_hip_norm,
                                                                                                      right_knee_norm,
                                                                                                      right_ankle_norm)
                    leg_right_visible_ok = leg_right_visible_ok and True
                else:
                    leg_right_visible_ok = False

                angle_info.append(f"Cadera Izq: {angle_hip_left:.1f}°")
                angle_info.append(f"Rodilla Izq: {angle_knee_left:.1f}°")
                angle_info.append(f"Cadera Der: {angle_hip_right:.1f}°")
                angle_info.append(f"Rodilla Der: {angle_knee_right:.1f}°")

                is_bilateral_support = True
                is_sitting = False
                is_unstable_support = False

                heel_vis_threshold = 0.4
                if (landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].visibility < heel_vis_threshold or
                        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].visibility < heel_vis_threshold or
                        abs(left_ankle_norm[1] - right_ankle_norm[1]) > 0.08):
                    is_bilateral_support = False

                neck_score = get_neck_score_reba(angle_neck, is_twisted=is_neck_twisted,
                                                 is_laterally_flexed=is_neck_laterally_flexed)
                trunk_score = get_trunk_score_reba(angle_trunk, is_twisted=is_trunk_twisted,
                                                   is_laterally_flexed=is_trunk_laterally_flexed)

                leg_score_left = get_leg_score_reba(angle_hip_left, is_bilateral_support, is_sitting,
                                                    is_unstable_support, leg_left_visible_ok)
                leg_score_right = get_leg_score_reba(angle_hip_right, is_bilateral_support, is_sitting,
                                                     is_unstable_support, leg_right_visible_ok)
                leg_score = max(leg_score_left, leg_score_right)

                upper_arm_score_left = get_upper_arm_score_reba(angle_upper_arm_left,
                                                                is_shoulder_raised=is_left_shoulder_raised,
                                                                is_arm_abducted=is_left_arm_abducted,
                                                                is_arm_supported=is_left_arm_supported)
                upper_arm_score_right = get_upper_arm_score_reba(angle_upper_arm_right,
                                                                 is_shoulder_raised=is_right_shoulder_raised,
                                                                 is_arm_abducted=is_right_arm_abducted,
                                                                 is_arm_supported=is_right_arm_supported)
                upper_arm_score = max(upper_arm_score_left, upper_arm_score_right)

                forearm_score_left = get_forearm_score_reba(angle_forearm_left)
                forearm_score_right = get_forearm_score_reba(angle_forearm_right)
                forearm_score = max(forearm_score_left, forearm_score_right)

                wrist_score_left = get_wrist_score_reba(angle_wrist_left, is_deviated=is_left_wrist_deviated,
                                                        is_twisted=is_left_wrist_twisted)
                wrist_score_right = get_wrist_score_reba(angle_wrist_right, is_deviated=is_right_wrist_deviated,
                                                         is_twisted=is_right_wrist_twisted)
                wrist_score = max(wrist_score_left, wrist_score_right)

                load_score = 1
                coupling_score = 0
                activity_score = 1

                score_a = get_score_a(trunk_score, neck_score, leg_score)
                score_b = get_score_b(upper_arm_score, forearm_score, wrist_score)

                score_c_final = get_score_c(score_a, score_b, load_score, coupling_score)

                reba_score_value = get_reba_final_score(score_c_final, activity_score)
                reba_risk_display = get_reba_risk_level(reba_score_value)
                reba_score_display = f'REBA: {int(reba_score_value)}'

                debug_info = (f"Vis Hombro Izq: {landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility:.2f}, "
                              f"Der: {landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility:.2f}\n"
                              f"Vis Cadera Izq: {landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility:.2f}, "
                              f"Der: {landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility:.2f}\n"
                              f"Vis Rodilla Izq: {landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility:.2f}, "
                              f"Der: {landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility:.2f}\n"
                              f"Vis Tobillo Izq: {landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility:.2f}, "
                              f"Der: {landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility:.2f}\n"
                              f"Pierna Izq OK: {leg_left_visible_ok}, Pierna Der OK: {leg_right_visible_ok}\n"
                              f"Soporte Bilateral: {is_bilateral_support}\n"
                              f"Load: {load_score}, Coupling: {coupling_score}, Activity: {activity_score}")

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                arc_radius_body = 50
                arc_radius_limbs = 30

                cv2.line(frame, mid_shoulders_px, ear_px, (255, 165, 0), 3)
                cv2.circle(frame, mid_shoulders_px, 5, (255, 165, 0), -1)
                draw_interior_arc(frame, mid_shoulders_px, angle_neck, neck_vec1_rad, neck_vec2_rad, arc_radius_limbs,
                                  (255, 165, 0), 2)
                cv2.putText(frame, f'Cuello: {neck_score}', (mid_shoulders_px[0] + 20, mid_shoulders_px[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, cv2.LINE_AA)

                cv2.line(frame, mid_shoulders_px, mid_hips_px, (0, 255, 0), 3)
                cv2.circle(frame, mid_hips_px, 5, (0, 255, 0), -1)
                vertical_up_px = (int(vertical_up_from_shoulders_norm[0] * width),
                                  int(vertical_up_from_shoulders_norm[1] * height))
                cv2.line(frame, mid_shoulders_px, vertical_up_px, (0, 255, 0), 2, cv2.LINE_AA)
                draw_interior_arc(frame, mid_shoulders_px, angle_trunk, trunk_vec1_rad, trunk_vec2_rad, arc_radius_body,
                                  (0, 255, 0), 2)
                cv2.putText(frame, f'Tronco: {trunk_score}', (mid_hips_px[0] + 20, mid_hips_px[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.line(frame, left_shoulder_px, left_elbow_px, (255, 0, 0), 3)
                draw_interior_arc(frame, left_shoulder_px, angle_upper_arm_left, ul_arm_l_vec1_rad, ul_arm_l_vec2_rad,
                                  arc_radius_limbs, (255, 0, 0), 2)
                cv2.putText(frame, f'Braz. Izq: {upper_arm_score_left}',
                            (left_shoulder_px[0] + 20, left_shoulder_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.line(frame, right_shoulder_px, right_elbow_px, (255, 0, 0), 3)
                draw_interior_arc(frame, right_shoulder_px, angle_upper_arm_right, ul_arm_r_vec1_rad, ul_arm_r_vec2_rad,
                                  arc_radius_limbs, (255, 0, 0), 2)
                cv2.putText(frame, f'Braz. Der: {upper_arm_score_right}',
                            (right_shoulder_px[0] - 120, right_shoulder_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.line(frame, left_elbow_px, left_wrist_px, (0, 0, 255), 3)
                draw_interior_arc(frame, left_elbow_px, angle_forearm_left, fa_l_vec1_rad, fa_l_vec2_rad,
                                  arc_radius_limbs, (0, 0, 255), 2)
                cv2.putText(frame, f'Ant. Izq: {forearm_score_left}', (left_elbow_px[0] + 20, left_elbow_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.line(frame, right_elbow_px, right_wrist_px, (0, 0, 255), 3)
                draw_interior_arc(frame, right_elbow_px, angle_forearm_right, fa_r_vec1_rad, fa_r_vec2_rad,
                                  arc_radius_limbs, (0, 0, 255), 2)
                cv2.putText(frame, f'Ant. Der: {forearm_score_right}', (right_elbow_px[0] - 120, right_elbow_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.line(frame, left_wrist_px, left_index_px, (128, 0, 128), 3)
                draw_interior_arc(frame, left_wrist_px, angle_wrist_left, w_l_vec1_rad, w_l_vec2_rad,
                                  arc_radius_limbs / 2, (128, 0, 128), 2)
                cv2.putText(frame, f'Mun. Izq: {wrist_score_left}', (left_wrist_px[0] + 20, left_wrist_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2, cv2.LINE_AA)

                cv2.line(frame, right_wrist_px, right_index_px, (128, 0, 128), 3)
                draw_interior_arc(frame, right_wrist_px, angle_wrist_right, w_r_vec1_rad, w_r_vec2_rad,
                                  arc_radius_limbs / 2, (128, 0, 128), 2)
                cv2.putText(frame, f'Mun. Der: {wrist_score_right}', (right_wrist_px[0] - 120, right_wrist_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2, cv2.LINE_AA)

                if leg_left_visible_ok:
                    cv2.line(frame, left_hip_px, left_knee_px, (0, 0, 255), 4)
                    cv2.line(frame, left_knee_px, left_ankle_px, (0, 0, 255), 4)
                    draw_interior_arc(frame, left_hip_px, angle_hip_left, hip_l_vec1_rad, hip_l_vec2_rad,
                                      arc_radius_limbs, (0, 0, 255), 2)
                    draw_interior_arc(frame, left_knee_px, angle_knee_left, knee_l_vec1_rad, knee_l_vec2_rad,
                                      arc_radius_limbs, (0, 0, 255), 2)
                    cv2.putText(frame, f'Cad. Izq: {leg_score_left}', (left_hip_px[0] + 20, left_hip_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Rod. Izq: {leg_score_left}', (left_knee_px[0] + 20, left_knee_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                if leg_right_visible_ok:
                    cv2.line(frame, right_hip_px, right_knee_px, (0, 0, 255), 4)
                    cv2.line(frame, right_knee_px, right_ankle_px, (0, 0, 255), 4)
                    draw_interior_arc(frame, right_hip_px, angle_hip_right, hip_r_vec1_rad, hip_r_vec2_rad,
                                      arc_radius_limbs, (0, 0, 255), 2)
                    draw_interior_arc(frame, right_knee_px, angle_knee_right, knee_r_vec1_rad, knee_r_vec2_rad,
                                      arc_radius_limbs, (0, 0, 255), 2)
                    cv2.putText(frame, f'Cad. Der: {leg_score_right}', (right_hip_px[0] - 120, right_hip_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Rod. Der: {leg_score_right}', (right_knee_px[0] - 120, right_knee_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            except Exception as e:
                error_message = f"Error detectado en cálculo/dibujo: {e}"
                print(error_message, file=sys.stderr)
                reba_score_display = "ERROR"
                reba_risk_display = "ERROR"
                debug_info = f"ERROR: {e}"
                cv2.putText(frame, f'ERROR: {e}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, pose_detected_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, reba_score_display, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f'Riesgo: {reba_risk_display}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA)

        y_offset_debug = 150
        for line in debug_info.split('\n'):
            cv2.putText(frame, line, (10, y_offset_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset_debug += 20

        y_offset_angles = y_offset_debug + 20
        cv2.putText(frame, "ANGULOS:", (10, y_offset_angles), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                    cv2.LINE_AA)
        y_offset_angles += 25
        for angle_line in angle_info:
            cv2.putText(frame, angle_line, (10, y_offset_angles), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                        cv2.LINE_AA)
            y_offset_angles += 20

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('REBA Pose Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

sys.stderr = original_stderr
devnull.close()

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
