import dataclasses
import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(
    color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad
)
right_eye_draw = DrawingSpec(
    color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad
)
right_eyebrow_draw = DrawingSpec(
    color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad
)
left_iris_draw = DrawingSpec(
    color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad
)
left_eye_draw = DrawingSpec(
    color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad
)
left_eyebrow_draw = DrawingSpec(
    color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad
)
mouth_draw = DrawingSpec(
    color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad
)
head_draw = DrawingSpec(
    color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad
)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw

# for edge in mp_face_mesh.FACEMESH_LIPS:
#     face_connection_spec[edge] = mouth_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
#     face_connection_spec[edge] = left_eye_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
#     face_connection_spec[edge] = left_eyebrow_draw
for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
    face_connection_spec[edge] = left_iris_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
#     face_connection_spec[edge] = right_eye_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
#     face_connection_spec[edge] = right_eyebrow_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
    face_connection_spec[edge] = right_iris_draw


def _as_cv_point(point, width, height):
    point = np.asarray(point, dtype=np.float32).reshape(-1)
    if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
        return None

    x = int(np.clip(np.rint(point[0]), 0, width - 1))
    y = int(np.clip(np.rint(point[1]), 0, height - 1))
    return (x, y)


def draw_landmarks_468(
        image: np.ndarray,
        landmark_list: np.ndarray,
        connections: Optional[List[Tuple[int, int]]] = face_connection_spec.keys(),
        connection_drawing_spec: Union[
            DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
        ] = face_connection_spec,
):
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    idx_to_coordinates = {
        i: _as_cv_point(landmark_list[i], width, height)
        for i in range(len(landmark_list))
    }
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            pt1 = idx_to_coordinates[start_idx]
            pt2 = idx_to_coordinates[end_idx]
            if pt1 is None or pt2 is None:
                continue
            drawing_spec = (
                connection_drawing_spec[connection]
                if isinstance(connection_drawing_spec, Mapping)
                else connection_drawing_spec
            )
            cv2.line(
                image,
                pt1,
                pt2,
                drawing_spec.color,
                drawing_spec.thickness,
            )

    # ``head_v2`` appends the 468-map iris points as [left, right].  Preserve
    # that order here; the previous reverse lookup colored the two eyes with
    # one another's semantic label even after their geometry was projected
    # correctly.
    left_iris = idx_to_coordinates.get(len(landmark_list) - 2)
    right_iris = idx_to_coordinates.get(len(landmark_list) - 1)
    if left_iris is not None:
        cv2.circle(image, left_iris, left_eye_draw.circle_radius, left_eye_draw.color, thickness=-1)
    if right_iris is not None:
        cv2.circle(image, right_iris, right_eye_draw.circle_radius, right_eye_draw.color, thickness=-1)
    return image[:, :, ::-1]  # flip BGR
