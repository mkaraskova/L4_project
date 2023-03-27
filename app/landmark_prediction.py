from util import *
import hickle


def evaluate_landmarks(image, yaw, box):
    ibug_exam, rotated = read_image(image, yaw, box, True)
    ibug_exam = ibug_exam[0]
    ibug_exam_boxes = get_bounding_box(ibug_exam)
    # load the trained regressor
    model = hickle.load("pi_ert_model.hkl")
    init_shapes, fin_shapes = model.apply(ibug_exam, [ibug_exam_boxes[0]])

    landmarks = fin_shapes[0].points
    # rotate the landmarks back
    if rotated:
        landmarks = [[224 - x, y] for x, y in landmarks]
    return landmarks
