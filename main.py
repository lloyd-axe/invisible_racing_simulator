import cv2
import math
import configparser
import numpy as np
import mediapipe as mp

from typing import Any
from pynput.keyboard import Controller, Key


class HolisticModel:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

        self.capture = cv2.VideoCapture(0)
        self.holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence
        )

        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.holistic = mp.solutions.holistic
        self.keyboard_controller = Controller()

    def release_resources(self):
        if self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _frame_adjustments(frame: Any, color: int, writeable: bool = True) -> Any:
        frame.flags.writeable = writeable
        return cv2.cvtColor(frame, color)

    def _draw_landmarks(self, frame: Any, results: Any, other_points: list, lines: list,
                        line_color: tuple = (0, 0, 255)) -> Any:
        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
            )
        if results.right_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_hand_connections_style()
            )
        if results.left_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_hand_connections_style()
            )
        for p in other_points:
            cv2.circle(frame, p, 5, (255, 255, 255), cv2.FILLED)
        for line in lines:
            if len(line) == 2:
                cv2.line(frame, line[0], line[1], line_color, 2)
        return frame


class InvisibleRacingSim(HolisticModel):
    def __init__(self, throttle_key: str, break_key: str, steer_left_key: str, steer_right_key: str,
                 steer_threshold: int, delta_angle_threshold: int, break_threshold: int, throttle_threshold: int,
                 show_video: bool = True,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        super().__init__(min_detection_confidence, min_tracking_confidence)
        self.frame_shape = None
        self.show_video = show_video
        self.steer_threshold = steer_threshold
        self.delta_angle_threshold = delta_angle_threshold
        self.break_threshold = break_threshold
        self.throttle_threshold = throttle_threshold
        self.throttle_key = throttle_key
        self.break_key = break_key
        self.steer_left_key = self.non_letter_keys(steer_left_key) if steer_left_key in self.key_dict.keys() \
            else steer_left_key
        self.steer_right_key = self.non_letter_keys(steer_right_key) if steer_right_key in self.key_dict.keys() \
            else steer_right_key

    @property
    def key_dict(self):
        return {
            'left': Key.left,
            'right': Key.right,
            'up': Key.up,
            'down': Key.down,
            'space': Key.space
        }

    def non_letter_keys(self, key: str) -> str:
        return self.key_dict.get(key, key)

    def get_hand_points(self, results: Any) -> tuple:
        pose_landmarks = [[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] \
            if results.pose_landmarks else np.zeros((33, 3))
        if np.array(pose_landmarks).any():
            right_mid = self.midpoint(pose_landmarks[15], pose_landmarks[19])
            left_mid = self.midpoint(pose_landmarks[16], pose_landmarks[20])
            return right_mid, left_mid
        return None, None

    def midpoint(self, point1: list, point2: list) -> tuple:
        x1, y1 = int(point1[0] * self.frame_shape[1]), int(point1[1] * self.frame_shape[0])
        x2, y2 = int(point2[0] * self.frame_shape[1]), int(point2[1] * self.frame_shape[0])
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def angle(right_mid: list, left_mid: list) -> float:
        return math.degrees(math.atan2(left_mid[1] - right_mid[1], left_mid[0] - right_mid[0]))

    @staticmethod
    def distance(right_mid: list, left_mid: list) -> float:
        return math.sqrt((left_mid[0] - right_mid[0]) ** 2 + (left_mid[1] - right_mid[1]) ** 2)

    @staticmethod
    def calc_turn(angle: float) -> tuple:
        state_direction = 1 if angle < 0 else 2
        percent_turn = ((180 - abs(angle)) / 90) * 100
        return percent_turn, state_direction

    def __calculate_values(self, right_mid: list, left_mid: list, prev_perc_turn: float):
        angle = self.angle(right_mid, left_mid)
        distance = self.distance(right_mid, left_mid)
        percent_turn, state_direction = self.calc_turn(angle)
        delta_angle = percent_turn - prev_perc_turn
        state_direction = 0 if percent_turn < self.steer_threshold else state_direction
        return distance, percent_turn, delta_angle, state_direction

    def __steer(self, percent_turn: float, delta_angle: float, state_direction: int, prev_state_direction: int) -> int:
        if percent_turn > self.steer_threshold:
            if state_direction == prev_state_direction:
                if abs(delta_angle) > self.delta_angle_threshold:
                    if delta_angle < 0:
                        steer_now = False
                    else:
                        steer_now = True
                else:
                    steer_now = True
            else:
                steer_now = True
        else:
            steer_now = False

        if steer_now:
            if state_direction == 1:
                self.keyboard_controller.press(self.steer_left_key)
                self.keyboard_controller.release(self.steer_right_key)
            else:
                self.keyboard_controller.press(self.steer_right_key)
                self.keyboard_controller.release(self.steer_left_key)
        else:
            self.keyboard_controller.release(self.steer_left_key)
            self.keyboard_controller.release(self.steer_right_key)
        return state_direction

    def __throttle(self, distance: float) -> int:
        throttle_state = 0
        if distance > self.throttle_threshold:
            self.keyboard_controller.press(self.throttle_key)
            self.keyboard_controller.release(self.break_key)
            throttle_state = 1
        else:
            if distance < self.break_threshold:
                self.keyboard_controller.press(self.break_key)
                self.keyboard_controller.release(self.throttle_key)
                throttle_state = 2
            else:
                self.keyboard_controller.release(self.break_key)
                self.keyboard_controller.release(self.throttle_key)
        return throttle_state

    def run(self):
        prev_perc_turn = 0
        prev_state_direction = 0
        while self.capture.isOpened():
            success, frame = self.capture.read()
            if not success:
                print('Ignoring empty camera frame.')
                continue
            if not self.frame_shape:
                self.frame_shape = frame.shape

            frame = self._frame_adjustments(frame, cv2.COLOR_BGR2RGB, False)
            results = self.holistic_model.process(frame)
            right_mid, left_mid = self.get_hand_points(results)

            frame = self._frame_adjustments(frame, cv2.COLOR_RGB2BGR, True)
            steering_line_color = (0, 0, 0)
            direction = "STRAIGHT"
            if right_mid and left_mid:
                distance, percent_turn, delta_angle, state_direction = \
                    self.__calculate_values(right_mid, left_mid, prev_perc_turn)

                state_direction = self.__steer(percent_turn, delta_angle, state_direction, prev_state_direction)
                throttle_state = self.__throttle(distance)
                if throttle_state == 1:
                    steering_line_color = (0, 255, 0)
                elif throttle_state == 2:
                    steering_line_color = (0, 0, 255)

                if state_direction == 1:
                    direction = "LEFT"
                elif state_direction == 2:
                    direction = "RIGHT"

                prev_perc_turn = percent_turn
                prev_state_direction = state_direction

            frame = self._draw_landmarks(frame, results, [right_mid, left_mid], [[right_mid, left_mid]],
                                         steering_line_color)

            resized_frame = cv2.resize(frame, (330, 300))
            flipped_frame = cv2.flip(resized_frame, 1)
            cv2.putText(flipped_frame, direction,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
                        )
            if self.show_video:
                cv2.imshow('Invisible Steering Wheel Simulator', flipped_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.release_resources()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    throttle_key = config["keys"]["throttle"]
    break_key = config["keys"]["break"]
    steer_left_key = config["keys"]["steer_left"]
    steer_right_key = config["keys"]["steer_right"]
    steer_threshold = int(config["sensitivity"]["steer_threshold"])
    delta_angle_threshold = int(config["sensitivity"]["delta_angle_threshold"])
    break_threshold = int(config["sensitivity"]["break_threshold"])
    throttle_threshold = int(config["sensitivity"]["throttle_threshold"])
    show_video = bool(config["video"]["show_video"])

    inst = InvisibleRacingSim(throttle_key=throttle_key, break_key=break_key,
                              steer_left_key=steer_left_key, steer_right_key=steer_right_key,
                              steer_threshold=steer_threshold, delta_angle_threshold=delta_angle_threshold,
                              break_threshold=break_threshold, throttle_threshold=throttle_threshold,
                              show_video=show_video)
    inst.run()
