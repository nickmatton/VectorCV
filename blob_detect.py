import cv2
import numpy as np
import cozmo
from cozmo.util import degrees


def cozmo_program(robot: cozmo.robot.Robot):
    robot.move_lift(-3)
    robot.set_head_angle(degrees(0)).wait_for_completed()
    while True:
        image = robot.world.latest_image.raw_image
        image = np.array(image)
        cv2.imwrite("input.jpg", image)
        img = cv2.imread("input.jpg")
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs.
        keypoints = detector.detect(img)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.imwrite('blobbed.jpg', img)


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)

