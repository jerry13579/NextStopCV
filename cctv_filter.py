import cv2
import numpy as np


def camera_source_arg(args, default="0"):
    return next((arg for arg in args[1:] if not arg.startswith("--")), default)


def filters_enabled(args):
    return "--raw" not in args and "--no-filter" not in args


def apply_gaussian_noise(image, sigma=11):
    return np.clip(image.astype(np.int16) + np.random.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def apply_cctv_filter(image):
    image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    image = cv2.convertScaleAbs(image, alpha=1.35, beta=-18)
    image = cv2.addWeighted(image, 0.86, cv2.GaussianBlur(image, (0, 0), 1.4), 0.14, 0)
    image[::3] = (image[::3] * 0.72).astype(np.uint8)
    image[:, :, 1] = np.clip(image[:, :, 1] * 1.08 + 8, 0, 255)
    image = apply_gaussian_noise(image)
    h, w = image.shape[:2]
    vignette = cv2.GaussianBlur(np.pad(np.ones((h - 80, w - 80), dtype=np.float32), ((40, 40), (40, 40))), (0, 0), 45)
    image = (image * (0.58 + 0.42 * vignette[:, :, None])).astype(np.uint8)
    cv2.putText(image, "CAM 04  REC", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
    cv2.circle(image, (118, 20), 5, (0, 0, 255), -1)
    return image


def apply_demo_filters(image, enabled=True):
    return apply_cctv_filter(image) if enabled else image
