# python -m pip install opencv-python
import cv2
from dataclasses import dataclass

@dataclass
class CheckResult:
    name : str
    passed : bool
    value : float
    reason : str

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'could not read image:{path}')
    return image

def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    return float(laplacian.var())

def check_blur(image,threshold = 100.0):
    score = blur_score(image)
    passed = score >= threshold
    if passed:
        reason = 'sharp'
    else:
        reason = f'blurry (score {score:.1f} < {threshold})'
    return CheckResult(name='blur',passed=passed,value=score,reason=reason)


