import numpy as np
import cv2
from skimage import img_as_float
from skimage.filters import gaussian

def high_boost_filter(image, boost_factor=1.5, sigma=1):
    image_float = img_as_float(image)
    blurred = gaussian(image_float, sigma=sigma)
    
    mask = image_float - blurred
    high_boosted = image_float + boost_factor * mask
    high_boosted = np.clip(high_boosted, 0, 1)
    
    return high_boosted

image = cv2.imread('teste01.tif')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = high_boost_filter(image_rgb, boost_factor=1.5, sigma=1)

result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

cv2.imshow('Imagem Original', image)
cv2.imshow('Resultado High Boost', result_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('resultado_high_boost.jpg', result_bgr)
