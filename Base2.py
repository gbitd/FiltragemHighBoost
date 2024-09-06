import numpy as np
import cv2
from skimage import img_as_float
from skimage.filters import gaussian

def high_boost_filter(image, boost_factor=1.5, sigma=1):
    # Converte a imagem para float
    image_float = img_as_float(image)
    
    # Aplica o filtro gaussiano para obter a imagem borrada
    blurred = gaussian(image_float, sigma=sigma, multichannel=True)
    
    # Calcula a máscara de detalhes
    mask = image_float - blurred
    
    # Aplica a filtragem high boost
    high_boosted = image_float + boost_factor * mask
    
    # Clipa os valores para manter no intervalo [0, 1]
    high_boosted = np.clip(high_boosted, 0, 1)
    
    return high_boosted

# Carrega a imagem usando OpenCV
image = cv2.imread('sua_imagem.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Aplica a filtragem high boost
result = high_boost_filter(image_rgb, boost_factor=1.5, sigma=1)

# Converte a imagem de volta para BGR para salvar com OpenCV
result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite('resultado_high_boost.jpg', result_bgr)

