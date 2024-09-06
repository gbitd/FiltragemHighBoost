import cv2
import numpy as np
from skimage.filters import rank
from skimage.morphology import disk

# Função para aplicar um filtro de média utilizando scikit-image (sem loops)
def aplicar_filtro_media(image, kernel_size=3):
    """
    Aplica um filtro de média para suavizar (borrar) a imagem utilizando scikit-image.
    
    Parâmetros:
    - image: Imagem original (array 2D do numpy).
    - kernel_size: Tamanho do kernel de suavização (deve ser ímpar).
    
    Retorno:
    - Imagem suavizada (array 2D do numpy).
    """
    # Garantir que a imagem está no formato uint8, pois o rank.mean funciona com uint8
    if image.dtype != np.uint8:
        # Normalizar a imagem para que os valores fiquem entre 0 e 255
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

    # Aplicar o filtro de média usando a função rank.mean da scikit-image
    footprint = disk(kernel_size // 2)  # Criar elemento estruturante do tamanho do kernel
    smoothed_image = rank.mean(image, footprint=footprint)  # Filtro de média
    
    return smoothed_image

# Função para subtrair a imagem borrada da imagem original
def subtrair_imagem_borrada(original_image, blurred_image):
    """
    Subtrai a imagem borrada da original para gerar uma máscara de nitidez.
    
    Parâmetros:
    - original_image: Imagem original (array 2D do numpy).
    - blurred_image: Imagem suavizada (array 2D do numpy).
    
    Retorno:
    - Máscara de nitidez (array 2D do numpy).
    """
    sharpness_mask = original_image - blurred_image
    return sharpness_mask

# Função para adicionar a máscara de nitidez à imagem original
def adicionar_mascara_imagem(original_image, sharpness_mask, k=1.0):
    """
    Adiciona a máscara de nitidez à imagem original para aplicar a filtragem high-boost.
    
    Parâmetros:
    - original_image: Imagem original (array 2D do numpy).
    - sharpness_mask: Máscara de nitidez gerada pela subtração da imagem borrada.
    - k: Fator de amplificação da máscara. Valores maiores que 1 aumentam o efeito de realce.
    
    Retorno:
    - Imagem filtrada (array 2D do numpy).
    """
    filtered_image = original_image + k * sharpness_mask
    return filtered_image

# Função principal que executa o pipeline de filtragem
def main():
    # Carregar a imagem .tif em escala de cinza
    image = cv2.imread('teste01.tif', cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Verificar se a imagem não está no formato uint8
    if image.dtype != np.uint8:
        # Se a imagem tiver mais de 8 bits por canal, converte para uint8
        print(f"Convertendo imagem de {image.dtype} para uint8...")
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

    # Aplicar o filtro de suavização com scikit-image
    kernel_size = 5
    blurred_image = aplicar_filtro_media(image, kernel_size)

    # Subtrair a imagem borrada da original para criar a máscara de nitidez
    sharpness_mask = subtrair_imagem_borrada(image, blurred_image)

    # Adicionar a máscara à imagem original para aplicar a filtragem high-boost
    k = 1.5  # Fator de amplificação ajustável
    filtered_image = adicionar_mascara_imagem(image, sharpness_mask, k)

    # Mostrar os resultados
    cv2.imshow("Imagem Original", image)
    cv2.imshow("Imagem Borrada", blurred_image)
    cv2.imshow("Máscara de Nitidez", sharpness_mask)
    cv2.imshow("Imagem Filtrada (High-Boost)", filtered_image)

    # Aguardar o usuário pressionar uma tecla para fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
