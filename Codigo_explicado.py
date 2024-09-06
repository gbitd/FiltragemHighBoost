import cv2  # Biblioteca para manipulação de imagens
import numpy as np  # Biblioteca para operações numéricas
from skimage.filters import rank
from skimage.morphology import square

def aplicar_filtro_media(image, kernel_size=3):
   
    output_image = rank.mean(image, square(kernel_size))
    
    return output_image

def subtrair_imagem_borrada(original_image, blurred_image):
    """
    Subtrai a imagem borrada da original para gerar a máscara de nitidez.
    
    Parâmetros:
    - original_image: Imagem original sem suavização.
    - blurred_image: Imagem suavizada (borrada).
    
    Retorno:
    - Máscara de nitidez.
    """
    # Subtração simples da imagem borrada da imagem original
    sharpness_mask = original_image - blurred_image
    return sharpness_mask

def adicionar_mascara_imagem(original_image, sharpness_mask, k=1.0):
    """
    Adiciona a máscara de nitidez à imagem original, aplicando o filtro high-boost.
    
    Parâmetros:
    - original_image: Imagem original.
    - sharpness_mask: Máscara de nitidez obtida pela subtração da imagem borrada.
    - k: Fator de amplificação da máscara de nitidez (ajustável).
    
    Retorno:
    - Imagem filtrada (high-boost) com a máscara de nitidez aplicada.
    """
    # Adiciona a máscara amplificada pela constante k à imagem original
    filtered_image = original_image + k * sharpness_mask
    
    # Clipa os valores para garantir que estejam no intervalo [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255)
    
    # Converte a imagem para o formato uint8 (valores inteiros entre 0 e 255)
    return filtered_image.astype(np.uint8)

def main():
    # Carregar a imagem em escala de cinza
    image = cv2.imread('celebro.tif', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        # Verifica se houve erro ao carregar a imagem
        print("Erro ao carregar a imagem.")
        return

    # Converte a imagem para float para permitir operações com precisão
    image_float = image.astype(float)

    # Define o tamanho do kernel para o filtro de suavização
    kernel_size = 5
    
    # Aplica o filtro de média à imagem
    blurred_image = aplicar_filtro_media(image_float, kernel_size)

    # Calcula a máscara de nitidez subtraindo a imagem borrada da original
    sharpness_mask = subtrair_imagem_borrada(image_float, blurred_image)

    # Fator de amplificação da máscara de nitidez (ajustável)
    k = 1.5
    
    # Aplica o filtro high-boost adicionando a máscara à imagem original
    filtered_image = adicionar_mascara_imagem(image_float, sharpness_mask, k)

    # Mostra as imagens (original, borrada, máscara de nitidez e imagem filtrada)
    cv2.imshow("Imagem Original", image)
    cv2.imshow("Imagem Borrada", blurred_image.astype(np.uint8))
    cv2.imshow("Máscara de Nitidez", sharpness_mask.astype(np.uint8))
    cv2.imshow("Imagem Filtrada (High-Boost)", filtered_image)

    # Espera o usuário fechar as janelas de visualização
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Executa o programa principal se este arquivo for executado diretamente
if __name__ == "__main__":
    main()
