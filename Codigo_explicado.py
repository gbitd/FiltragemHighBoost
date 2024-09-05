import cv2
import numpy as np

# Função para aplicar o filtro de média com fatiamento (sem loops explícitos)
def aplicar_filtro_media(image, kernel_size=3):
    """
    Aplica um filtro de média para suavizar (borrar) a imagem usando fatiamento.
    
    Parâmetros:
    - image: Imagem original (array 2D do numpy).
    - kernel_size: Tamanho do kernel de suavização (deve ser ímpar).
    
    Retorno:
    - Imagem suavizada (array 2D do numpy).
    """
    # Criar o kernel de média
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    
    # Padding da imagem
    pad_h, pad_w = kernel_size // 2, kernel_size // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Aplicar o filtro com fatiamento (convolução)
    output_image = np.zeros_like(image)
    
    # Usando fatiamento e somatório para aplicar o kernel de uma vez em blocos
    for i in range(kernel_size):
        for j in range(kernel_size):
            output_image += padded_image[i:i + image.shape[0], j:j + image.shape[1]] * kernel[i, j]

    return output_image

# Função para subtrair a imagem borrada da original (sem alterações, pois já é eficiente)
def subtrair_imagem_borrada(original_image, blurred_image):
    """
    Subtrai a imagem borrada da imagem original para gerar a máscara de nitidez.
    
    Parâmetros:
    - original_image: Imagem original (array 2D do numpy).
    - blurred_image: Imagem borrada (array 2D do numpy).
    
    Retorno:
    - Máscara de nitidez (array 2D do numpy).
    """
    sharpness_mask = original_image - blurred_image
    return sharpness_mask

# Função para adicionar a máscara de nitidez à imagem original (sem alterações, pois já é eficiente)
def adicionar_mascara_imagem(original_image, sharpness_mask, k=1.0):
    """
    Adiciona a máscara de nitidez à imagem original para aplicar a filtragem high-boost.
    
    Parâmetros:
    - original_image: Imagem original (array 2D do numpy).
    - sharpness_mask: Máscara de nitidez (array 2D do numpy), gerada pela subtração da imagem borrada.
    - k: Fator de amplificação da máscara (1.0 é a filtragem normal, >1.0 é a filtragem high-boost).
    
    Retorno:
    - Imagem filtrada (array 2D do numpy).
    """
    filtered_image = original_image + k * sharpness_mask
    return filtered_image

# Função principal para executar o pipeline de filtragem
def main():
    # Carregar a imagem em escala de cinza
    image = cv2.imread('teste02.tif', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Aplicar o filtro de suavização com fatiamento
    kernel_size = 5
    blurred_image = aplicar_filtro_media(image, kernel_size)

    # Subtrair a imagem borrada da original para criar a máscara de nitidez
    sharpness_mask = subtrair_imagem_borrada(image, blurred_image)

    # Adicionar a máscara à imagem original para aplicar a filtragem high-boost
    k = 1.5  # fator de amplificação (ajustável)
    filtered_image = adicionar_mascara_imagem(image, sharpness_mask, k)

    # Mostrar os resultados
    cv2.imshow("Imagem Original", image)
    cv2.imshow("Imagem Borrada", blurred_image)
    cv2.imshow("Máscara de Nitidez", sharpness_mask)
    cv2.imshow("Imagem Filtrada (High-Boost)", filtered_image)

    # Aguardar o usuário fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
