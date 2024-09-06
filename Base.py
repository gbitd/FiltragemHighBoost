import cv2
import numpy as np

# Funções previamente definidas (aplicar_filtro_media, subtrair_imagem_borrada, adicionar_mascara_imagem)

def aplicar_filtro_media(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    image_h, image_w = image.shape
    pad_h, pad_w = kernel_size // 2, kernel_size // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output_image = np.zeros_like(image)
    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output_image[i, j] = np.sum(region * kernel)
    return output_image

def subtrair_imagem_borrada(original_image, blurred_image):
    sharpness_mask = original_image - blurred_image
    return sharpness_mask

def adicionar_mascara_imagem(original_image, sharpness_mask, k=1.0):
    filtered_image = original_image + k * sharpness_mask
    return filtered_image

def main():
    # Carregar a imagem em escala de cinza
    image = cv2.imread('placa.tif', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Aplicar o filtro de suavização
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



