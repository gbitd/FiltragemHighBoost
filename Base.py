import numpy as np
import cv2

# Função para gerar uma imagem borrada (suavizada)
def borrar_imagem(image, kernel_size=5, sigma=1):
    """
    Aplica um filtro Gaussiano para borrar (suavizar) a imagem.
    
    Parâmetros:
    - image: Imagem original (array 2D do numpy).
    - kernel_size: Tamanho do kernel Gaussiano.
    - sigma: Desvio padrão do Gaussiano.
    
    Retorno:
    - Imagem borrada (array 2D do numpy).
    """
    # Criar o kernel Gaussiano
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    
    # Aplicar convolução usando o kernel
    imagem_borrada = cv2.filter2D(image, -1, kernel)
    return imagem_borrada

# Função para subtrair a imagem borrada da imagem original (Máscara de nitidez/unsharp masking)
def subtrair_imagem_borrada(image, imagem_borrada):
    """
    Subtrai a imagem borrada da imagem original para criar a máscara de nitidez.
    
    Parâmetros:
    - image: Imagem original (array 2D do numpy).
    - imagem_borrada: Imagem borrada (array 2D do numpy).
    
    Retorno:
    - Máscara de nitidez (array 2D do numpy).
    """
    mascara = image - imagem_borrada
    return mascara

# Função para adicionar a máscara à imagem original (filtragem high-boost)
def adicionar_mascara_imagem(image, mascara, k=1.0):
    """
    Adiciona a máscara à imagem original com um fator de ponderação k.
    
    Parâmetros:
    - image: Imagem original (array 2D do numpy).
    - mascara: Máscara de nitidez (array 2D do numpy).
    - k: Fator de ponderação para a máscara.
    
    Retorno:
    - Imagem final com filtragem high-boost aplicada (array 2D do numpy).
    """
    imagem_high_boost = cv2.addWeighted(image, 1.0, mascara, k, 0)
    return imagem_high_boost

# Função principal para executar o processo de filtragem high-boost
def aplicar_filtragem_high_boost(image_path, kernel_size=5, sigma=1, k_values=[1.0, 2.0, 4.5]):
    """
    Aplica o processo completo de filtragem high-boost em uma imagem.
    
    Parâmetros:
    - image_path: Caminho para a imagem a ser processada.
    - kernel_size: Tamanho do kernel Gaussiano.
    - sigma: Desvio padrão do Gaussiano.
    - k_values: Lista de valores de k para experimentar.
    """
    # Carregar a imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")
    
    # Aplicar o borramento
    imagem_borrada = borrar_imagem(image, kernel_size, sigma)
    
    # Criar a máscara de nitidez
    mascara = subtrair_imagem_borrada(image, imagem_borrada)
    
    # Aplicar a filtragem high-boost para diferentes valores de k
    for k in k_values:
        imagem_high_boost = adicionar_mascara_imagem(image, mascara, k)
        
        # Exibir a imagem resultante usando OpenCV
        cv2.imshow(f'Imagem High-Boost com k={k}', imagem_high_boost)
    
    # Aguardar até que qualquer tecla seja pressionada para fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Caminho para a imagem de entrada
image_path = '\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0340(a)(dipxe_text)'

# Aplicar a filtragem high-boost
aplicar_filtragem_high_boost(image_path)


