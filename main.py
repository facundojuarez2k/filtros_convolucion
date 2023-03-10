import sys
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import kernels
from scipy import fftpack


def main():
    # Cargar imagen original
    image = cv2.imread('./source.jpg')

    if image is None or image.size == 0:
        sys.exit("Imagen no válida")

    # Aplicar el filtro
    filtered_image = convolve_fft(image, kernels.large_blur)

    render_image(filtered_image)

    sys.exit(0)


def render_image(image):
    cv2.namedWindow("original_image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("original_image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolve_fft(image, kernel):
    '''
        Calcula la convolución image * kernel usando el algoritmo FFT
    '''
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]

    size = (image_height - kernel_height, image_width - kernel_width)

    # Agregar padding al kernel
    kernel_padded = np.pad(
        kernel, (((size[0]+1)//2, size[0]//2), ((size[1]+1)//2, size[1]//2)), 'constant')

    # Shifting de la frecuencia cero al centro del espectro
    kernel_padded = fftpack.ifftshift(kernel_padded)

    # Separar canales de la imagen
    channels = cv2.split(image)
    filtered_channels = []

    # Aplicar el filtro a cada canal independiente de la imagen
    for c in channels:
        # Aplicar convolución en el dominio frecuencial
        filtered_c = np.real(fftpack.ifft2(
            fftpack.fft2(c) * fftpack.fft2(kernel_padded)))
        filtered_channels.append(filtered_c)

    # Unir los canales con el filtro aplicado
    output = cv2.merge(filtered_channels)
    output = output.astype("uint8")

    return output


def convolve(image, kernel):
    '''
        Calcula la convolución image * kernel en el dominio espacial
    '''
    # Obtener las dimensiones de la imagen y del kernel
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]
    channels = image.shape[-1] if image.ndim == 3 else 1

    # Padding para los bordes horizontales y verticales de la imagen
    h_pad = (kernel_width - 1) // 2
    v_pad = (kernel_height - 1) // 2

    # Agregar padding a los bordes de la imagen
    image = cv2.copyMakeBorder(src=image, top=v_pad, bottom=v_pad,
                               left=h_pad, right=h_pad, borderType=cv2.BORDER_REPLICATE)

    # Crear matriz nula
    output = np.zeros((image_height, image_width, channels))

   # Recorrer la imagen de izquierda a derecha y desde arriba hacia abajo aplicando el kernel a cada pixel
    for y in np.arange(v_pad, image_height + v_pad):
        for x in np.arange(h_pad, image_width + h_pad):
            # Obtener la región de la imagen sobre la cual se aplicara la convolución
            y_start = (y - v_pad)
            y_end = (y + v_pad + 1)
            x_start = (x - h_pad)
            x_end = (x + h_pad + 1)

            roi = image[y_start:y_end, x_start:x_end]

            # Aplicar la convolución a cada canal
            for channel_index in range(channels):
                # Obtener el canal (r, g, b)
                channel_values = roi[:, :, channel_index]

                # Aplicar la convolución
                k = (channel_values * kernel).sum()

                output[y-v_pad, x-h_pad][channel_index] = k

    # Normalizar pixeles
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


if __name__ == '__main__':
    main()
