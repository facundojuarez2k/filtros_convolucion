import sys
import os
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
from filters import filters
import argparse
from scipy import fftpack


def main():
    # Leer entrada del programa
    (image_path, filter_name, no_fft) = parse_args()

    if not os.path.exists(image_path):
        sys.exit("Ruta de imagen inválida")

    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        sys.exit("Imagen no válida")

    # Cargar filtro
    filter = filters.get(filter_name, None)
    if filter is None:
        sys.exit("Filtro inexistente")

    # Aplicar el filtro
    if no_fft:
        filtered_image = convolve(image, filter['kernel'], filter['grayscale'])
    else:
        filtered_image = convolve_fft(image, filter['kernel'], filter['grayscale'])

    render_image(filtered_image)

    sys.exit(0)


def render_image(image):
    '''
        Mostrar la imagen en una ventana
    '''
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolve_fft(image, kernel, grayscale=False):
    '''
        Calcula la convolución image * kernel usando el algoritmo FFT
    '''
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invertir matriz sobre eje x y sobre eje y
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)

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

        # Normalizar pixels
        filtered_c = rescale_intensity(filtered_c, in_range=(0, 255))
        filtered_c = filtered_c * 255
        filtered_channels.append(filtered_c)

    # Unir los canales con el filtro aplicado
    output = cv2.merge(filtered_channels)
    output = output.astype("uint8")

    return output


def convolve(image, kernel, grayscale=False):
    '''
        Calcula la convolución image * kernel en el dominio espacial
    '''
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invertir matriz sobre eje x y sobre eje y
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)

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

    # Separar canales de la imagen
    channels = cv2.split(image)
    filtered_channels = []

    # Recorrer la imagen de izquierda a derecha y desde arriba hacia abajo aplicando el kernel a cada pixel
    for chnl in channels:
        # Crear matriz nula
        filtered_c = np.zeros((image_height, image_width))

        for y in np.arange(v_pad, image_height + v_pad):
            for x in np.arange(h_pad, image_width + h_pad):
                # Obtener la región de la imagen sobre la cual se aplicara la convolución
                y_start = (y - v_pad)
                y_end = (y + v_pad + 1)
                x_start = (x - h_pad)
                x_end = (x + h_pad + 1)

                roi = chnl[y_start:y_end, x_start:x_end]

                # Aplicar la convolución
                filtered_c[y-v_pad, x-h_pad] = (roi * kernel).sum()

        # Normalizar pixels
        filtered_c = rescale_intensity(filtered_c, in_range=(0, 255))
        filtered_c = filtered_c * 255
        filtered_channels.append(filtered_c)

    output = cv2.merge(filtered_channels)
    output = output.astype("uint8")

    return output


def parse_args():
    filter_options = 'Filtros disponibles: \n'

    for k, v in filters.items():
        filter_options += f' {k}: {v["description"]} \n'

    parser = argparse.ArgumentParser(
        description="Aplica el filtro especificado a la imagen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'{filter_options}')
    parser.add_argument("image_path", metavar="image",
                        type=str, help="Ruta a la imagen")
    parser.add_argument("filter_name", metavar="filter",
                        type=str, help="Nombre del filtro a aplicar")
    parser.add_argument('--no-fft', action='store_true', help="Ejecuta la convolución de forma directa")

    args = parser.parse_args()

    return (args.image_path, args.filter_name, args.no_fft)


if __name__ == '__main__':
    main()
