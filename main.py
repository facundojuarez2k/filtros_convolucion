import numpy as np
from numpy import ndarray
import cv2
from cv2 import Mat


def main():
    # Cargar imagen original
    image = cv2.imread('./source.jpg')
    image_gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    small_blur = np.ones(shape=(7, 7), dtype="float") * (1.0 / (7 * 7))

    # Aplicar el filtro
    result_image = convolve(image, small_blur)

    # Mostrar imágenes
    cv2.namedWindow("original_image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("original_image", image_gray_scale)
    cv2.imshow("original_image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolve(image: Mat, kernel: ndarray) -> Mat:
    # Obtener las dimensiones de la imagen y del kernel
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]

    # Padding para los bordes horizontales y verticales de la imagen
    h_pad = (kernel_width - 1) // 2
    v_pad = (kernel_height - 1) // 2

    # Agregar padding a los bordes de la imagen
    image = cv2.copyMakeBorder(src=image, top=v_pad, bottom=v_pad,
                               left=h_pad, right=h_pad, borderType=cv2.BORDER_REPLICATE)

    # Crear matriz nula
    output = np.zeros((image_height, image_width), dtype="float32")

    # Recorrer la imagen de izquierda a derecha y desde arriba hacia abajo aplicando el kernel a cada pixel
    for y in np.arange(v_pad, image_height + v_pad):
        for x in np.arange(v_pad, image_width + v_pad):
            # Obtener la región de la imagen sobre la cual se aplicara la convolución
            y_start = (y - v_pad)
            y_end = (y + v_pad + 1)
            x_start = (x - h_pad)
            x_end = (x + h_pad + 1)

            roi = image[y_start:y_end, x_start:x_end]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y, x] = k

    normalized_image = cv2.normalize(output,  output, 0, 255, cv2.NORM_MINMAX)

    return normalized_image


if __name__ == '__main__':
    main()
