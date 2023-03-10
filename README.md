## Descripción

Aplica el filtro especificado a la imagen utilizando la operación convolución entre la matriz de la imagen y el kernel asociado al filtro

## Requisitos

-   [Python3](https://www.python.org/downloads/)

## Instalar módulos

cd filtros_convolucion

python -m pip install -r requirements.txt

## Uso

**python main.py <RUTA_IMAGEN> <FILTRO>**

e.g: `python main.py /home/user/imagen.png small_blur`

### Filtros disponibles

| Nombre     | Descripción             |
| ---------- | ----------------------- |
| identity   | Filtro identidad        |
| sharpen    | Realza los bordes       |
| small_blur | Blur gaussiano leve     |
| large_blur | Blur gaussiano grande   |
| laplacian  | Filtro laplaciano       |
| sobel_x    | Filtro sobel horizontal |
| sobel_x    | Filtro sobel vertical   |

## Creditos
- Adrian Rosebrock: https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
- Cris Luengo: https://stackoverflow.com/questions/54877892/convolving-image-with-kernel-in-fourier-domain
