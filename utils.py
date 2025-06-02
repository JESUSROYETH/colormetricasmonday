import cv2
import numpy as np


def rgbarray_to_lab(rgb_array):
    # Convertir RGB a XYZ
    rgb_array = rgb_array / 255.0
    mask = rgb_array <= 0.04045
    rgb_array[mask] = rgb_array[mask] / 12.92
    rgb_array[~mask] = ((rgb_array[~mask] + 0.055) / 1.055) ** 2.4

    rgb_array = rgb_array * 100

    x = (
        rgb_array[..., 0] * 0.4124
        + rgb_array[..., 1] * 0.3576
        + rgb_array[..., 2] * 0.1805
    )
    y = (
        rgb_array[..., 0] * 0.2126
        + rgb_array[..., 1] * 0.7152
        + rgb_array[..., 2] * 0.0722
    )
    z = (
        rgb_array[..., 0] * 0.0193
        + rgb_array[..., 1] * 0.1192
        + rgb_array[..., 2] * 0.9505
    )

    # Convertir XYZ a L*a*b*
    x /= 95.047
    y /= 100.000
    z /= 108.883

    mask_x = x > 0.008856
    mask_y = y > 0.008856
    mask_z = z > 0.008856

    x[mask_x] = x[mask_x] ** (1 / 3)
    x[~mask_x] = (7.787 * x[~mask_x]) + (16 / 116)

    y[mask_y] = y[mask_y] ** (1 / 3)
    y[~mask_y] = (7.787 * y[~mask_y]) + (16 / 116)

    z[mask_z] = z[mask_z] ** (1 / 3)
    z[~mask_z] = (7.787 * z[~mask_z]) + (16 / 116)

    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    lab_array = np.stack([l, a, b], axis=-1)

    return lab_array



def labarray_to_rgb(lab_array):
    # Separar los canales L, a, b
    L = lab_array[..., 0]
    a = lab_array[..., 1]
    b = lab_array[..., 2]

    # Paso 1: Convertir L*a*b* a XYZ
    y = (L + 16.0) / 116.0
    x = y + (a / 500.0)
    z = y - (b / 200.0)

    # Definir umbrales
    delta = 6.0 / 29.0
    delta_cube = delta ** 3
    delta_sq = delta ** 2

    # Aplicar la transformación inversa
    x = np.where(x > delta, x ** 3, 3 * delta_sq * (x - (4.0 / 29.0)))
    y = np.where(y > delta, y ** 3, 3 * delta_sq * (y - (4.0 / 29.0)))
    z = np.where(z > delta, z ** 3, 3 * delta_sq * (z - (4.0 / 29.0)))

    # Multiplicar por los valores de referencia blancos
    x *= 95.047
    y *= 100.0
    z *= 108.883

    # Paso 2: Convertir XYZ a RGB
    X = x / 100.0
    Y = y / 100.0
    Z = z / 100.0

    # Matriz de conversión XYZ a sRGB
    R_lin = X *  3.2406 + Y * -1.5372 + Z * -0.4986
    G_lin = X * -0.9689 + Y *  1.8758 + Z *  0.0415
    B_lin = X *  0.0557 + Y * -0.2040 + Z *  1.0570

    # Asegurar que los valores estén en el rango [0, 1]
    R_lin = np.clip(R_lin, 0.0, 1.0)
    G_lin = np.clip(G_lin, 0.0, 1.0)
    B_lin = np.clip(B_lin, 0.0, 1.0)

    # Paso 3: Aplicar la corrección gamma inversa
    threshold = 0.0031308
    R = np.where(R_lin > threshold,
                 1.055 * (R_lin ** (1 / 2.4)) - 0.055,
                 12.92 * R_lin)
    G = np.where(G_lin > threshold,
                 1.055 * (G_lin ** (1 / 2.4)) - 0.055,
                 12.92 * G_lin)
    B = np.where(B_lin > threshold,
                 1.055 * (B_lin ** (1 / 2.4)) - 0.055,
                 12.92 * B_lin)

    # Paso 4: Combinar y escalar los valores a [0, 255]
    rgb_array = np.stack([R, G, B], axis=-1) * 255.0
    rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

    return rgb_array


def bgr_to_lab(bgr_array):
    """
    Conviertes un array (N,3) en BGR a Lab,
    usando tu función 'rgbarray_to_lab' (que asume RGB).
    """
    # 1) Reordenar BGR -> RGB
    rgb_array = bgr_array[..., ::-1].copy()  # invierte [B,G,R] -> [R,G,B]
    # 2) Llamar a la que sí tienes: 'rgbarray_to_lab'
    lab_array = rgbarray_to_lab(rgb_array)
    return lab_array



def dibujar_color_checker(corners, imagen, frameoriginal):
    corners = np.array(corners, dtype=np.float32)

    # Parámetros físicos
    long_total = 115.0  # mm
    ancho_total = 80.0  # mm
    borde_largo = 3.0
    borde_ancho = 5.0
    separacion = 2.0

    num_filas = 4
    num_columnas = 6

    # Calcular tamaño de parches
    long_parches = (long_total - 2 * borde_largo - (num_columnas - 1) * separacion) / num_columnas
    ancho_parches = (ancho_total - 2 * borde_ancho - (num_filas - 1) * separacion) / num_filas

    tamano_parches = 0.5  # Ajuste interno de parche

    # Generar parches normalizados
    parches_normalizados = []
    for i in range(num_filas):
        for j in range(num_columnas):
            x_ini = (borde_largo + j * (long_parches + separacion)) / long_total
            x_fin = (borde_largo + j * (long_parches + separacion) + long_parches) / long_total
            y_ini = (borde_ancho + i * (ancho_parches + separacion)) / ancho_total
            y_fin = (borde_ancho + i * (ancho_parches + separacion) + ancho_parches) / ancho_total

            # Ajustar tamaño del parche
            x_centro = (x_ini + x_fin) / 2.0
            y_centro = (y_ini + y_fin) / 2.0
            ancho_adj = (x_fin - x_ini) * tamano_parches / 2.0
            alto_adj = (y_fin - y_ini) * tamano_parches / 2.0

            x_ini_adj = x_centro - ancho_adj
            x_fin_adj = x_centro + ancho_adj
            y_ini_adj = y_centro - alto_adj
            y_fin_adj = y_centro + alto_adj

            # Orden: sup_izq, sup_der, inf_der, inf_izq
            parches_normalizados.append([
                [x_ini_adj, y_ini_adj],
                [x_fin_adj, y_ini_adj],
                [x_fin_adj, y_fin_adj],
                [x_ini_adj, y_fin_adj]
            ])

    parches_normalizados = np.array(parches_normalizados, dtype=np.float32)

    # Esquinas normalizadas
    pts_src = np.array([
        [0, 0],   # sup izq
        [1, 0],   # sup der
        [1, 1],   # inf der
        [0, 1]    # inf izq
    ], dtype=np.float32)

    # Matriz de perspectiva
    M = cv2.getPerspectiveTransform(pts_src, corners)

    # Transformar parches al plano imagen
    parches_imagen = []
    for parche in parches_normalizados:
        parche_img = cv2.perspectiveTransform(np.array([parche]), M)[0]
        parches_imagen.append(parche_img)
    parches_imagen = np.array(parches_imagen)

    colores_promedio = []
    centros_parches = []
    rectangulos = []  # (x, y, w, h)

    for parche in parches_imagen:
        mascara = np.zeros(imagen.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mascara, [parche.astype(np.int32)], 255)

        mean_val = cv2.mean(frameoriginal, mask=mascara)
        colores_promedio.append(mean_val)

        M_parche = cv2.moments(mascara)
        if M_parche["m00"] != 0:
            cX = M_parche["m10"] / M_parche["m00"]
            cY = M_parche["m01"] / M_parche["m00"]
        else:
            cX, cY = (0, 0)
        centros_parches.append((cX, cY))

        # Rectángulo bounding
        parche_int = parche.astype(np.int32)
        x, y, w, h = cv2.boundingRect(parche_int)
        rectangulos.append((x, y, w, h))

    centros_parches = np.array(centros_parches)

    # Numeración estándar
    lista_numeros = np.arange(1, num_filas * num_columnas + 1).reshape((num_filas, num_columnas))
    numeros_parches = lista_numeros.flatten()

    # Dibujar parches y números
    for idx, (parche, numero_parche) in enumerate(zip(parches_imagen, numeros_parches)):
        parche_int = parche.astype(np.int32)
        cv2.polylines(imagen, [parche_int], True, (255, 0, 0), 2)

        cX, cY = centros_parches[idx]
        cv2.putText(imagen, str(numero_parche), (int(cX) - 10, int(cY) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Dibujar contorno del checker
    cv2.polylines(imagen, [corners.astype(np.int32)], True, (0, 255, 0), 3)

    # Convertir (x, y, w, h) a (x_min, y_min, x_max, y_max) para igualar el formato del código manual
    rectangulos_final = []
    for (x, y, w, h) in rectangulos:
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        rectangulos_final.append((x_min, y_min, x_max, y_max))

    return rectangulos_final,imagen
