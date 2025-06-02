import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import functions_framework
import os
from google.cloud import storage

from utils import dibujar_color_checker
from colorestarget import colores_target_bgr
from utils import bgr_to_lab, rgbarray_to_lab
print("CUDA disponible:", torch.cuda.is_available())


def find_corners(points: np.ndarray) -> np.ndarray:
    """
    Devuelve los cuatro vértices (tl, tr, br, bl) de la caja mínima.
    """
    pts = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.int32)

    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).flatten()
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]

    return np.array([tl, tr, br, bl])


def order_corners_from_first(corners: np.ndarray) -> np.ndarray:
    """
    Reordena los vértices tomando corners[0] como punto inicial.
    Devuelve [P0, P1, P2, P3] en sentido horario o antihorario
    según la orientación real del patrón.
    """
    P0 = corners[0]
    others = corners[1:]

    # Distancias cuadradas al primer punto
    dists = np.linalg.norm(others - P0, axis=1)**2
    diag_idx = np.argmax(dists)          # punto diagonal (más lejano)
    P2 = others[diag_idx]

    # Los dos puntos adyacentes
    adj = [others[i] for i in range(3) if i != diag_idx]
    # Selecciona como P1 el de mayor delta en x o y
    deltas = [np.max(np.abs(p - P0)) for p in adj]
    idx_max = int(np.argmax(deltas))
    P1 = adj[idx_max]
    P3 = adj[1 - idx_max]

    return np.array([P0, P1, P2, P3], dtype=np.int32)


def generar_candidatos(corners: np.ndarray) -> list[np.ndarray]:
    """
    Recibe `corners` (shape = (4,2) ya ordenado por tu lógica previa)
    y devuelve las cuatro posibles orientaciones:

    0) (1,2,3,4)  → corners
    1) (4,3,2,1)  → corners[::-1]
    2) (2,1,4,3)  → corners[[1,0,3,2]]
    3) (3,4,1,2)  → corners[[2,3,0,1]]
    """
    return [
        corners,
        corners[::-1],
        corners[[1, 0, 3, 2]],
        corners[[2, 3, 0, 1]],
    ]

def rect_mean_lab(frame, rect):
    x0, y0, x1, y1 = rect
    mean_bgr = frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)        # (3,)
    mean_lab = bgr_to_lab(mean_bgr.reshape(1, 3))[0]                  # (3,)
    return mean_lab        # (3,)

def deltaE(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)                    # ΔE-76

def evaluar_candidato(frame, rects, target_lab):
    pred_lab = np.array([rect_mean_lab(frame, r) for r in rects])
    return np.mean([deltaE(p, t) for p, t in zip(pred_lab, target_lab)])

def detect_colorchecker_with_ultralytics_video(video_path: str, model_path: str, guardar: bool = False, mostrar: bool = True):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)



    target_lab = bgr_to_lab(np.array(colores_target_bgr, dtype=np.float32))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metricas = []

    if guardar:
        base, ext = os.path.splitext(video_path)
        out_name = f"{base}_procesado.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # o 'MJPG'
        out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        if not results or not results[0].masks:
            print("No se detectó el color checker.")
            if mostrar:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.imshow('BEST', frame)
            if guardar:
                out.write(frame)
            continue
        raw_pts = results[0].masks.xy[0]          # (N, 2)
        base_corners = find_corners(raw_pts)       # tl, tr, br, bl
        ordered_corners = order_corners_from_first(base_corners)

        candidatos = generar_candidatos(ordered_corners)
        cuatro_opciones = []
        for idx, cand in enumerate(candidatos):
            rects, img_vis = dibujar_color_checker(cand, frame.copy(), frame)
            cuatro_opciones.append([rects, img_vis])

        deltaE_scores = []
        for rects, _ in cuatro_opciones:  # img_vis no hace falta aquí
            score = evaluar_candidato(frame, rects, target_lab)
            deltaE_scores.append(score)

        best_idx = int(np.argmin(deltaE_scores))  # el de menor ΔE medio
        best_rects, best_img = cuatro_opciones[best_idx]  # opcional: usa best_img para display
        valor_deltaE = deltaE_scores[best_idx]
        cv2.putText(best_img, f"Delta_E={valor_deltaE:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        all_x_coords = base_corners[:, 0]  # Obtiene todas las coordenadas x de las esquinas
        min_x_cc = np.min(all_x_coords)
        max_x_cc = np.max(all_x_coords)
        center_x_cc = (min_x_cc + max_x_cc) / 2  # Centro horizontal del color checker

        # 'w' es el ancho del frame, ya lo tienes calculado.
        left_exclusion_limit = 0.25 * w
        right_exclusion_limit = 0.75 * w  # que es lo mismo que w - (0.25 * w)

        # Condición para añadir la métrica: SOLO si NO está en la zona de exclusión
        if center_x_cc >= left_exclusion_limit and center_x_cc <= right_exclusion_limit:
            metricas.append(valor_deltaE)
            # El cv2.putText con el Delta_E y la visualización/guardado de best_img
            # se mantienen como están, mostrando la detección independientemente de si se guarda la métrica.
            # Si quisieras no mostrar/guardar la detección si se ignora la métrica,
            # el flujo de cv2.putText, imshow, out.write debería estar dentro de este 'if'.
        else:
            print(
                f"Color checker en zona de exclusión horizontal (centro en {center_x_cc:.0f}px de {w}px). Métrica Delta_E={valor_deltaE:.1f} ignorada.")
        if mostrar:
            cv2.imshow('BEST', best_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if guardar:
            out.write(best_img)

    cap.release()
    cv2.destroyAllWindows()
    if guardar:
        out.release()
    # retornar el promedio de las métricas
    if metricas:
        promedio_metricas = np.mean(metricas)
        print(f"Promedio de métricas Delta_E: {promedio_metricas:.2f}")
    else:
        print("No se registraron métricas Delta_E válidas.")



@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket_name = data["bucket"]
    file_name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket_name}")
    print(f"File: {file_name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")

    # Comprobar si el archivo es .avi
    if file_name.lower().endswith(".avi"):
        print(f"Archivo AVI detectado: {file_name}")

        try:
            # Inicializar el cliente de Cloud Storage
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            # Definir la ruta de destino en el sistema de archivos temporal de la Cloud Function
            # Asegúrate de que el nombre del archivo no contenga caracteres inválidos para nombres de archivo.
            # Podrías querer sanitizar `file_name` si puede contener rutas.
            base_name = os.path.basename(file_name)
            destination_file_name = f"/tmp/{base_name}"

            # Descargar el archivo
            blob.download_to_filename(destination_file_name)
            print(f"Archivo {file_name} descargado a {destination_file_name}")
            model_path = 'colour-checker-detection-l-seg.pt'
            detect_colorchecker_with_ultralytics_video(
                destination_file_name, model_path, guardar=True, mostrar=False)
            print(f"Procesamiento completado para {file_name}.")




        except Exception as e:
            print(f"Error al descargar o procesar el archivo {file_name}: {e}")
    else:
        print(f"El archivo {file_name} no es un archivo .avi. Se omite la descarga.")


# if __name__ == "__main__":
#     video_path = 'color_checker_25-05-26_11-18-16-064877__mowi-chacabuco.avi'
#     #video_path = '24-11-08_18-07-29-185002__aq_calbuco.avi'
#     model_path = 'colour-checker-detection-l-seg.pt'
#     detect_colorchecker_with_ultralytics_video(video_path, model_path, guardar=False, mostrar=False)
