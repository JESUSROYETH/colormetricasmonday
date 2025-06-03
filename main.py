import base64
import json

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from google.cloud import storage
import requests
import os
from utils import dibujar_color_checker
from colorestarget import colores_target_bgr
from utils import bgr_to_lab, rgbarray_to_lab
import flask


# crear app
app = flask.Flask(__name__)

def descargar_modelo(url, ruta_destino):


    if os.path.exists(ruta_destino):
        print(f"El modelo ya existe en {ruta_destino}")
        return ruta_destino

    print(f"Descargando modelo desde {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    with open(ruta_destino, 'wb') as f:
        if total == 0:
            f.write(response.content)
        else:
            descargado = 0
            for datos in response.iter_content(chunk_size=8192):
                descargado += len(datos)
                f.write(datos)
                progreso = int(50 * descargado / total)
                print(
                    f"\r[{'=' * progreso}{' ' * (50 - progreso)}] {descargado / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB",
                    end='')
    print(f"\nModelo guardado en {ruta_destino}")
    return ruta_destino


# URL y ruta del modelo
modelo_url = "https://storage.googleapis.com/lythium-datasets-hot/colour-checker-detection-l-seg.pt"
modelo_ruta = "colour-checker-detection-l-seg.pt"

# Descargar el modelo
modelo_ruta = descargar_modelo(modelo_url, modelo_ruta)

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

@app.route('/', methods=['POST'])
def index():
    envelope = flask.request.get_json()
    print(f"DEBUG: Received envelope: {json.dumps(envelope, indent=2)}")
    if not envelope:
        msg = "No Pub/Sub message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "Invalid Pub/Sub message format"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        try:
            data_bytes = base64.b64decode(pubsub_message["data"])
            data_str = data_bytes.decode("utf-8")
            gcs_event_data = json.loads(data_str)
            print(f"Datos del evento GCS recibidos: {gcs_event_data}")

        except Exception as e:
            msg = f"Error decodificando o parseando datos del mensaje Pub/Sub: {e}"
            print(msg)
            return f"Bad Request: {msg}", 400

        bucket_name = gcs_event_data.get("bucket")
        file_name = gcs_event_data.get("name")
        # metageneration = gcs_event_data.get("metageneration") # Para referencia
        # timeCreated = gcs_event_data.get("timeCreated") # Para referencia


        if not bucket_name or not file_name:
            msg = "Datos del evento GCS incompletos (falta bucket o name)."
            print(msg)
            return f"Bad Request: {msg}", 400

        print(f"Evento para Bucket: {bucket_name}, Archivo: {file_name}")

        if file_name.lower().endswith(".avi"):
            print(f"Archivo AVI detectado: {file_name}")
            try:
                storage_client = storage.Client()
                bucket_gcs = storage_client.bucket(bucket_name)
                blob = bucket_gcs.blob(file_name)

                base_name = os.path.basename(file_name)
                destination_file_name = f"/tmp/{base_name}"

                print(f"Descargando {file_name} de gs://{bucket_name} a {destination_file_name}...")
                blob.download_to_filename(destination_file_name)
                print(f"Archivo descargado exitosamente.")

                # Llama a tu función de procesamiento.
                # 'modelo_ruta' contiene la ruta al modelo descargado globalmente.
                # Asumimos que detect_colorchecker_with_ultralytics_video está definida
                # y maneja la lógica de 'guardar' y dónde guarda el archivo.
                # Si necesitas el nombre del archivo procesado para subirlo,
                # tu función detect_... debería retornarlo.
                detect_colorchecker_with_ultralytics_video(
                    video_path=destination_file_name,
                    model_path=modelo_ruta, # Esta es la variable global con la ruta al modelo
                    guardar=True,
                    mostrar=False
                )
                print(f"Procesamiento de video completado para {file_name}.")

                # (Opcional) Lógica para subir el video procesado de vuelta a GCS
                # Si 'detect_colorchecker_with_ultralytics_video' guarda el archivo en una ruta conocida
                # (ej. en /tmp/ con un sufijo _procesado), puedes subirlo aquí.
                # Ejemplo:
                # processed_video_local_path = f"/tmp/{os.path.splitext(base_name)[0]}_procesado.avi"
                # if os.path.exists(processed_video_local_path):
                #    processed_video_gcs_name = f"processed/{os.path.basename(processed_video_local_path)}"
                #    output_blob = bucket_gcs.blob(processed_video_gcs_name)
                #    output_blob.upload_from_filename(processed_video_local_path)
                #    print(f"Video procesado subido a gs://{bucket_name}/{processed_video_gcs_name}")
                #    os.remove(processed_video_local_path) # Limpiar archivo local procesado
                # else:
                #    print(f"Video procesado {processed_video_local_path} no encontrado para subir.")


                if os.path.exists(destination_file_name): # Limpiar archivo local descargado
                    os.remove(destination_file_name)

                return flask.jsonify({"message": f"Procesamiento iniciado para {file_name}"}), 200

            except Exception as e:
                import traceback
                error_msg = f"Error procesando el archivo {file_name}: {e}\n{traceback.format_exc()}"
                print(error_msg)
                return error_msg, 500
        else:
            msg = f"El archivo {file_name} no es .avi. Se omite el procesamiento."
            print(msg)
            return flask.jsonify({"message": msg, "file_processed": False}), 200
    else:
        msg = "Formato de mensaje Pub/Sub inesperado (falta 'data')."
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

