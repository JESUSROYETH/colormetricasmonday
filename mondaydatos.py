import os
import requests
import json
import datetime

# --- Configuración Inicial ---
apiKey = os.getenv("MONDAY")
if not apiKey:
    print("❌ Error: La variable de entorno MONDAY (API Key) no está configurada.")
    exit()

apiUrl = "https://api.monday.com/v2"
headers = {"Authorization": apiKey, "API-Version": "2023-10"}  # Revisa la última versión recomendada

board_id_to_work_on = 9238184753
#group_name_target = "Grupo de prueba"
  # El ítem y subítem tendrán este nombre


def manage_monday_group_and_item(group_name_target, valordelta,fecha):
    item_name_target = group_name_target
    # --- IDs DE COLUMNAS DEL ÍTEM PRINCIPAL (Verificados por el usuario) ---
    main_item_status_column_id = "status"
    main_item_date_column_id = "date4"
    main_item_deltae_column_id = "numeric_mkrb2nma"

    # --- IDs DE COLUMNAS DEL SUBÍTEM (¡MUY IMPORTANTE!) ---
    # Basado en el JSON proporcionado por el usuario:
    subitem_status_column_id = "status"  # ¡VERIFICAR ESTE ID! No estaba claro en el JSON si los subítems tienen esta columna o cuál es su ID.
    subitem_date_column_id = "date0"  # Actualizado según el JSON del usuario.
    subitem_deltae_column_id = "numeric_mkrbah6z"  # Actualizado según el JSON del usuario.

    # --- Variables Globales ---
    target_group_id = None
    processed_main_item_id = None  # ID del ítem principal, ya sea creado o existente

    # --- Parte 1: Obtener o Crear el Grupo ---
    print(f"--- Iniciando gestión del grupo '{group_name_target}' en el tablero ID: {board_id_to_work_on} ---")
    query_get_groups = f'''
    query {{
      boards (ids: {board_id_to_work_on}) {{
        groups {{
          id
          title
        }}
      }}
    }}
    '''
    data_get_groups = {'query': query_get_groups}
    print("Consultando grupos existentes...")
    r_get_groups = requests.post(url=apiUrl, json=data_get_groups, headers=headers)
    group_exists = False

    if r_get_groups.status_code == 200:
        try:
            response_get_groups = r_get_groups.json()
            if "errors" in response_get_groups:
                print(f"\n❌ Errores al obtener grupos: {response_get_groups['errors']}")
            elif "data" in response_get_groups and response_get_groups["data"]["boards"] and \
                    response_get_groups["data"]["boards"][0].get("groups"):
                for group in response_get_groups["data"]["boards"][0]["groups"]:
                    if group["title"] == group_name_target:
                        group_exists = True
                        target_group_id = group["id"]
                        print(f"\n✅ El grupo '{group_name_target}' ya existe con el ID: {target_group_id}.")
                        break
                if not group_exists:
                    print(f"ℹ️ Grupo '{group_name_target}' no encontrado en el tablero. Se creará.")
            else:
                print(
                    f"ℹ️ No se encontraron grupos en el tablero o la respuesta tuvo una estructura inesperada. Se intentará crear '{group_name_target}'.")
        except json.JSONDecodeError:
            print(f"\n❌ Error de JSON al parsear la respuesta de los grupos: {r_get_groups.text}")
    else:
        print(f"\n❌ Error en la solicitud HTTP al obtener grupos: {r_get_groups.status_code}, {r_get_groups.text}")

    if not group_exists:
        print(f"\nCreando grupo '{group_name_target}'...")
        mutation_create_group = f'''
        mutation {{
          create_group (board_id: {board_id_to_work_on}, group_name: "{group_name_target}") {{
            id
            title
          }}
        }}'''
        data_create_group = {'query': mutation_create_group}
        r_create_group = requests.post(url=apiUrl, json=data_create_group, headers=headers)
        if r_create_group.status_code == 200:
            try:
                response_data_create = r_create_group.json()
                if "errors" in response_data_create:
                    print(f"\n❌ Errores de la API al crear el grupo: {response_data_create['errors']}")
                elif "data" in response_data_create and response_data_create["data"]["create_group"]:
                    target_group_id = response_data_create["data"]["create_group"]["id"]
                    print(
                        f"\n✅ Grupo '{response_data_create['data']['create_group']['title']}' creado con éxito! ID: {target_group_id}")
                else:
                    print("❌ Respuesta inesperada de la API al crear el grupo.")
            except json.JSONDecodeError:
                print(f"\n❌ Error de JSON al parsear la respuesta de creación de grupo: {r_create_group.text}")
        else:
            print(f"\n❌ Error en la solicitud HTTP al crear el grupo: {r_create_group.status_code}, {r_create_group.text}")

    # --- Parte 2: Crear o Actualizar el Ítem Principal ---
    if target_group_id:
        print(f"\n--- Iniciando gestión del ítem principal '{item_name_target}' en el grupo ID: {target_group_id} ---")

        #today_date_iso = datetime.date.today().isoformat()
        desired_column_values_main_item = {
            main_item_status_column_id: {"label": "Listo"},
            main_item_date_column_id: {"date": fecha},
            main_item_deltae_column_id: 4
        }
        desired_column_values_main_item_json_str = json.dumps(desired_column_values_main_item)
        formatted_main_item_column_values_for_gql = json.dumps(desired_column_values_main_item_json_str)

        print(f"Verificando si el ítem '{item_name_target}' ya existe en el grupo '{group_name_target}'...")
        query_get_items_in_group = f'''
        query {{
          boards(ids: {board_id_to_work_on}) {{
            groups(ids: "{target_group_id}") {{
              items_page (limit: 100) {{ 
                items {{
                  id
                  name
                }}
              }}
            }}
          }}
        }}
        '''
        data_get_items = {'query': query_get_items_in_group}
        r_get_items = requests.post(url=apiUrl, json=data_get_items, headers=headers)

        existing_main_item_id = None
        if r_get_items.status_code == 200:
            try:
                response_get_items = r_get_items.json()
                if "errors" in response_get_items:
                    print(f"\n❌ Errores al buscar ítems: {response_get_items['errors']}")
                elif "data" in response_get_items and response_get_items["data"]["boards"] and \
                        response_get_items["data"]["boards"][0].get("groups") and \
                        response_get_items["data"]["boards"][0]["groups"][0].get("items_page") and \
                        response_get_items["data"]["boards"][0]["groups"][0]["items_page"].get("items"):

                    items_in_group = response_get_items["data"]["boards"][0]["groups"][0]["items_page"]["items"]
                    for item_data in items_in_group:
                        if item_data["name"] == item_name_target:
                            existing_main_item_id = item_data["id"]
                            print(
                                f"✅ Ítem principal '{item_name_target}' encontrado con ID: {existing_main_item_id}. Se actualizará.")
                            break
                    if not existing_main_item_id:
                        print(f"ℹ️ Ítem principal '{item_name_target}' no encontrado en el grupo. Se creará.")
                else:
                    print(
                        f"ℹ️ No se encontraron ítems en el grupo '{group_name_target}' o la respuesta tuvo una estructura inesperada. Se intentará crear el ítem.")
            except json.JSONDecodeError:
                print(f"\n❌ Error de JSON al parsear la respuesta de búsqueda de ítems: {r_get_items.text}")
        else:
            print(f"\n❌ Error en la solicitud HTTP al buscar ítems: {r_get_items.status_code}, {r_get_items.text}")

        if existing_main_item_id:
            processed_main_item_id = existing_main_item_id
            print(f"Actualizando ítem principal ID: {existing_main_item_id}...")
            mutation_update_item = f'''
            mutation {{
              change_multiple_column_values (
                item_id: {existing_main_item_id},
                board_id: {board_id_to_work_on},
                column_values: {formatted_main_item_column_values_for_gql} 
              ) {{ id }}
            }}'''
            data_update_item = {'query': mutation_update_item}
            r_update_item = requests.post(url=apiUrl, json=data_update_item, headers=headers)
            if r_update_item.status_code == 200:
                try:
                    response_update = r_update_item.json()
                    if "errors" in response_update:
                        print(f"\n❌ Errores de API al actualizar ítem principal: {response_update['errors']}")
                    elif "data" in response_update and response_update["data"]["change_multiple_column_values"]:
                        print(
                            f"✅ Ítem principal ID: {response_update['data']['change_multiple_column_values']['id']} actualizado exitosamente.")
                    else:
                        print("❌ Respuesta inesperada de API al actualizar ítem principal.")
                except json.JSONDecodeError:
                    print(f"\n❌ Error de JSON al parsear respuesta de actualización de ítem: {r_update_item.text}")
            else:
                print(f"\n❌ Error HTTP al actualizar ítem principal: {r_update_item.status_code}, {r_update_item.text}")

        else:
            print(f"Creando nuevo ítem principal '{item_name_target}' en grupo ID: {target_group_id}...")
            mutation_create_item = f'''
            mutation {{
              create_item (
                board_id: {board_id_to_work_on},
                group_id: "{target_group_id}",
                item_name: "{item_name_target}",
                column_values: {formatted_main_item_column_values_for_gql} 
              ) {{ id name }}
            }}'''
            data_create_item = {'query': mutation_create_item}
            r_create_item = requests.post(url=apiUrl, json=data_create_item, headers=headers)
            if r_create_item.status_code == 200:
                try:
                    response_create_item = r_create_item.json()
                    if "errors" in response_create_item:
                        print(f"\n❌ Errores de API al crear ítem principal: {response_create_item['errors']}")
                    elif "data" in response_create_item and response_create_item["data"]["create_item"]:
                        processed_main_item_id = response_create_item["data"]["create_item"]["id"]
                        print(f"✅ Nuevo ítem principal '{item_name_target}' creado con ID: {processed_main_item_id}.")
                    else:
                        print("❌ Respuesta inesperada de API al crear ítem principal.")
                except json.JSONDecodeError:
                    print(f"\n❌ Error de JSON al parsear respuesta de creación de ítem: {r_create_item.text}")
            else:
                print(f"\n❌ Error HTTP al crear ítem principal: {r_create_item.status_code}, {r_create_item.text}")

        # --- Parte 3: Crear SIEMPRE un Nuevo Subítem ---
        if processed_main_item_id:
            subitem_name_to_create = item_name_target
            print(
                f"\n--- Creando SIEMPRE un nuevo subítem '{subitem_name_to_create}' para el ítem principal ID: {processed_main_item_id} ---")

            # Prepara los valores para las columnas del subítem.
            # Si no quieres establecer el estado en el subítem (o no tiene columna de estado),
            # elimina la línea de 'subitem_status_column_id' de este diccionario.
            desired_column_values_subitem = {
                subitem_status_column_id: {"label": "Listo"},
                subitem_date_column_id: {"date": today_date_iso},
                subitem_deltae_column_id: float(valordelta)
            }

            # Si no hay columna de estado para subítems o no se quiere establecer, se puede omitir:
            # Por ejemplo, si 'subitem_status_column_id' no es válido o no se quiere usar:
            # if not subitem_status_column_id_is_valid_or_wanted: # Necesitarías una lógica para esto
            #     if subitem_status_column_id in desired_column_values_subitem:
            #         del desired_column_values_subitem[subitem_status_column_id]

            subitem_column_values_json_str = json.dumps(desired_column_values_subitem)
            formatted_subitem_column_values_for_gql = json.dumps(subitem_column_values_json_str)

            mutation_create_subitem = f'''
            mutation {{
              create_subitem (
                parent_item_id: {processed_main_item_id},
                item_name: "{subitem_name_to_create}",
                column_values: {formatted_subitem_column_values_for_gql}
              ) {{
                id
                name
                board {{ id }} 
              }}
            }}
            '''
            data_create_subitem = {'query': mutation_create_subitem}
            r_create_subitem = requests.post(url=apiUrl, json=data_create_subitem, headers=headers)

            if r_create_subitem.status_code == 200:
                try:
                    response_create_subitem = r_create_subitem.json()
                    if "errors" in response_create_subitem:
                        print(f"\n❌ Errores de API al crear el subítem:")
                        for error in response_create_subitem["errors"]:
                            msg = error.get('message', str(error))
                            col_id = error.get('column_id', '')
                            print(f"- {msg} (Columna ID afectada si aplica: {col_id if col_id else 'N/A'})")
                    elif "data" in response_create_subitem and response_create_subitem["data"]["create_subitem"]:
                        new_subitem_id = response_create_subitem["data"]["create_subitem"]["id"]
                        print(
                            f"✅ Nuevo subítem '{subitem_name_to_create}' creado con ID: {new_subitem_id} bajo el ítem principal {processed_main_item_id}.")
                    else:
                        print("❌ Respuesta inesperada de API al crear el subítem.")
                except json.JSONDecodeError:
                    print(f"\n❌ Error de JSON al parsear respuesta de creación de subítem: {r_create_subitem.text}")
            else:
                print(
                    f"\n❌ Error HTTP al crear subítem: {r_create_subitem.status_code}. Respuesta: {r_create_subitem.text}")
        else:
            print("\n❌ No se pudo obtener o crear un ID de ítem principal válido. No se creará el subítem.")

    else:
        print("\n❌ No se pudo obtener o crear el grupo. No se procederá con la creación/actualización del ítem y subítem.")

    print("\n--- Proceso finalizado ---")
