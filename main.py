import cv2
import numpy as np
import time
from djitellopy import Tello

# Declaramos las coordenadas "X" y "Y" o sea el centro del frame
rifX = 960 / 2
rifY = 720 / 2

# Constantes del controlador PID
Kp_X = 0.1
Ki_X = 0.0
Kd_X = 0.01
Kp_Y = 0.2
Ki_Y = 0.0
Kd_Y = 0.01
Kp_D = 0.02
Ki_D = 0.0
Kd_D = 0.01

# Constante para definir el tiempo del ciclo
Tc = 0.05

# Términos de PID inicializado
integral_X = 0
error_X = 0
previous_error_X = 0
integral_Y = 0
error_Y = 0
previous_error_Y = 0
integral_D = 0
previous_error_D = 0

centroX_pre = rifX
centroY_pre = rifY

# Distancia segura basada en el tamaño del cuadro delimitador
safe_distance = 800  # Ajusta esta distancia según sea necesario
margin = 70

# Red neuronal convolucional aquí hacemos la carga del modelo (Red neuronal preentrenada)
net = cv2.dnn.readNetFromCaffe("archive/MobileNetSSD_deploy.prototxt.txt", "archive/MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Inicializamos el dron e imprimimos el estado de la conexión,
# la batería, la inicialización de la cámara y cuando aterriza el dron
drone = Tello()
time.sleep(2.0)
print("Conectando al dron")
drone.connect()
print("BATTERY: ")
print(drone.get_battery())
time.sleep(1.0)
print("Cargando")
drone.streamon()
print("Ascendiendo")
drone.takeoff()
time.sleep(4.0)

# Variables para control de giro
turning = False
turning_speed = 20
max_turn_duration = 80
start_turn_time = None
search_start_time = time.time()

# Variable para asegurar que no se envíen comandos de movimiento antes de detectar una persona
person_detected = False

# Altura deseada en cm
target_height = 130

while True:
    try:
        # Obtener la altura actual del dron
        current_height = drone.get_height()
        height_error = target_height - current_height
        print(f"Current Height: {current_height}, Target Height: {target_height}, Height Error: {height_error}")

        # Control de altura simple basado en el error de altura
        if abs(height_error) > 5:
            height_adjustment = int(height_error * 0.5)
        else:
            height_adjustment = 0

        start = time.time()
        frame = drone.get_frame_read().frame
        cv2.circle(frame, (int(rifX), int(rifY)), 1, (0, 0, 255), 10)

        h, w, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        detected_in_this_frame = False

        for i in np.arange(0, detections.shape[2]):
            idx = int(detections[0, 0, i, 1])
            confidence = detections[0, 0, i, 2]

            if CLASSES[idx] == "person" and confidence > 0.5:
                detected_in_this_frame = True

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

                centroX = (startX + endX) / 2
                centroY = (startY + endY) / 2

                centroX_pre = centroX
                centroY_pre = centroY

                cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

                error_X = -(rifX - centroX)
                error_Y = rifY - centroY

                box_size = endX - startX
                distance_error = safe_distance - box_size

                cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

                integral_X = integral_X + error_X * Tc
                derivative_X = (error_X - previous_error_X) / Tc
                uX = Kp_X * error_X + Ki_X * integral_X + Kd_X * derivative_X
                previous_error_X = error_X

                integral_Y = integral_Y + error_Y * Tc
                derivative_Y = (error_Y - previous_error_Y) / Tc
                uY = Kp_Y * error_Y + Ki_Y * integral_Y + Kd_Y * derivative_Y
                previous_error_Y = error_Y

                if abs(distance_error) > margin:
                    integral_D += distance_error * Tc
                    derivative_D = (distance_error - previous_error_D) / Tc
                    uD = Kp_D * distance_error + Ki_D * integral_D + Kd_D * derivative_D
                    previous_error_D = distance_error

                max_value = 30
                min_value = -30
                uX = max(min(uX, max_value), min_value)
                uY = max(min(uY, max_value), min_value)
                uD = max(min(uD, max_value), min_value)

                # Mandamos los comandos al dron para que este los siga solo si se ha detectado una persona
                drone.send_rc_control(0, round(uD), round(uY), round(uX))
                turning = False
                break

        if detected_in_this_frame:
            person_detected = True
            print("Persona detectada")
        else:
            person_detected = False

        if not person_detected:
            drone.send_rc_control(0, 0, height_adjustment, 0)
            print("Persona no detectada, manteniendo posición inicial.")
            if not turning:
                print("Iniciando vuelta de reconocimiento...")
                turning = True
                start_turn_time = time.time()
            else:
                elapsed_turn_time = time.time() - start_turn_time
                if elapsed_turn_time < max_turn_duration:
                    drone.send_rc_control(0, 0, height_adjustment, turning_speed)
                    print("Vuelta de reconocimiento en progreso...")
                else:
                    print("Vuelta de reconocimiento completada, sin detección de persona.")
                    drone.send_rc_control(0, 0, height_adjustment, 0)
                    turning = False

                    if time.time() - search_start_time > 5:
                        print("No se detectó ninguna persona, descendiendo...")
                        drone.land()
                        break

        cv2.imshow("Frame", frame)

        end = time.time()
        elapsed = end - start
        if Tc - elapsed > 0:
            time.sleep(Tc - elapsed)
        end_ = time.time()
        elapsed_ = end_ - start
        fps = 1 / elapsed_
        print("FPS: ", fps)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error en el procesamiento del frame: {e}")
        continue

drone.streamoff()
cv2.destroyAllWindows()
drone.land()
print("Aterrizando...")
print("Bateria: ")
print(drone.get_battery())
drone.end()
