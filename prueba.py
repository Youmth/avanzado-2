import cv2

# Inicializar la cámara (0 es generalmente la cámara por defecto)
cap = cv2.VideoCapture(1)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Verificar si la captura fue exitosa
    if not ret:
        print("No se pudo recibir frame. Saliendo ...")
        break

    # Mostrar el frame actual
    cv2.imshow('Video en Tiempo Real', frame)

    # Salir del loop al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
