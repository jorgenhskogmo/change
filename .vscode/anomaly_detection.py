import cv2
import numpy as np

# Last inn videoen
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Bakgrunnsmodell for å oppdage endringer
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Definer gule fargeterskler i HSV
yellow_lower = np.array([20, 100, 100], dtype=np.uint8)  # Juster verdiene etter behov
yellow_upper = np.array([30, 255, 255], dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Forbehandling: bruk bakgrunnssubtraksjon for å fjerne uønskede detaljer
    foreground_mask = background_subtractor.apply(frame)

    # Konverter bildet til HSV-fargerom
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detekter gule områder
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    
    # Kombiner med bakgrunnsmaske for å detektere endringer som også er gule
    combined_mask = cv2.bitwise_and(foreground_mask, yellow_mask)

    # Morfologiske operasjoner for å forbedre masken
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Finn konturer i den kombinerte masken
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tell og merk boksene
    count = 0
    for contour in contours:
        # Filtrer ut små konturer for å unngå falske deteksjoner
        if cv2.contourArea(contour) > 500:  # Juster terskelverdi etter behov
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

    # Vis antall bokser
    cv2.putText(frame, f"Antall bokser: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Vis resultatet
    cv2.imshow('Detected Changes', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()