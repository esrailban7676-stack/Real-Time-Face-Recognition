import cv2
import numpy as np
import os
import glob

# Yüz Algılayıcı Modülü
face_cascade = None
try:
    import cv2.objdetect as objdetect
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = objdetect.CascadeClassifier(cascade_path)
except Exception:
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade = None

def onislem(img_gray):
    kucuk = cv2.resize(img_gray, (100, 100))
    return cv2.equalizeHist(kucuk)

# TÜM REFERANS FOTOĞRAFLARI YÜKLE (ben*.jpg)
ref_list = []
fotolar = glob.glob("ben*.jpg") + glob.glob("ben*.png")

if len(fotolar) == 0:
    print("HATA: Klasorde 'ben' ile baslayan hic fotograf bulunamadi!")
    exit()

print(f"--> {len(fotolar)} adet referans fotograf yukleniyor...")

for dosya in fotolar:
    bgr = cv2.imread(dosya)
    if bgr is None:
        continue
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    if face_cascade and not face_cascade.empty():
        rf = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(rf) > 0:
            rx, ry, rw, rh = rf[0]
            gray = gray[ry:ry+rh, rx:rx+rw]
            
    islenmis = onislem(gray)
    h_ref = cv2.calcHist([islenmis], [0], None, [32], [0, 256])
    cv2.normalize(h_ref, h_ref)
    ref_list.append((islenmis, h_ref))
    print(f"   [+] {dosya} basariyla hafizaya alindi.")

# KAMERAYI BAŞLAT
cap = cv2.VideoCapture(0)
ESIK = 0.22

print("\n--> KAMERA BASLATILDI! Cikmak icin 'q' tusuna basin.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = []
    if face_cascade and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

    if len(faces) == 0:
        faces = [(int(fw * 0.28), int(fh * 0.15), int(fw * 0.44), int(fh * 0.65))]

    for (x, y, w, h) in faces:
        live_crop = gray[y:y+h, x:x+w]
        if live_crop.size == 0:
            continue

        live_face = onislem(live_crop)
        h_live = cv2.calcHist([live_face], [0], None, [32], [0, 256])
        cv2.normalize(h_live, h_live)

        # Tüm fotoğraflarla karşılaştır ve EN YÜKSEK benzerliği al
        skorlar = []
        for ref_face, h_ref in ref_list:
            match_val = cv2.matchTemplate(live_face, ref_face, cv2.TM_CCOEFF_NORMED)[0][0]
            hist_val = cv2.compareHist(h_ref, h_live, cv2.HISTCMP_CORREL)
            s = max(0.0, float(match_val * 0.6 + hist_val * 0.4))
            skorlar.append(s)

        skor = max(skorlar) if len(skorlar) > 0 else 0.0

        # Karar: Yeşil mi Kırmızı mı?
        if skor >= ESIK:
            puan = min(99, int((skor / 0.55) * 100))
            durum = f"SENSIN (%{puan})"
            renk = (0, 255, 0)      # YEŞİL
        else:
            durum = f"SEN DEGILSIN (Skor: {round(skor,2)})"
            renk = (0, 0, 255)      # KIRMIZI

        cv2.rectangle(frame, (x, y), (x + w, y + h), renk, 3)
        cv2.putText(frame, durum, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)

    cv2.imshow("Coklu Referansli Yuz Tanima - Odev", frame)

    # Anlık ekleme: Kameradayken 'k' basarsan o anki yüzünü de hafızaya ekler
    key = cv2.waitKey(1) & 0xFF
    if key == ord('k') and len(faces) > 0:
        (x, y, w, h) = faces[0]
        yeni_yuz = onislem(gray[y:y+h, x:x+w])
        h_yeni = cv2.calcHist([yeni_yuz], [0], None, [32], [0, 256])
        cv2.normalize(h_yeni, h_yeni)
        ref_list.append((yeni_yuz, h_yeni))
        print("--> Anlik yuzun de yeni referans olarak eklendi!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()