import cv2
import numpy as np
import base64

def b64_to_cv2(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def extract_face_embedding(frame_bgr: np.ndarray):
    """
    Dummy/basic face embedding extraction for proctoring.
    We convert the face region to a color histogram.
    """
    if frame_bgr is None:
        return None
    # For a simple face verification, just use cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    
    # Use the largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    face_crop = frame_bgr[y:y+h, x:x+w]
    
    face_crop = cv2.resize(face_crop, (128, 128))
    hist = cv2.calcHist([face_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def verify_face(reference_embedding, live_frame_bgr, threshold=0.6):
    """
    Returns True if face matches, False otherwise.
    """
    if reference_embedding is None:
        return {"match": True, "alert": "No reference checking"}
        
    live_embedding = extract_face_embedding(live_frame_bgr)
    if live_embedding is None:
        return {"match": False, "alert": "No face detected in live feed!"}
        
    similarity = cv2.compareHist(
        reference_embedding.astype(np.float32),
        live_embedding.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    match = similarity >= threshold
    
    return {
        "match": match,
        "similarity": float(similarity),
        "alert": "" if match else "Face mismatch detected! (Proctoring alert)"
    }
