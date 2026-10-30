import os, cv2, sqlite3, uuid, subprocess, wave, struct, math
from datetime import datetime
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
UPLOADS = os.path.join(DATA, "uploads")
FACES = os.path.join(DATA, "faces")
DB = os.path.join(DATA, "attendance.db")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(FACES, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-local-demo-secret"
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png"}
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            image_file TEXT NOT NULL, created_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
            created_at TEXT NOT NULL, duration REAL DEFAULT 0,
            frames_analyzed INTEGER DEFAULT 0, face_detections INTEGER DEFAULT 0,
            audio_activity REAL DEFAULT NULL, summary TEXT DEFAULT '')""")
        c.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL, first_seen REAL NOT NULL,
            last_seen REAL NOT NULL, appearances INTEGER NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id),
            FOREIGN KEY(student_id) REFERENCES students(id))""")

def get_recognizer():
    # LBPH is available with opencv-contrib-python, not the base opencv-python package.
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        return None, {}, "Install opencv-contrib-python to enable named student recognition."
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, label_to_student = [], [], {}
    with db() as c:
        students = c.execute("SELECT id, image_file FROM students").fetchall()
    for s in students:
        path = os.path.join(FACES, s["image_file"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        found = face_cascade.detectMultiScale(img, 1.2, 5, minSize=(50, 50))
        if len(found):
            x,y,w,h = max(found, key=lambda r:r[2]*r[3])
            crop = cv2.resize(img[y:y+h, x:x+w], (160,160))
            faces.append(crop); labels.append(int(s["id"]))
            label_to_student[int(s["id"])] = int(s["id"])
    if not faces:
        return None, {}, "Add clear student face photos first; each photo must show one face."
    recognizer.train(faces, __import__("numpy").array(labels, dtype="int32"))
    return recognizer, label_to_student, ""

def analyze_audio(video_path):
    """Optional offline audio activity estimate; requires ffmpeg available on PATH."""
    wav_path = os.path.splitext(video_path)[0] + "_audio.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=90)
        with wave.open(wav_path, "rb") as wf:
            rate, width, frames = wf.getframerate(), wf.getsampwidth(), wf.getnframes()
            raw = wf.readframes(frames)
        if width != 2 or not raw: return None
        samples = struct.unpack("<" + "h"*(len(raw)//2), raw)
        chunk = max(1, int(rate * 0.5))
        active = total = 0
        for i in range(0, len(samples), chunk):
            part = samples[i:i+chunk]
            if not part: continue
            rms = math.sqrt(sum(v*v for v in part)/len(part))
            total += 1
            if rms > 500: active += 1
        return round(100*active/max(total,1), 1)
    except Exception:
        return None
    finally:
        try:
            if os.path.exists(wav_path): os.remove(wav_path)
        except OSError: pass

def analyze_video(video_path):
    recognizer, label_map, err = get_recognizer()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video. Try MP4 (H.264) or AVI."
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0
    # Sample about one frame per second, capped for a responsive classroom demo.
    step = max(1, int(fps))
    frame_no = 0
    detections = 0
    seen = {}
    analyzed = 0
    unknown_count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_no % step == 0:
            analyzed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(55,55))
            timestamp = frame_no / fps if fps else 0
            detections += len(found)
            for x,y,w,h in found:
                crop = cv2.resize(gray[y:y+h, x:x+w], (160,160))
                if recognizer is None:
                    unknown_count += 1
                    continue
                label, confidence = recognizer.predict(crop)
                # LBPH confidence is a distance: lower is a better match. Threshold is approximate and needs calibration.
                if confidence < 75 and label in label_map:
                    rec = seen.setdefault(label, {"first": timestamp, "last": timestamp, "count": 0})
                    rec["first"] = min(rec["first"], timestamp)
                    rec["last"] = max(rec["last"], timestamp)
                    rec["count"] += 1
                else:
                    unknown_count += 1
        frame_no += 1
    cap.release()
    audio_activity = analyze_audio(video_path)
    with db() as c:
        cur = c.execute("""INSERT INTO sessions(filename, created_at, duration, frames_analyzed,
            face_detections, audio_activity, summary) VALUES(?,?,?,?,?,?,?)""",
            (os.path.basename(video_path), datetime.now().isoformat(timespec="seconds"),
             round(duration,1), analyzed, detections, audio_activity, ""))
        session_id = cur.lastrowid
        for sid, v in seen.items():
            c.execute("INSERT INTO attendance(session_id, student_id, first_seen, last_seen, appearances) VALUES(?,?,?,?,?)",
                      (session_id, sid, v["first"], v["last"], v["count"]))
    return {"id": session_id, "duration": round(duration,1), "frames": analyzed,
            "detections": detections, "recognized": len(seen), "unknown": unknown_count,
            "audio_activity": audio_activity, "recognition_error": err}, None

@app.route("/")
def index():
    with db() as c:
        students = c.execute("SELECT * FROM students ORDER BY name").fetchall()
        sessions = c.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 8").fetchall()
        total_attendance = c.execute("SELECT COUNT(*) n FROM attendance").fetchone()["n"]
    return render_template("index.html", students=students, sessions=sessions, total_attendance=total_attendance)

@app.route("/students/add", methods=["POST"])
def add_student():
    name = request.form.get("name","").strip()
    image = request.files.get("photo")
    if not name or not image or not image.filename:
        flash("Enter a student name and upload a face photo.", "error")
        return redirect(url_for("index"))
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in ALLOWED_IMAGE:
        flash("Photo must be JPG or PNG.", "error")
        return redirect(url_for("index"))
    filename = uuid.uuid4().hex + ext
    path = os.path.join(FACES, filename)
    image.save(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    found = face_cascade.detectMultiScale(img, 1.2, 5, minSize=(50,50)) if img is not None else []
    if len(found) == 0:
        os.remove(path)
        flash("No face detected in that photo. Try a clear, front-facing photo.", "error")
        return redirect(url_for("index"))
    with db() as c:
        c.execute("INSERT INTO students(name,image_file,created_at) VALUES(?,?,?)",
                  (name, filename, datetime.now().isoformat(timespec="seconds")))
    flash(f"Added {name}. For a demo, use one clear face photo per student.", "success")
    return redirect(url_for("index"))

@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files.get("video")
    if not video or not video.filename:
        flash("Choose a classroom video first.", "error")
        return redirect(url_for("index"))
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in ALLOWED_VIDEO:
        flash("Supported formats: MP4, AVI, MOV, MKV.", "error")
        return redirect(url_for("index"))
    filename = uuid.uuid4().hex + ext
    path = os.path.join(UPLOADS, filename)
    video.save(path)
    result, error = analyze_video(path)
    if error:
        flash(error, "error")
        return redirect(url_for("index"))
    return redirect(url_for("session_detail", session_id=result["id"]))

@app.route("/session/<int:session_id>")
def session_detail(session_id):
    with db() as c:
        session = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session: return "Session not found", 404
        attendance = c.execute("""SELECT a.*, s.name FROM attendance a JOIN students s ON s.id=a.student_id
                                  WHERE a.session_id=? ORDER BY s.name""", (session_id,)).fetchall()
    # Advice is explicitly heuristic, not a validated measure of attention or learning.
    suggestions = []
    if session["face_detections"] == 0:
        suggestions.append("No faces were detected in sampled frames. Check camera angle, lighting, and video quality.")
    else:
        suggestions.append("Review the timeline and validate detections manually before using this report for official attendance.")
        if session["face_detections"] / max(session["frames_analyzed"],1) < 1.0:
            suggestions.append("Few faces appeared in sampled frames. Consider a wider camera view and clear sightlines.")
    if session["audio_activity"] is None:
        suggestions.append("Audio activity was unavailable. Install FFmpeg and ensure the video contains an audio track.")
    elif session["audio_activity"] < 10:
        suggestions.append("Low audio activity was detected. This is not proof of low engagement; consider adding a discussion or question.")
    else:
        suggestions.append("Audio activity was detected. It does not distinguish teacher speech from student participation.")
    return render_template("session.html", session=session, attendance=attendance, suggestions=suggestions)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS, filename)

@app.route("/health")
def health():
    return {"status":"ok", "face_detection": bool(face_cascade and not face_cascade.empty()),
            "named_recognition_available": bool(hasattr(cv2, "face"))}

init_db()
if __name__ == "__main__":
    print("Local demo: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
