# Multimodal Attendance & Engagement Agent — local prototype

A starter project for a privacy-conscious classroom session analysis demo. It runs on the local computer and stores data in a local SQLite database.

## What works
- Enroll a learner using a clear face photo.
- Upload a short classroom video.
- Sample video at roughly one frame per second and detect faces using OpenCV.
- Attempt approximate LBPH face recognition against enrolled photos.
- Save attendance records with first/last observed timestamps and sampled appearances.
- Estimate audio activity from the audio energy in a video **if FFmpeg is installed**.
- Show a local report with cautious, rule-based instructor prompts.

## Important limitations
This is a prototype, not a production attendance system. A single photo per learner is not robust enough for reliable identification. Recognition can fail with different lighting, angles, distance, occlusion, or similar-looking faces. Audio energy is not speech recognition and cannot distinguish teacher speech from student participation. Visible faces or movement are not valid standalone measures of attention or learning. Do not use this system as the sole basis for grades, discipline, or official attendance. Obtain informed consent and follow applicable privacy, biometric-data, and institutional rules.

The "teaching suggestions" are simple heuristics, not a trained generative-AI agent. This keeps the prototype honest and runnable without external APIs. A later version could add a locally hosted language model, explicit opt-in student feedback, speech diarization, and validated evaluation.

## Setup (Windows)
1. Install Python 3.10 or 3.11.
2. Extract the ZIP and open the project folder in VS Code.
3. In the VS Code terminal, create a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Optional: install FFmpeg and add it to PATH to enable audio activity estimation.
6. Run:
   ```powershell
   python app.py
   ```
7. Open `http://127.0.0.1:5000` in your browser.

## Suggested first test
- Add 2–3 consenting test participants with clear, front-facing photos.
- Use a short, well-lit MP4 video showing those participants.
- Open the generated session report and manually check the results.

## Data and privacy
Uploaded videos and face photos are stored under `data/`; SQLite records are stored in `data/attendance.db`. Processing is local by default. Do not expose the Flask server to the internet. Delete the `data/` directory to remove local demo data. For real deployments, add authentication, role-based access, retention controls, encryption, audit logging, consent handling, and a documented deletion workflow.

## Project structure
```
multimodal_attendance_agent/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   └── session.html
├── static/
│   └── style.css
└── data/                 # created at runtime
    ├── faces/
    ├── uploads/
    └── attendance.db
```

## Troubleshooting
- **No recognized learners:** check that the photo has one visible face, lighting is even, and the person is visible in the video. LBPH uses a rough confidence threshold and needs real-world calibration.
- **`cv2.face` missing:** install the packages in `requirements.txt` inside the active virtual environment. Avoid installing `opencv-python` alongside `opencv-contrib-python`; they conflict.
- **Audio activity unavailable:** FFmpeg may not be installed or the video may not contain audio.
- **Video fails to open:** try converting it to MP4/H.264 or AVI.
