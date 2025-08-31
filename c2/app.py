import os
import time
import json
import threading
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv
import cv2

from detector import Detector

load_dotenv()

CAM_SOURCE_ENV = os.getenv("CAM_SOURCE", "0")
try:
    CAM_SOURCE = int(CAM_SOURCE_ENV)  # numeric -> webcam
except ValueError:
    CAM_SOURCE = CAM_SOURCE_ENV      # string path -> video file or RTSP

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
CONF_TH = float(os.getenv("CONF_TH", "0.35"))
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
SNAP_MIN_SECS = int(os.getenv("SNAP_MIN_SECS", "5"))

app = FastAPI(title="Exam Guard")

if os.path.isfile(SNAPSHOT_DIR):
    os.remove(SNAPSHOT_DIR)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

detector = Detector(
    cam_source=CAM_SOURCE,
    weights=YOLO_WEIGHTS,
    conf_th=CONF_TH,
    snapshot_dir=SNAPSHOT_DIR,
    snap_min_secs=SNAP_MIN_SECS
)

last_frame = None
last_incidents: List[Dict[str, Any]] = []
clients: List[WebSocket] = []

def camera_loop():
    global last_frame, last_incidents
    while True:
        try:
            annotated, incidents = detector.process()
            if annotated is not None:
                last_frame = annotated
                last_incidents = incidents
                if incidents:
                    payload = {"ts": int(time.time()), "incidents": incidents}
                    for ws in list(clients):
                        try:
                            ws.send_text(json.dumps(payload))
                        except:
                            clients.remove(ws)
            else:
                time.sleep(0.02)
        except Exception as e:
            print("[camera_loop error]", e)
            time.sleep(0.5)

threading.Thread(target=camera_loop, daemon=True).start()

# --- HTML dashboard ---
@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
  <title>Exam Guard Dashboard</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; background: #f5f5f5; margin: 20px; }
    h1 { color: #333; }
    #container { display: flex; gap: 20px; }
    #video { border: 2px solid #333; }
    #alerts { max-height: 500px; overflow-y: auto; background: #fff; padding: 10px; border: 1px solid #ccc; width: 320px;}
    li { margin-bottom: 5px; }
  </style>
</head>
<body>
<h1>Exam Guard</h1>
<div id="container">
  <div>
    <h3>Live Stream</h3>
    <img id="video" src="/video" width="640">
  </div>
  <div>
    <h3>Alerts</h3>
    <ul id="alerts"></ul>
  </div>
</div>

<script>
  const alertsEl = document.getElementById('alerts');
  const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws');
  ws.onmessage = (e) => {
      try {
          const msg = JSON.parse(e.data);
          if(msg.incidents && msg.incidents.length){
              msg.incidents.forEach(i=>{
                  const li = document.createElement('li');
                  li.innerHTML = '<strong>' + (i.type||'') + '</strong>: ' + (i.label||'') + (i.snap?(' — <a href="'+i.snap+'" target="_blank">snapshot</a>'):'');
                  alertsEl.prepend(li);
              });
          }
      } catch(err){ console.log(err) }
  };
</script>
</body>
</html>
"""
    return HTMLResponse(html)

@app.get("/video")
def video():
    boundary = "frame"
    def gen():
        while True:
            if last_frame is None:
                time.sleep(0.02)
                continue
            ok, jpg = cv2.imencode('.jpg', last_frame)
            if not ok:
                time.sleep(0.02)
                continue
            yield (b'--' + boundary.encode() + b'\r\n' +
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)

@app.post("/snapshot")
def manual_snapshot():
    global last_frame
    if last_frame is None:
        return JSONResponse({"message":"no frame available"}, status_code=404)
    ts = int(time.time())
    path = f"{SNAPSHOT_DIR}/manual_{ts}.jpg"
    cv2.imwrite(path, last_frame)
    detector.last_snapshot_path = path
    detector.snapshot_count += 1
    return {"message":"snapshot saved", "path": path}

@app.get("/status")
def status():
    return JSONResponse({
        "incidents": last_incidents,
        "snapshot_count": detector.snapshot_count,
        "last_snapshot": detector.get_last_snapshot()
    })

@app.on_event("shutdown")
def shutdown():
    try:
        detector.release()
    except:
        pass
