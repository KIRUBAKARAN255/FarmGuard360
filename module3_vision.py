import cv2
import csv
import os
import json
import time
import math
import random
import socket
import numpy as np
from datetime import datetime, timedelta

# =========================================================================
# CONFIGURATION & DATASET PATHS (Inside FarmGuard360_Final/)
# =========================================================================
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5050
TRAINING_DATASET_CSV = "farmguard360_1year_seasonal_dataset.csv"
ACTIVITY_CSV_FILE    = "module3_activity_log.csv"

DEFAULT_PROFILES = [
    {"name": "TEMPERATE_STD", "base_rest_sec": 8.0,  "tolerance": 1.25, "limit_sec": 10.0},
    {"name": "SUMMER_WARM",   "base_rest_sec": 10.0, "tolerance": 1.35, "limit_sec": 13.5},
    {"name": "WINTER_COOL",   "base_rest_sec": 7.0,  "tolerance": 1.20, "limit_sec": 8.4},
]

def generate_1year_dataset_if_missing(csv_path):
    """Automatically creates the 1,095-row 1-year seasonal dataset locally if not present."""
    if os.path.exists(csv_path):
        return
    random.seed(42)
    start_date = datetime(2025, 11, 1)
    shifts = [
        ("08:00:00", "Morning",   -2.5),
        ("14:00:00", "Afternoon", +4.0),
        ("18:00:00", "Evening",   -1.0),
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date", "Time", "Shift", "Month", "Season_Profile",
            "Ambient_Temp_C", "Humidity_Pct", "Zone",
            "Observed_Rest_Sec", "Learned_Baseline_Sec", "Adaptive_Limit_Sec",
            "Lethargy_Index", "Safety_Suit_Coverage_Pct", "Safety_Suit_OK", "Compliance_Status"
        ])
        for day_offset in range(365):
            dt = start_date + timedelta(days=day_offset)
            month = dt.month
            date_str = dt.strftime("%Y-%m-%d")

            if month in (3, 4, 5, 6):
                season, base_temp, base_hum, base_rest, tol = "SUMMER_WARM", 35.5, 48.0, 10.0, 1.35
            elif month in (11, 12, 1, 2):
                season, base_temp, base_hum, base_rest, tol = "WINTER_COOL", 23.5, 72.0, 7.0, 1.20
            else:
                season, base_temp, base_hum, base_rest, tol = "TEMPERATE_STD", 29.0, 62.0, 8.0, 1.25

            adaptive_limit = round(base_rest * tol, 2)
            for time_str, shift_name, temp_delta in shifts:
                ambient_temp = round(base_temp + temp_delta + random.gauss(0, 1.1), 1)
                humidity = round(max(32.0, min(92.0, base_hum + random.gauss(0, 4.5))), 1)
                if random.random() < 0.06:
                    obs_rest = round(adaptive_limit * random.uniform(1.05, 1.45), 2)
                else:
                    obs_rest = round(max(1.2, random.gauss(base_rest * 0.72, 1.6)), 2)

                lethargy_idx = round(min(1.0, obs_rest / adaptive_limit), 2)
                if random.random() < 0.08:
                    suit_cov = round(random.uniform(6.0, 27.5), 1)
                    suit_ok = "NO"
                else:
                    suit_cov = round(random.uniform(35.0, 78.0), 1)
                    suit_ok = "YES"

                if lethargy_idx >= 0.75 and suit_ok == "NO":
                    status = "DUAL_ALERT_LETHARGY_AND_SUIT"
                elif lethargy_idx >= 0.75:
                    status = "SICK_ANIMAL_ALERT"
                elif suit_ok == "NO":
                    status = "SAFETY_SUIT_BREACH"
                else:
                    status = "NORMAL"

                writer.writerow([
                    date_str, time_str, shift_name, dt.strftime("%b"), season,
                    ambient_temp, humidity, "SHIELD-3 (BIO-SHED)",
                    obs_rest, base_rest, adaptive_limit,
                    lethargy_idx, suit_cov, suit_ok, status
                ])
    print(f"[DATASET CREATED] Generated 1,095 seasonal records at '{os.path.abspath(csv_path)}'")

def load_dataset_calibration(csv_path):
    generate_1year_dataset_if_missing(csv_path)
    season_limits = {}
    compliant_coverages = []
    row_count = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            s_name = row.get("Season_Profile", "").strip()
            lim_val = float(row.get("Adaptive_Limit_Sec", 0) or 0)
            if s_name and lim_val > 0:
                season_limits.setdefault(s_name, []).append(lim_val)

            if row.get("Safety_Suit_OK", "").strip().upper() == "YES":
                cov_val = float(row.get("Safety_Suit_Coverage_Pct", 0) or 0)
                if cov_val > 0:
                    compliant_coverages.append(cov_val / 100.0)

    calibrated = []
    for prof in DEFAULT_PROFILES:
        name = prof["name"]
        learned_limit = round(sum(season_limits[name]) / len(season_limits[name]), 2) if name in season_limits else prof["limit_sec"]
        calibrated.append({
            "name": name,
            "base_rest_sec": prof["base_rest_sec"],
            "tolerance": prof["tolerance"],
            "limit_sec": learned_limit
        })

    learned_suit_thresh = round(max(0.25, min(compliant_coverages) - 0.03), 2) if compliant_coverages else 0.32
    print(f"[DATASET OK] Loaded {row_count} records from '{csv_path}'")
    for p in calibrated:
        print(f"             -> {p['name']}: Adaptive Rest Limit = {p['limit_sec']}s")
    print(f"             -> Calibrated Safety Suit Threshold = {int(learned_suit_thresh*100)}%")
    return calibrated, learned_suit_thresh, row_count

def check_safety_suit_hsv(roi_bgr, min_coverage):
    if roi_bgr is None or roi_bgr.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_white_blue = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([135, 75, 255]))
    mask_hi_vis     = cv2.inRange(hsv, np.array([12, 100, 120]), np.array([38, 255, 255]))
    combined_suit   = cv2.bitwise_or(mask_white_blue, mask_hi_vis)
    coverage = float(np.count_nonzero(combined_suit)) / float(combined_suit.size)
    return (coverage >= min_coverage), round(coverage, 2)

def push_to_dashboard(payload):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            s.connect((DASHBOARD_HOST, DASHBOARD_PORT))
            s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    except Exception:
        pass

def init_csv():
    if not os.path.exists(ACTIVITY_CSV_FILE):
        with open(ACTIVITY_CSV_FILE, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "Timestamp", "Zone", "Subject_ID", "Subject_Type",
                "Season_Profile", "Rest_Sec", "Limit_Sec",
                "Lethargy_Index", "Safety_Suit_OK", "Status"
            ])

def log_to_csv(sid, stype, season_name, rest_sec, limit_sec, lethargy, suit_ok, status):
    with open(ACTIVITY_CSV_FILE, mode="a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "SHIELD-3 (BIO-SHED)", f"SUBJ-{sid}", stype,
            season_name, round(rest_sec, 1), round(limit_sec, 1),
            round(lethargy, 2), "YES" if suit_ok else "NO", status
        ])

def load_cascade(filename):
    try:
        cascade_dir = getattr(cv2, "data", None)
        if cascade_dir is not None:
            xml_path = os.path.join(cascade_dir.haarcascades, filename)
            clf = cv2.CascadeClassifier(xml_path)
            if not clf.empty():
                return clf
    except Exception:
        pass
    return None

def boxes_overlap(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def main():
    seasonal_profiles, min_suit_cov, dataset_rows = load_dataset_calibration(TRAINING_DATASET_CSV)
    init_csv()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    upper_cascade = load_cascade("haarcascade_upperbody.xml")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=45, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    tracked_animals = {}
    next_id = 1
    season_idx = 0
    manual_suit_breach = False
    manual_lethargy_alert = False
    last_push = 0.0
    last_csv = 0.0

    print("[INFO] Keys: [L] Toggle Lethargy Alert | [P] Toggle Safety Suit Breach | [S] Switch Season | [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        season = seasonal_profiles[season_idx]
        rest_limit = season["limit_sec"]

        fg_mask = bg_sub.apply(frame, learningRate=0.002)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # USE 2: WORKER DETECTION & SAFETY SUIT VERIFICATION
        worker_regions = []
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(45, 45))
            for (fx, fy, fw, fh) in faces:
                wx1 = max(0, fx - int(fw * 1.10))
                wx2 = min(640, fx + fw + int(fw * 1.10))
                wy1 = max(72, fy - int(fh * 0.25))
                wy2 = 480
                tx1 = max(0, fx - int(fw * 0.65))
                tx2 = min(640, fx + fw + int(fw * 0.65))
                ty1 = min(470, fy + int(fh * 1.15))
                ty2 = min(480, ty1 + int(fh * 2.00))
                worker_regions.append(((wx1, wy1, wx2, wy2), (tx1, ty1, tx2, ty2)))

        if len(worker_regions) == 0 and upper_cascade is not None:
            uppers = upper_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(90, 90))
            for (ux, uy, uw, uh) in uppers:
                wx1, wy1, wx2, wy2 = max(0, ux), max(72, uy), min(640, ux + uw), 480
                tx1, tx2 = ux + int(uw * 0.20), ux + int(uw * 0.80)
                ty1, ty2 = min(470, uy + int(uh * 0.45)), min(480, uy + uh)
                worker_regions.append(((wx1, wy1, wx2, wy2), (tx1, ty1, tx2, ty2)))

        if len(worker_regions) == 0:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if area >= 14000 or h >= 175:
                    wx1, wy1 = max(0, x - 15), max(72, y)
                    wx2, wy2 = min(640, x + w + 15), min(480, y + h + 20)
                    tx1, tx2 = x + int(w * 0.20), x + int(w * 0.80)
                    ty1, ty2 = min(470, y + int(h * 0.38)), min(480, y + int(h * 0.90))
                    worker_regions.append(((wx1, wy1, wx2, wy2), (tx1, ty1, tx2, ty2)))

        worker_count = len(worker_regions)
        all_workers_suit_ok = True

        for (full_box, torso_box) in worker_regions:
            wx1, wy1, wx2, wy2 = full_box
            tx1, ty1, tx2, ty2 = torso_box
            torso_roi = frame[ty1:ty2, tx1:tx2]
            suit_ok, coverage = check_safety_suit_hsv(torso_roi, min_suit_cov)
            if manual_suit_breach:
                suit_ok = False

            if not suit_ok:
                all_workers_suit_ok = False
                box_color = (0, 0, 239)
                label = f"WORKER: NO SAFETY SUIT ({int(coverage*100)}%)"
            else:
                box_color = (16, 185, 129)
                label = f"WORKER: PROPER SAFETY SUIT ({int(coverage*100)}%)"

            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), box_color, 2)
            if ty2 > ty1 + 10 and tx2 > tx1 + 10:
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), box_color, 1)
            cv2.putText(frame, label, (max(8, wx1), max(90, wy1 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        if manual_suit_breach:
            all_workers_suit_ok = False

        # USE 1: ANIMAL ACTIVITY & LETHARGY MONITORING
        animal_count = 0
        max_lethargy = 0.08
        sick_animals = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2500 or area >= 14000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 175:
                continue
            cand_box = (x, y, x + w, y + h)
            if any(boxes_overlap(cand_box, wb) for (wb, _) in worker_regions):
                continue

            cx, cy = x + w // 2, y + h // 2
            animal_count += 1
            matched_id = None
            for sid, info in tracked_animals.items():
                if math.hypot(cx - info["cx"], cy - info["cy"]) < 55:
                    matched_id = sid
                    break

            if matched_id is None:
                matched_id = next_id
                next_id += 1
                tracked_animals[matched_id] = {"cx": cx, "cy": cy, "rest_start": now}
            else:
                dist_moved = math.hypot(cx - tracked_animals[matched_id]["cx"], cy - tracked_animals[matched_id]["cy"])
                tracked_animals[matched_id]["cx"] = cx
                tracked_animals[matched_id]["cy"] = cy
                if dist_moved > 12.0:
                    tracked_animals[matched_id]["rest_start"] = now

            rest_sec = now - tracked_animals[matched_id]["rest_start"]
            lethargy = min(1.0, round(rest_sec / max(rest_limit, 1.0), 2))
            if lethargy > max_lethargy:
                max_lethargy = lethargy
            if lethargy >= 0.75:
                sick_animals += 1

            a_color = (0, 0, 239) if lethargy >= 0.75 else (248, 189, 56)
            cv2.rectangle(frame, (x, y), (x + w, y + h), a_color, 2)
            cv2.putText(frame, f"ANIMAL-{matched_id} | Lethargy: {lethargy:.2f}",
                        (x, max(90, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, a_color, 2)

        if manual_lethargy_alert:
            max_lethargy = 0.92
            sick_animals = max(1, sick_animals)

        # TOP HUD
        cv2.rectangle(frame, (0, 0), (640, 72), (15, 23, 42), -1)
        hud1 = f"USE 1 (ANIMAL): Lethargy={max_lethargy:.2f} | {season['name']} (Limit {rest_limit:.1f}s)"
        if worker_count == 0 and not manual_suit_breach:
            hud2 = "USE 2 (WORKER PPE): NO WORKER IN FRAME (STANDBY)"
            hud2_color = (148, 163, 184)
        elif all_workers_suit_ok:
            hud2 = f"USE 2 (WORKER PPE): PROPER SAFETY SUIT ({worker_count} Worker)"
            hud2_color = (16, 185, 129)
        else:
            hud2 = "USE 2 (WORKER PPE): BREACH - NO SAFETY SUIT!"
            hud2_color = (0, 0, 239)
        hud3 = f"DATASET CALIBRATION: {dataset_rows} Seasonal Rows Loaded | Suit Thresh: {int(min_suit_cov*100)}%"

        cv2.putText(frame, hud1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1)
        cv2.putText(frame, hud2, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.50, hud2_color, 2)
        cv2.putText(frame, hud3, (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (250, 204, 21), 1)

        if now - last_csv >= 5.0:
            status_str = "ALERT" if (sick_animals > 0 or not all_workers_suit_ok) else "NORMAL"
            log_to_csv(0, "DUAL_MONITOR", season["name"], 0.0, rest_limit, max_lethargy, all_workers_suit_ok, status_str)
            last_csv = now

        if now - last_push >= 1.0:
            status_code = "NORMAL"
            if sick_animals > 0:
                status_code = "SICK_ANIMAL_ALERT"
            elif not all_workers_suit_ok:
                status_code = "SAFETY_SUIT_BREACH"

            push_to_dashboard({
                "source": "VISION",
                "season": season["name"],
                "ppe": all_workers_suit_ok,
                "persons": worker_count,
                "animals": animal_count,
                "sick_animals": sick_animals,
                "lethargy": round(max_lethargy, 2),
                "status": status_code
            })
            last_push = now

        cv2.imshow("FarmGuard 360 - Module 3 Dataset-Calibrated AI Vision", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("p"), ord("P")):
            manual_suit_breach = not manual_suit_breach
        elif key in (ord("l"), ord("L")):
            manual_lethargy_alert = not manual_lethargy_alert
        elif key in (ord("s"), ord("S")):
            season_idx = (season_idx + 1) % len(seasonal_profiles)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()