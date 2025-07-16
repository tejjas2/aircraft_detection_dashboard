import cv2
import numpy as np
from ultralytics import YOLO
import datetime

def detect_vessel_dimensions_from_bytes_aircraft(
    image_bytes: bytes,
    category_to_detect: str = "all",
    pixels_per_meter: float = 3.7,
    confidence_threshold: float = 0.25,
    is_aircraft: bool = True
) -> list:
    try:
        if is_aircraft:
            model = YOLO("core/aircraft.pt")
            class_thresholds = {
                "cargo": 0.7,
                "passenger": 0.7,
                "a10":0.5,
                "f14":0.5,
                "f15":0.5,
                "f16":0.5,
                "f22":0.5,
                "f4":0.5,
                "hawkt1":0.5,
                "mirage":0.5
            }
        else:
            print("Warning: Using naval model due to bad argument input")
            model = YOLO("core/best (4).pt")
            class_thresholds = {}
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not decode image from bytes.")
        return []

    class_names = model.names
    yolo_results = model.predict(source=img, conf=confidence_threshold)
    temp_detections = []

    for result in yolo_results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            current_class_name = class_names[cls_id]
            clean_class_name = current_class_name.split()[-1].lower()
            conf = float(boxes.conf[i].item())

            print(f" Detected class: {current_class_name}, confidence: {conf:.2f}")

            
            if is_aircraft:
                threshold = class_thresholds.get(clean_class_name, 1.0)
                if conf < threshold:
                    print(f" Skipping {clean_class_name}: below threshold {threshold}")
                    continue
            else:
                if conf < confidence_threshold:
                    continue

            if category_to_detect != "all" and clean_class_name != category_to_detect.lower():
                continue

            xywh = boxes.xywh[i].tolist()
            x_center, y_center, box_width, box_height = xywh
            #length_pixels, width_pixels = (box_width, box_height) if box_width > box_height else (box_height, box_width)
            #length_meters = round(length_pixels / pixels_per_meter, 2)
            #width_meters = round(width_pixels / pixels_per_meter, 2)
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            temp_detections.append({
                "category": clean_class_name,
                #"length_meters": length_meters,
                #"width_meters": width_meters,
                "confidence": round(conf, 2),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    if not temp_detections:
        print("No detections found.")
        return []
    display_class_names = {
    "f14": "F-14",
    "f15": "F-15",
    "f16": "F-16",
    "f22": "F-22",
    "f4": "F-4",
    "a10": "A-10",
    "mirage": "Mirage",
    "hawkt1": "Hawk T1",
    "cargo": "Cargo",
    "passenger": "Passenger"
}

    annotated_image = img.copy()
    for det in temp_detections:
        box = det['bounding_box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        display_name = display_class_names.get(det['category'], det['category'])
        label = f"{display_name}  C:{det['confidence']:.2f}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y_pos = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
        cv2.rectangle(annotated_image,
                  (x1, label_y_pos - text_height - 5),
                  (x1 + text_width, label_y_pos + baseline),
                  (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, label_y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    
    is_success, buffer = cv2.imencode(".jpg", annotated_image)
    if not is_success:
        print("Error: Failed to encode annotated image.")
        return []

    annotated_image_bytes = buffer.tobytes()
    final_results = []
    for det in temp_detections:
        final_results.append({
            'annotated_image': annotated_image_bytes,
            'detected_category': det['category'],
            'confidence': det['confidence'],
            'metadata': {
                'image_path': 'in-memory',
                'image_dimensions': f"{img.shape[1]}x{img.shape[0]}",
                'processing_timestamp_utc': datetime.datetime.utcnow().isoformat() + "Z",
                'custom_info': 'Detection Result'
            },
            #'length_meters': det['length_meters'],
            #'width_meters': det['width_meters'],
            'bounding_box': det['bounding_box']
        })

    return final_results

  