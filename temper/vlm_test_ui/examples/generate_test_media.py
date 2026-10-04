import os
import cv2
import numpy as np

def generate_test_video(path: str, duration_s: int = 12, fps: int = 24):
    """Generates a synthetic test video with a moving timestamp bar for UI verification."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w, h = 640, 360
    total_frames = duration_s * fps
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in range(total_frames):
        ts_s = i / fps
        frame = np.full((h, w, 3), (25, 35, 50), dtype=np.uint8)
        
        # Moving box to simulate motion
        box_x = int((i * 5) % (w - 80))
        cv2.rectangle(frame, (box_x, 150), (box_x + 80, 230), (0, 165, 255), -1)

        # Timestamp text
        text = f"Synthetic Scene: {ts_s:.2f}s / {duration_s}s"
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"[+] Generated synthetic test video at: {path}")

if __name__ == "__main__":
    generate_test_video(r"F:\URCA_PROJECTS\vlm_test_ui\examples\synthetic_test.mp4")
