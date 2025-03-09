from ultralytics import YOLO
import cv2
import numpy as np
import win32gui
import win32ui
import win32con


model = YOLO('trained_model/fairy_fbS.pt')


class WindowCapture:
    def __init__(self, hwnd):
        self.hwnd = hwnd

    def capture_win(self):
        """Захватываем окно приложения."""
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bot - top

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        save_dc.BitBlt(
            (0, 0),
            (width, height),
            mfc_dc,
            (0, 0),
            win32con.SRCCOPY,
        )

        bmp_info = bitmap.GetInfo()
        bmp_str = bitmap.GetBitmapBits(True)
        img = np.frombuffer(
            bmp_str,
            dtype=np.uint8,
        ).reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))

        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

hwnd = win32gui.FindWindow(None, "Warspear Online")
capture = WindowCapture(hwnd)

while True:
    frame = capture.capture_win()
    results = model(frame, conf=0.6)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv5 Detection", annotated_frame)

    # Нажмите 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
