import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO  # Assuming you're using YOLO models
import pytesseract  # For OCR
from paddleocr import PaddleOCR

# Set the path to your Tesseract-OCR executable (update this based on your installation)
pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'
class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rider Detection System")
        self.root.geometry("800x600")

        # Load YOLO models
        self.rider_model = YOLO(r'yolov5models\rider.pt')  # Rider detection model
        self.helmet_model = YOLO(r'yolov5models\HelmetOrNo.pt')  # Helmet detection model
        self.plate_model = YOLO(r'yolov5models\NumberPlate.pt')  # Number plate detection model

        # Variables
        self.image_path = None
        self.video_path = None
        self.current_image = None
        self.is_processing = False  # Flag to control video processing
        self.detected_plates = []  # Store detected plates
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR# GUI Components
        self.create_widgets()

    def create_widgets(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        self.upload_img_btn = tk.Button(button_frame, text="Upload Image", command=self.upload_image)
        self.upload_img_btn.grid(row=0, column=0, padx=5)
        self.upload_vid_btn = tk.Button(button_frame, text="Upload Video", command=self.upload_video)
        self.upload_vid_btn.grid(row=0, column=1, padx=5)
        self.detect_rider_btn = tk.Button(button_frame, text="Detect Rider", command=self.detect_rider)
        self.detect_rider_btn.grid(row=0, column=2, padx=5)
        self.detect_plate_btn = tk.Button(button_frame, text="Detect Number Plate", command=self.detect_number_plate)
        self.detect_plate_btn.grid(row=0, column=3, padx=5)
        self.process_video_btn = tk.Button(button_frame, text="Process Video", command=self.process_video)
        self.process_video_btn.grid(row=0, column=4, padx=5)
        self.stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=5, padx=5)

        self.display_label = tk.Label(self.root)
        self.display_label.pack(pady=10)
        self.result_text = tk.Text(self.root, height=4, width=50)
        self.result_text.pack(pady=10)
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.image_path:
            if hasattr(self, 'plate_frame'):
                self.plate_frame.destroy()
            self.detected_plates = []
            image = Image.open(self.image_path)
            image = image.resize((500, 400), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(image)
            self.display_label.configure(image=self.current_image)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Image uploaded successfully\n")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            # Show the first frame as thumbnail
            if hasattr(self, 'plate_frame'):
                self.plate_frame.destroy()
            self.detected_plates = []
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image = pil_image.resize((500, 400), Image.Resampling.LANCZOS)
                self.current_image = ImageTk.PhotoImage(pil_image)
                self.display_label.configure(image=self.current_image)
            cap.release()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Video uploaded successfully - Showing thumbnail\n")
    def detect_rider(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first")
            return

        image = cv2.imread(self.image_path)
        results = self.rider_model(image)
        
        annotated_image = results[0].plot()
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image = pil_image.resize((500, 400), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(pil_image)
        self.display_label.configure(image=self.current_image)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Rider detection completed\n")
    def extract_text_from_plate(self, image, plate_box):
        """Extract text from the detected number plate region using PaddleOCR."""
        x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
        plate_region = image[y1:y2, x1:x2]

        # Perform OCR
        result = self.ocr.ocr(plate_region, cls=True)
        text = ""
        if result and result[0]:
            for line in result[0]:
                text += line[1][0] + " "  # Extract text from each detected line
        return text.strip(),plate_region
    def stop_processing(self):
        self.is_processing = False
        self.stop_btn.config(state=tk.DISABLED)
        self.show_detected_plates()

    def show_detected_plates(self):
        # Clear previous plate display if any
        if hasattr(self, 'plate_frame'):
            self.plate_frame.destroy()

        # Create a scrollable canvas for detected plates
        self.plate_frame = tk.Frame(self.root)
        self.plate_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(self.plate_frame, height=200)  # Fixed height for the scrollable area
        scrollbar = tk.Scrollbar(self.plate_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        scrollable_frame = tk.Frame(canvas)

        canvas.configure(xscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        if not self.detected_plates:
            self.result_text.insert(tk.END, "No number plates detected\n")
            return

        self.result_text.insert(tk.END, "Showing all detected number plates:\n")
        for idx, (text, plate_region) in enumerate(self.detected_plates):
            plate_rgb = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)
            pil_plate = Image.fromarray(plate_rgb)
            pil_plate = pil_plate.resize((200, 100), Image.Resampling.LANCZOS)
            plate_image = ImageTk.PhotoImage(pil_plate)
            
            # Create a frame for each plate (image + text)
            plate_container = tk.Frame(scrollable_frame)
            plate_container.grid(row=0, column=idx, padx=5)

            plate_label = tk.Label(plate_container, image=plate_image)
            plate_label.image = plate_image  # Keep reference
            plate_label.pack()

            text_label = tk.Label(plate_container, text=f"Plate {idx + 1}: {text}")
            text_label.pack()

            self.result_text.insert(tk.END, f"Plate {idx + 1}: {text}\n")

        # Update scroll region after adding all plates
        scrollable_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    def detect_number_plate(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first")
            return

        image = cv2.imread(self.image_path)
        rider_results = self.rider_model(image)
        if len(rider_results[0].boxes) == 0:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "No riders detected in the image\n")
            return

        self.result_text.delete(1.0, tk.END)
        self.detected_plates = []  # Reset for image processing
        all_results = ""

        for idx, rider_box in enumerate(rider_results[0].boxes):
            x1, y1, x2, y2 = map(int, rider_box.xyxy[0])
            rider_sub_image = image[y1:y2, x1:x2]

            helmet_results = self.helmet_model(rider_sub_image)
            has_helmet = False
            for r in helmet_results[0].boxes:
                if r.cls == 0:
                    has_helmet = True
                    break

            if not has_helmet:
                plate_results = self.plate_model(rider_sub_image)
                plate_text = ""
                
                if len(plate_results[0].boxes) > 0:
                    print(plate_results[0].boxes)
                    for box in plate_results[0].boxes:
                        text, plate_region = self.extract_text_from_plate(rider_sub_image, box)
                        if text:
                            plate_text += f"Rider {idx + 1} - Detected Plate: {text}\n"
                        self.detected_plates.append((text, plate_region))
                
                # Show the rider sub-image during processing
                # rider_rgb = cv2.cvtColor(rider_sub_image, cv2.COLOR_BGR2RGB)
                # pil_rider = Image.fromarray(rider_rgb)
                # pil_rider = pil_rider.resize((500, 400), Image.Resampling.LANCZOS)
                # self.current_image = ImageTk.PhotoImage(pil_rider)
                # self.display_label.configure(image=self.current_image)

                all_results += f"Rider {idx + 1} detected - No helmet found\n"
                if plate_text:
                    all_results += plate_text
                else:
                    all_results += f"Rider {idx + 1} - No readable number plate found\n"
            else:
                # If helmet is detected, show the rider sub-image
                rider_rgb = cv2.cvtColor(rider_sub_image, cv2.COLOR_BGR2RGB)
                pil_rider = Image.fromarray(rider_rgb)
                pil_rider = pil_rider.resize((500, 400), Image.Resampling.LANCZOS)
                # self.current_image = ImageTk.PhotoImage(pil_rider)
                # self.display_label.configure(image=self.current_image)
                all_results += "Rider detected - Helmet found, no number plate processing needed\n"

        self.result_text.insert(tk.END, all_results)
        self.show_detected_plates()
    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first")
            return

        self.is_processing = True
        self.stop_btn.config(state=tk.NORMAL)
        self.detected_plates = []  # Reset detected plates
        self.result_text.delete(1.0, tk.END)

        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened() and self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break

            rider_results = self.rider_model(frame)
            if len(rider_results[0].boxes) > 0:
                helmet_results = self.helmet_model(frame)
                has_helmet = False
                for r in helmet_results[0].boxes:
                    if r.cls == 0:  # Assuming 0 is 'with helmet'
                        has_helmet = True
                        break

                if not has_helmet:
                    plate_results = self.plate_model(frame)
                    annotated_frame = plate_results[0].plot()

                    plate_text = ""
                    if len(plate_results[0].boxes) > 0:
                        for box in plate_results[0].boxes:
                            text, plate_region = self.extract_text_from_plate(frame, box)
                            if text:
                                plate_text += f"Detected Plate: {text}\n"
                                self.detected_plates.append((text, plate_region))
                else:
                    annotated_frame = helmet_results[0].plot()
            else:
                annotated_frame = frame

            # Display frame
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.resize((500, 400), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(pil_image)
            self.display_label.configure(image=self.current_image)
            self.root.update()

            # Update result text for this frame
            if has_helmet:
                self.result_text.insert(tk.END, "Helmet detected\n")
            elif plate_text:
                self.result_text.insert(tk.END, "No helmet detected - Showing number plate\n")
                self.result_text.insert(tk.END, plate_text)
            else:
                self.result_text.insert(tk.END, "No helmet detected - No readable plate found\n")

        cap.release()
        self.is_processing = False
        self.stop_btn.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Video processing completed\n")
        self.show_detected_plates()
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()