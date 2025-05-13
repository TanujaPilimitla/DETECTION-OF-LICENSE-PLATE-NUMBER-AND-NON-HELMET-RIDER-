# DETECTION-OF-LICENSE-PLATE-NUMBER-AND-NON-HELMET-RIDER

 A desktop application built with Python and Tkinter that uses YOLO-based object detection and OCR to identify motorcycle riders, detect whether they are wearing helmets, and extract number plate information from images and videos.

 🚀 **Features**
 
- 📷 Upload and process images and videos

- 🤖 Detect motorcycle riders using YOLO

- 🪖 Detect presence/absence of helmets

- 🔍 Detect and extract number plates using YOLO and OCR

- 📝 Text extraction from plates using PaddleOCR

- 🖼️ Display results with annotations and plate images

- 📋 Scrollable view of detected number plates

- ⏹️ Option to stop video processing manually

🛠️ **Tech Stack**

- Language: Python 3.x

- GUI: Tkinter

- Object Detection: YOLOv5/YOLOv8

- OCR: PaddleOCR and Tesseract OCR

- Image Processing: OpenCV, PIL

🖥️**How It Works**

1. Upload an image or video.

2. Click Detect Rider to find motorcyclists in the image.

3. Click Detect Number Plate to:

   - Identify riders without helmets

   - Locate number plates

   - Perform OCR to extract plate text

4. Optionally, click Process Video to perform detection frame-by-frame.

5. Click Stop to end video processing early.

🧪 **Sample Output**

- Image with riders and helmets annotated

- Scrollable window showing detected plate images

- Extracted plate numbers shown in text box

📌 **Summary**
  
This project combines Computer Vision, Deep Learning, and OCR to build a practical and scalable Rider Detection System. By integrating these technologies into a GUI-based application, the solution is easy to operate for non-technical users and can be extended to real-time deployment with camera streams or edge devices.
