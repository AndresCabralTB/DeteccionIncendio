import cv2
import os
import numpy as np
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QLineEdit,  QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSlider, QSizePolicy, QDialog, QComboBox, QDialogButtonBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load all YOLOv8 models
        self.models = {
            "Nano": YOLO("/Users/andrescabral/Desktop/Prácticas /WildFire UI/Models/fire_n.pt"),
            "Small": YOLO("/Users/andrescabral/Desktop/Prácticas /WildFire UI/Models/fire_s.pt"),
            "Medium": YOLO("/Users/andrescabral/Desktop/Prácticas /WildFire UI/Models/fire_m.pt"),
            "Large": YOLO("/Users/andrescabral/Desktop/Prácticas /WildFire UI/Models/fire_l.pt"),
        }

       # Set default model
        self.model = self.models['Nano']

        # Initialize video and slider properties
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_counter = 0
        self.total_frames = 0
        self.video_length_slider_pressed = False  # To track when the user is dragging the slider

        # Add toggle for frame enhancement
        self.enhancement_enabled = False
        self.airflow_enabled = False

    def initUI(self):
        self.setWindowTitle("Detección de Incendios y Humo")
        self.adjustSize()

        # Main layout
        main_layout = QHBoxLayout()

        # Video area
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Seleccionar video")
        self.video_label.setFixedSize(800, 450)
        # Set the alignment to center
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Add border using setStyleSheet
        self.video_label.setStyleSheet("border: 2px solid black;")  # Example: 2px solid black border

        video_layout.addWidget(self.video_label)
        

        # Slider for video progress
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setValue(0)
        self.video_slider.setTickInterval(1)
        self.video_slider.sliderPressed.connect(self.pause_slider_update)
        self.video_slider.sliderReleased.connect(self.seek_video)

        video_layout.addWidget(self.video_slider)  # Add the slider below the video

        # Controls layout on the right
        controls_layout = QVBoxLayout()

        # Add airflow checkbox for airflow visualization
        self.airflow_checkbox = QCheckBox("Mostrar flujo de aire")
        self.airflow_checkbox.setChecked(False)
        self.airflow_checkbox.stateChanged.connect(self.toggle_airflow)
        controls_layout.addWidget(self.airflow_checkbox)

         # Enhancement toggle
        self.enhancement_checkbox = QCheckBox("Activar Estandarización")
        self.enhancement_checkbox.setChecked(False)
        self.enhancement_checkbox.stateChanged.connect(self.toggle_enhancement)
        controls_layout.addWidget(self.enhancement_checkbox)

        # Model Selector button
        self.model_button = self.create_button("Seleccionar Modelo")
        self.model_button.clicked.connect(self.open_model_selector)
        controls_layout.addWidget(self.model_button)

        # Upload video or image button
        self.upload_button = self.create_button("Seleccionar Video")
        self.upload_button.clicked.connect(self.open_file)
        controls_layout.addWidget(self.upload_button)

        # Threshold sliders
        self.iou_slider = self.create_slider("Threshold IOU", controls_layout, self.update_iou_threshold, 0, 100, 10, 50)
        self.confidence_slider = self.create_slider("Threshold Confidence", controls_layout, self.update_confidence_threshold, 0, 100, 10, 20)

        # FPS Slider
        self.fps_slider = self.create_slider("FPS", controls_layout, self.update_fps, 1, 60, 10, 30)

        # Brightness button
        #self.brightness_button = self.create_button("Brightness")
        #controls_layout.addWidget(self.brightness_button)

        # Start and Stop buttons
        start_stop_layout = QHBoxLayout()
        self.start_button = self.create_button("Inicar")
        self.start_button.clicked.connect(self.start_video)
        start_stop_layout.addWidget(self.start_button)

        self.stop_button = self.create_button("Pausa")
        self.stop_button.clicked.connect(self.stop_video)
        start_stop_layout.addWidget(self.stop_button)
        controls_layout.addLayout(start_stop_layout)

        # **Add Fire Report Button**
        self.report_button = self.create_button("Generar Reporte de Incendio")
        self.report_button.clicked.connect(self.open_fire_report_form)
        controls_layout.addWidget(self.report_button)

        # Add video and controls to the main layout
        main_layout.addLayout(video_layout)
        main_layout.addLayout(controls_layout)

        # Set the main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set default thresholds
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.2
    
    def toggle_airflow(self):
        """Toggle the airflow visualization on or off."""
        self.airflow_enabled = self.airflow_checkbox.isChecked()

    def create_button(self, text):
        """Helper function to create styled buttons"""
        button = QPushButton(text)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button.setFont(QFont("Arial", 12))
        return button

    def create_slider(self, label_text, layout, on_change, min_value, max_value, intervals, default_Value):
        """Helper function to create a slider with a numerical value on the right"""
        slider_layout = QVBoxLayout()  # Vertical layout for label and slider-value pair

        # Create label for slider description
        label = QLabel(label_text)

        # Create a horizontal layout for the slider and value label
        slider_and_value_layout = QHBoxLayout()

        # Create the slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_Value)  # Default value
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(intervals)

        # Create a label to show the current slider value
        value_label = QLabel(str(default_Value))  # Start with default value (50)

        # Connect slider value change to update the value label and on_change function
        slider.valueChanged.connect(on_change)
        slider.valueChanged.connect(lambda: value_label.setText(str(slider.value())))

        # Add the slider and value label to the horizontal layout
        slider_and_value_layout.addWidget(slider)      # Slider on the left
        slider_and_value_layout.addWidget(value_label)  # Value label on the right

        # Add the main label and the horizontal slider-value pair to the vertical layout
        slider_layout.addWidget(label)  # Label above the slider
        slider_layout.addLayout(slider_and_value_layout)  # Slider and value label side by side

        # Add the complete layout to the provided layout
        layout.addLayout(slider_layout)

        return slider

    def update_iou_threshold(self):
        """Update IOU threshold based on the slider value"""
        self.iou_threshold = self.iou_slider.value() / 100.0

    def update_confidence_threshold(self):
        """Update Confidence threshold based on the slider value"""
        self.confidence_threshold = self.confidence_slider.value() / 100.0

    def open_model_selector(self):
        """Open a dialog to select the model"""
        dialog = ModelSelectorDialog(self)
        if dialog.exec():
            selected_model = dialog.get_selected_model()
            if selected_model in self.models:
                self.model = self.models[selected_model]

    def toggle_enhancement(self):
        """Toggle the enhancement feature on or off."""
        self.enhancement_enabled = self.enhancement_checkbox.isChecked()

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image or Video", filter="Video Files (*.mp4 *.mov);;Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            if file_name.endswith(('.mp4', '.mov')):
                self.load_video(file_name)
            else:
                self.run_object_detection(file_name)

    def load_video(self, file_name):
        """Load the video and prepare for playback"""
        self.cap = cv2.VideoCapture(file_name)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_slider.setMaximum(self.total_frames)  # Set slider range based on total frames

    def start_video(self):
        if self.cap is not None and not self.timer.isActive():
            self.update_fps()  # Update timer interval based on the current FPS slider value
            self.timer.start()  # Start the timer

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()

    def increase_orange_intensity(self, frame):
        """Enhances the orange areas in a video frame (e.g., fire)"""
        try:
            # Convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define a more focused range for detecting fire-like orange (in HSV space)
            lower_orange = np.array([5, 50, 50])     # Lower bound (reddish tones)
            upper_orange = np.array([25, 255, 255])  # Upper bound (yellowish-orange tones)

            # Create a mask for orange areas
            mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # Intensify the orange color
            hsv[:, :, 1] = np.where(mask > 0, np.clip(hsv[:, :, 1] * 2.5, 0, 255), hsv[:, :, 1])  # Increase saturation
            hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] * 1.8, 0, 255), hsv[:, :, 2])  # Increase brightness

            # Convert back to BGR (regular color format)
            enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            return enhanced_frame
        except Exception as e:
            print(f"Failed to enhance frame: {e}")
            return frame  # Return the original frame if there's an issue

    def update_frame(self):
        """Read the next frame from the video, enhance its colors, and process it"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_counter += 1
                if not self.video_length_slider_pressed:
                    self.video_slider.setValue(self.frame_counter)

                    # Enhance the frame's orange areas only if the toggle is enabled
                if self.enhancement_enabled:
                    frame = self.increase_orange_intensity(frame)

                # **Airflow Visualization**: Apply optical flow if the checkbox is checked
                if self.airflow_enabled:
                    frame = self.visualize_airflow(frame)

                # Display the processed frame in the UI
                self.display_results(frame)
            else:
                self.stop_video()  # Stop video if it reaches the end

    def visualize_airflow(self, frame):
        """Visualize airflow using dense optical flow."""
        try:
            # Convert to grayscale for optical flow calculation
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Initialize the previous frame for the first iteration
            if not hasattr(self, 'prev_gray_frame'):
                self.prev_gray_frame = gray_frame
                return frame  # Return the original frame as there's no previous frame to compare

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Get the flow vectors (dx, dy)
            h, w = gray_frame.shape[:2]
            step = 20  # Define step size to place arrows

            for y in range(0, h, step):
                for x in range(0, w, step):
                    # Get the flow vector at the position (x, y)
                    dx, dy = flow[y, x]

                    # Draw an arrow for the flow
                    cv2.arrowedLine(
                        frame,
                        (x, y),
                        (int(x + dx), int(y + dy)),
                        (0, 255, 0), 2, tipLength=0.5
                    )

            # Update the previous frame
            self.prev_gray_frame = gray_frame

            return frame  # Return the frame with airflow arrows
        except Exception as e:
            print(f"Failed to visualize airflow: {e}")
            return frame  # Return original frame if there's an issue

    def display_results(self, img):
            """Convert the detected frame to QImage and display it"""
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            q_img = QImage(rgb_image.data, width, height, 3 * width, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
    
    def pause_slider_update(self):
        """Pause video updates when the slider is being dragged"""
        self.video_length_slider_pressed = True
        self.timer.stop()

    def update_fps(self):
        """Update the timer interval based on the FPS slider value."""
        fps = self.fps_slider.value()
        self.timer.setInterval(1000 // fps)  # Set interval in milliseconds based on FPS value

    def seek_video(self):
        """Seek the video to the frame corresponding to the slider position"""
        self.video_length_slider_pressed = False
        slider_value = self.video_slider.value()
        self.frame_counter = slider_value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, slider_value)
        self.start_video()

    def run_object_detection(self, file_path):
        """Run detection on a single image file"""
        img = cv2.imread(file_path)
        results = self.model.predict(img, conf=self.confidence_threshold, iou=self.iou_threshold)
        result_img = results[0].plot()
        self.display_results(result_img)

    def open_fire_report_form(self):
        """Open the form to report wildfire details"""
        form = FireReportForm(self)
        form.exec()

class ModelSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Model")
        self.setFixedSize(300, 100)

        # Layout
        layout = QVBoxLayout()

        # ComboBox for model selection
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(['Nano', 'Small', 'Medium', 'Large'])
        layout.addWidget(self.combo_box)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_model(self):
        """Get the selected model name from the combo box"""
        return self.combo_box.currentText()
    
# Placeholder for sending SMS
# Placeholder for sending SMS
def send_sms(location, area, height, municipio, estado, localidad, fire_type, people_around):
    message = f"""
    🚨 Alerta de Incendio
    Localización: {location}, {localidad}, {municipio}, {estado}
    Área afectada: {area} m²
    Altura de las llamas: {height} m
    Tipo de incendio: {fire_type}
    Precaución: Se recomienda evitar la zona.
    Estado de personas: {'Hay' if people_around == 'Sí' else 'No hay'} personas alrededor.

    Si te encuentras en las cercanías, por favor mantén la calma y sigue las indicaciones de seguridad locales.
    """
    print(f"Sending SMS with message:\n{message}")
    # Replace the print statement with actual SMS API integration (e.g., Twilio)


class FireReportForm(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reporte de Incendio")
        self.setFixedSize(400, 575)  # Ajustar tamaño de ventana para incluir nuevos campos

        layout = QVBoxLayout()

        # Localización del incendio Label
        self.location_label = QLabel("Localización del incendio")
        layout.addWidget(self.location_label)
        # Localización del incendio TextField
        self.location_input = QLineEdit(self)
        self.location_input.setPlaceholderText("Localización del incendio")
        layout.addWidget(self.location_input)

        # Área del incendio Label
        self.area_label = QLabel("Área del incendio (m²)")
        layout.addWidget(self.area_label)
        # Área del incendio TextField
        self.area_input = QLineEdit(self)
        self.area_input.setPlaceholderText("Área del incendio (m²)")
        layout.addWidget(self.area_input)

        # Altura del incendio Label
        self.altura_label = QLabel("Altura del incendio (m)")
        layout.addWidget(self.altura_label)
        # Altura del incendio
        self.height_input = QLineEdit(self)
        self.height_input.setPlaceholderText("Altura del incendio (m)")
        layout.addWidget(self.height_input)

        # **Municipio Label**
        self.municipio_label = QLabel("Municipio")
        layout.addWidget(self.municipio_label)
        # **Municipio TextField**
        self.municipio_input = QLineEdit(self)
        self.municipio_input.setPlaceholderText("Municipio")
        layout.addWidget(self.municipio_input)

        # **Estado Label**
        self.estado_label = QLabel("Estado")
        layout.addWidget(self.estado_label)
        # **Estado TextField**
        self.estado_input = QLineEdit(self)
        self.estado_input.setPlaceholderText("Estado")
        layout.addWidget(self.estado_input)

        # **Localidad Label**
        self.localidad_label = QLabel("Localidad")
        layout.addWidget(self.localidad_label)
        # **Localidad TextField**
        self.localidad_input = QLineEdit(self)
        self.localidad_input.setPlaceholderText("Localidad")
        layout.addWidget(self.localidad_input)

        # Incendio controlado Label
        self.tipo_label = QLabel("Indique tipo de incendio")
        layout.addWidget(self.tipo_label)
        # Tipo de incendio: Controlado o No Controlado
        self.fire_type_combo = QComboBox(self)
        self.fire_type_combo.addItems(["Controlado", "No Controlado"])
        layout.addWidget(self.fire_type_combo)

        # Incendio controlado Label
        self.personas_label = QLabel("Hay personas alrededor?")
        layout.addWidget(self.personas_label)
        # Hay personas alrededor: Sí o No
        self.people_around_combo = QComboBox(self)
        self.people_around_combo.addItems(["Sí", "No"])
        layout.addWidget(self.people_around_combo)

        # Send button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.send_report)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def send_report(self):
        """Send SMS with the filled report data."""
        location = self.location_input.text()
        area = self.area_input.text()
        height = self.height_input.text()
        municipio = self.municipio_input.text()
        estado = self.estado_input.text()
        localidad = self.localidad_input.text()
        fire_type = self.fire_type_combo.currentText()
        people_around = self.people_around_combo.currentText()

        # Validación de campos requeridos
        if not location or not area or not height or not municipio or not estado or not localidad:
            QMessageBox.warning(self, "Error", "Por favor, rellena todos los campos.")
            return

        send_sms(location, area, height, municipio, estado, localidad, fire_type, people_around)
        QMessageBox.information(self, "Enviado", "El reporte ha sido enviado exitosamente.")
        self.accept()


if __name__ == '__main__':
    app = QApplication([])
    window = ObjectDetectionApp()
    window.show()
    app.exec()