"""
Modern Qt UI for Face Recognition Attendance System
With Pi Camera Support for Raspberry Pi
"""
# ===== MUST BE FIRST: prevent DLL conflicts between cv2, ONNX Runtime, OpenMP, and PyQt5 =====
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = 'CPUExecutionProvider'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# ==================================================================================
import sys
from core import AttendanceSystem, Config

import cv2
import numpy as np
import glob
import pandas as pd
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog, QPrintPreviewDialog




try:
    from hardware_controller import HardwareController
    HARDWARE_AVAILABLE = True
    print("✅ Hardware controller imported successfully")
except ImportError:
    HARDWARE_AVAILABLE = False
    print("⚠️ Hardware controller not available - running without sensor/IR LED")
except Exception as e:
    HARDWARE_AVAILABLE = False
    print(f"⚠️ Hardware controller error: {e}")



# ========== TRY TO IMPORT PICAMERA2 FOR RASPBERRY PI ==========
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA_AVAILABLE = True
    print(" picamera2 loaded successfully - Pi Camera supported")
except ImportError:
    PICAMERA_AVAILABLE = False
    print("ℹ picamera2 not available - using OpenCV for camera")

# ========== MODERN THEME STYLESHEET ==========
MODERN_STYLE = """
/* Global Styles */
QMainWindow {
    background-color: #0f172a;
}

QWidget {
    background-color: #1e293b;
    color: #f8fafc;
    font-family: 'Segoe UI', 'Arial', sans-serif;
}

/* Modern Cards */
QFrame.card {
    background-color: #1e293b;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 10px;
}

QFrame.card:hover {
    border: 1px solid #8b5cf6;
    background-color: #2d3a4f;
}

/* Buttons */
QPushButton {
    background-color: #8b5cf6;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: bold;
    font-size: 13px;
    min-width: 120px;
    min-height: 38px;
}

QPushButton:hover {
    background-color: #7c3aed;
}

QPushButton:pressed {
    background-color: #6d28d9;
}

QPushButton.menuButton {
    font-size: 18px;
    padding: 24px 18px;
    min-width: 180px;
    min-height: 120px;
    background-color: #334155;
}

QPushButton.menuButton:hover {
    background-color: #8b5cf6;
}

QPushButton#dangerButton {
    background-color: #ef4444;
}

QPushButton#dangerButton:hover {
    background-color: #dc2626;
}

QPushButton#successButton {
    background-color: #10b981;
}

QPushButton#successButton:hover {
    background-color: #059669;
}

/* Labels */
QLabel {
    color: #f8fafc;
    font-size: 14px;
}

QLabel.headerLabel {
    font-size: 22px;
    font-weight: bold;
    color: #8b5cf6;
    padding: 10px;
}

QLabel.cardTitle {
    font-size: 16px;
    font-weight: bold;
    color: #8b5cf6;
    padding: 5px;
}

QLabel.timeLabel {
    font-size: 34px;
    font-weight: bold;
    color: #06b6d4;
}

QLabel.dateLabel {
    font-size: 14px;
    color: #cbd5e1;
}

/* Combo Box - Dark Theme */
QComboBox {
    background-color: #334155;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 10px 14px;
    color: white;
    min-width: 150px;
    min-height: 36px;
    font-size: 14px;
    selection-background-color: #8b5cf6;
    selection-color: white;
}

QComboBox:hover {
    border: 1px solid #8b5cf6;
    background-color: #3e4a5e;
}

QComboBox:focus {
    border: 2px solid #8b5cf6;
    background-color: #3e4a5e;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
    subcontrol-origin: padding;
    subcontrol-position: right center;
    background-color: transparent;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #cbd5e1;
    margin-right: 8px;
}

QComboBox::down-arrow:hover {
    border-top: 5px solid #8b5cf6;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: white;
    border: 2px solid #475569;
    border-radius: 4px;
    selection-background-color: #8b5cf6;
    selection-color: white;
    outline: none;
    padding: 4px;
}

QComboBox QAbstractItemView::item {
    padding: 10px 14px;
    min-height: 32px;
    border-radius: 2px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #334155;
    color: white;
}

QComboBox QAbstractItemView::item:selected {
    background-color: #8b5cf6;
    color: white;
}

/* Class Card */
.class-card {
    background-color: #334155;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 15px;
    margin: 5px;
}

.class-card:hover {
    border: 1px solid #8b5cf6;
    background-color: #3e4a5e;
}

.class-subject {
    font-size: 18px;
    font-weight: bold;
    color: #8b5cf6;
}

.class-teacher {
    font-size: 14px;
    color: #cbd5e1;
}

.class-time {
    font-size: 14px;
    color: #06b6d4;
    font-weight: bold;
}

.class-room {
    font-size: 14px;
    color: #f59e0b;
}

/* TreeView (Table) */
QTreeView {
    background-color: #334155;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 5px;
    alternate-background-color: #1e293b;
}

QTreeView::item {
    padding: 10px;
    border-bottom: 1px solid #475569;
    min-height: 32px;
}

QTreeView::item:selected {
    background-color: #8b5cf6;
}

QHeaderView::section {
    background-color: #8b5cf6;
    color: white;
    padding: 10px;
    font-size: 14px;
    border: none;
    font-weight: bold;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #1e293b;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #475569;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #8b5cf6;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #475569;
    border-radius: 4px;
    text-align: center;
    color: white;
    background-color: #1e293b;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #8b5cf6, stop:0.5 #ec4899, stop:1 #f59e0b);
    border-radius: 3px;
}

/* Line Edit */
QLineEdit {
    background-color: #334155;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 10px 12px;
    color: white;
    font-size: 14px;
    min-height: 36px;
}

QLineEdit:focus {
    border: 2px solid #8b5cf6;
}

/* Status Bar */
QStatusBar {
    background-color: #334155;
    color: #cbd5e1;
    font-size: 14px;
}

/* Group Box */
QGroupBox {
    color: #8b5cf6;
    font-weight: bold;
    font-size: 14px;
    border: 1px solid #475569;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
"""


# ========== RESPONSIVE UI HELPERS ==========

def get_available_screen_geometry():
    """Return the available screen geometry with a safe fallback."""
    app = QApplication.instance()
    screen = app.primaryScreen() if app else None
    if screen is None:
        return QRect(0, 0, 1280, 720)
    return screen.availableGeometry()


def responsive_size(target_w, target_h, min_w, min_h, max_w_ratio=0.92, max_h_ratio=0.88):
    """Scale a target size to the current screen while preserving proportions."""
    geo = get_available_screen_geometry()
    avail_w = max(min_w, int(geo.width() * max_w_ratio))
    avail_h = max(min_h, int(geo.height() * max_h_ratio))
    scale = min(avail_w / float(target_w), avail_h / float(target_h), 1.0)
    scale = max(0.78, scale)
    width = max(min_w, min(int(target_w * scale), avail_w))
    height = max(min_h, min(int(target_h * scale), avail_h))
    return width, height, scale, geo

# ========== PI CAMERA HANDLER CLASS ==========

class PiCameraHandler:
    """Robust handler for Raspberry Pi Camera"""
    
    def __init__(self):
        import time
        self.time = time
        self.picam2 = None
        self.initialized = False
        self.error_message = None
        
    def initialize(self):
        """Initialize the Pi Camera with proper sequence"""
        try:
            print(" Initializing Pi Camera...")
            
            # Make sure any previous instance is closed
            if self.picam2:
                self.close()
            
            # Create Picamera2 instance
            self.picam2 = Picamera2()
            
            # Configure camera
            camera_config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameDurationLimits": (33333, 33333)}  # Limit to 30fps
            )
            
            # Apply configuration
            self.picam2.configure(camera_config)
            print(" Camera configured")
            
            # Start camera with retry
            for attempt in range(3):
                try:
                    self.picam2.start()
                    print(f" Camera started on attempt {attempt + 1}")
                    self.time.sleep(1)
                    
                    # Test capture
                    test_frame = self.picam2.capture_array()
                    if test_frame is not None:
                        print(f" Test capture successful: {test_frame.shape}")
                        self.initialized = True
                        break
                    else:
                        print(f" Test capture returned None on attempt {attempt + 1}")
                        self.picam2.stop()
                        self.time.sleep(1)
                        
                except Exception as e:
                    print(f" Start attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        self.time.sleep(2)
                    else:
                        raise
            
            if not self.initialized:
                raise Exception("Camera failed to initialize after 3 attempts")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            print(f" Camera init error: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame with error handling"""
        if not self.initialized or not self.picam2:
            return None
        
        try:
            # Capture array (returns RGB)
            frame = self.picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV compatibility
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            print(f" Frame capture error: {e}")
            return None
    
    def restart_camera(self):
        """Restart the camera after error"""
        try:
            print(" Restarting camera...")
            if self.picam2:
                self.picam2.stop()
                self.time.sleep(1)
                self.picam2.start()
                self.time.sleep(1)
                print(" Camera restarted")
        except Exception as e:
            print(f" Restart failed: {e}")
            self.initialized = False
    
    def stop(self):
        """Stop the camera"""
        if self.picam2:
            try:
                self.picam2.stop()
                print(" Camera stopped")
            except:
                pass
    
    def close(self):
        """Completely close and release the camera safely"""
        if not self.picam2:
            self.initialized = False
            return

        try:
            print(" Closing camera...")
            try:
                self.picam2.stop()
            except Exception as e:
                print(f" Camera stop warning: {e}")

            self.time.sleep(0.2)

            try:
                self.picam2.close()
            except Exception as e:
                print(f" Camera close warning: {e}")

            print(" Camera closed")

        except Exception as e:
            print(f" Error closing camera: {e}")

        finally:
            self.picam2 = None
            self.initialized = False
            self.time.sleep(0.2)


# ========== MODERN CARD CLASS ==========

class ModernCard(QFrame):
    """Modern card widget with title and content"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumHeight(100)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title with accent bar
        title_container = QWidget()
        title_container.setStyleSheet("background-color: transparent;")
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 5)
        title_layout.setSpacing(5)
        
        # Accent bar
        accent_bar = QFrame()
        accent_bar.setFixedHeight(3)
        accent_bar.setFixedWidth(30)
        accent_bar.setStyleSheet(f"background-color: {Config.THEME['primary']}; border: none;")
        title_layout.addWidget(accent_bar, alignment=Qt.AlignLeft)
        
        # Title
        title_label = QLabel(title)
        title_label.setProperty("class", "cardTitle")
        title_layout.addWidget(title_label)
        
        layout.addWidget(title_container)
        
        # Content area
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: transparent;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.content_widget)
    
    def add_content(self, text):
        """Add content line to card"""
        label = QLabel(text)
        label.setStyleSheet(f"color: {Config.THEME['text_secondary']}; font-size: 12px;")
        self.content_layout.addWidget(label)
        return label
    
    def add_widget(self, widget):
        """Add custom widget to card"""
        self.content_layout.addWidget(widget)
        return widget

# ========== MODERN LOADING DIALOG ==========

class ModernLoadingDialog(QDialog):
    """Simple loading dialog with animation"""
    
    def __init__(self, title="Processing", message="Please wait...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.animation_angle = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        
        # Main container
        self.container = QFrame()
        self.container.setObjectName("loadingContainer")
        self.container.setStyleSheet("""
            QFrame#loadingContainer {
                background-color: #1e293b;
                border-radius: 24px;
                border: 1px solid #8b5cf6;
            }
        """)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #8b5cf6;")
        layout.addWidget(title_label)
        
        # Spinner
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setMinimumSize(80, 80)
        layout.addWidget(self.spinner_label)
        
        # Progress bar (optional - can be hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #334155;
                border-radius: 10px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #8b5cf6;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel(message)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #cbd5e1; font-size: 14px;")
        layout.addWidget(self.status_label)
        
        # Cancel button (optional)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(100)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)
        
        # Center on parent
        self.center_on_parent()
        
        self.start_animation()
    
    def center_on_parent(self):
        """Center the dialog on its parent"""
        if self.parent():
            parent_geo = self.parent().geometry()
            self.setGeometry(
                parent_geo.x() + (parent_geo.width() - 500) // 2,
                parent_geo.y() + (parent_geo.height() - 400) // 2,
                500, 400
            )
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            self.setGeometry(
                (screen.width() - 500) // 2,
                (screen.height() - 400) // 2,
                500, 400
            )
    
    def start_animation(self):
        """Start the spinner animation"""
        self.animation_timer.start(50)
    
    def update_animation(self):
        """Update the animated spinner"""
        self.animation_angle = (self.animation_angle + 10) % 360
        
        spinner = QPixmap(60, 60)
        spinner.fill(Qt.transparent)
        
        painter = QPainter(spinner)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen()
        pen.setWidth(4)
        pen.setCapStyle(Qt.RoundCap)
        
        gradient = QConicalGradient(30, 30, self.animation_angle)
        gradient.setColorAt(0.0, QColor(139, 92, 246))  # #8b5cf6
        gradient.setColorAt(0.5, QColor(236, 72, 153))  # #ec4899
        gradient.setColorAt(1.0, QColor(245, 158, 11))  # #f59e0b
        
        pen.setBrush(QBrush(gradient))
        painter.setPen(pen)
        
        rect = QRectF(8, 8, 44, 44)
        start_angle = self.animation_angle * 16
        span_angle = 270 * 16
        painter.drawArc(rect, start_angle, span_angle)
        
        painter.end()
        self.spinner_label.setPixmap(spinner)
    
    def update_progress(self, value, message=None):
        """Update progress bar and status message"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
        QApplication.processEvents()
    
    def set_message(self, message):
        """Update status message"""
        self.status_label.setText(message)
        QApplication.processEvents()
    
    def show_success(self, message, timeout=1500):
        """Show success state with checkmark"""
        self.animation_timer.stop()
        self.spinner_label.clear()
        
        # Draw checkmark
        checkmark = QPixmap(60, 60)
        checkmark.fill(Qt.transparent)
        
        painter = QPainter(checkmark)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor(16, 185, 129))  # #10b981
        painter.setPen(pen)
        
        path = QPainterPath()
        path.moveTo(15, 30)
        path.lineTo(25, 40)
        path.lineTo(45, 20)
        painter.drawPath(path)
        
        painter.end()
        
        self.spinner_label.setPixmap(checkmark)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #10b981; font-size: 14px;")
        
        # Auto-close after timeout
        QTimer.singleShot(timeout, self.accept)
    
    def show_error(self, message, timeout=2000):
        """Show error state with X mark"""
        self.animation_timer.stop()
        self.spinner_label.clear()
        
        # Draw X mark
        xmark = QPixmap(60, 60)
        xmark.fill(Qt.transparent)
        
        painter = QPainter(xmark)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor(239, 68, 68))  # #ef4444
        painter.setPen(pen)
        
        painter.drawLine(15, 15, 45, 45)
        painter.drawLine(45, 15, 15, 45)
        
        painter.end()
        
        self.spinner_label.setPixmap(xmark)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #ef4444; font-size: 14px;")
        
        # Auto-close after timeout
        QTimer.singleShot(timeout, self.reject)
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.animation_timer.stop()
        event.accept()

# ========== MODERN BUTTON CLASS ==========

class ModernButton(QPushButton):
    """Modern styled button"""
    
    def __init__(self, text, parent=None, button_type="primary"):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        
        if button_type == "danger":
            self.setObjectName("dangerButton")
        elif button_type == "success":
            self.setObjectName("successButton")


# ========== CLASS CARD FOR UPCOMING CLASSES ==========

class ClassCard(QFrame):
    """Card widget for displaying class information"""
    
    clicked = pyqtSignal(object)
    
    def __init__(self, class_data, parent=None):
        super().__init__(parent)
        self.class_data = class_data
        self.setFrameShape(QFrame.NoFrame)
        self.setProperty("class", "class-card")
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Subject
        subject_label = QLabel(class_data['subject'])
        subject_label.setProperty("class", "class-subject")
        layout.addWidget(subject_label)
        
        # Teacher
        teacher_label = QLabel(f" {class_data['teacher']}")
        teacher_label.setProperty("class", "class-teacher")
        layout.addWidget(teacher_label)
        
        # Time and Room
        info_layout = QHBoxLayout()
        
        time_label = QLabel(f" {class_data['start_time']}")
        time_label.setProperty("class", "class-time")
        info_layout.addWidget(time_label)
        
        room_label = QLabel(f" {class_data['classroom']}")
        room_label.setProperty("class", "class-room")
        info_layout.addWidget(room_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # Section
        section_label = QLabel(f" Section {class_data['section']}")
        section_label.setStyleSheet(f"color: {Config.THEME['text_secondary']}; font-size: 12px;")
        layout.addWidget(section_label)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.class_data)
        super().mousePressEvent(event)


# ========== FACE SCANNER DIALOG (Attendance) ==========

class FaceScannerDialog(QDialog):
    """Modern face scanner dialog for attendance - with Pi Camera support"""
    
    face_detected = pyqtSignal(str, str)  # name, status
    
    def __init__(self, system, hardware=None, parent=None, distance_fn=None):
        super().__init__(parent)
        self.system = system
        self.hardware = hardware
        self.distance_fn = distance_fn  # callable → returns last known distance or None
        self.camera_handler = None
        self.capture = None
        self.use_picamera = False
        self.use_hardware_camera = False
        self.timer = QTimer()
        self.scan_start_time = QDateTime.currentDateTime()
        self.processing = False
        self._camera_stopping = False
        
        self.init_ui()

        # Hardware is only for sensor + IR LED. The dialog always owns the camera.
        if self.hardware:
            self.hardware.manual_activate()

        self.use_hardware_camera = False
        self.start_camera()

        
    
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Face Scanner")
        self.setModal(True)
        self.setStyleSheet(MODERN_STYLE)

        dialog_w, dialog_h, self.ui_scale, _ = responsive_size(560, 560, 500, 460, 0.62, 0.88)
        self.video_target_w = max(430, min(520, dialog_w - 40))
        self.video_target_h = max(250, min(330, int(self.video_target_w * 0.63)))
        self.setMinimumSize(500, 460)
        self.resize(dialog_w, dialog_h)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(max(14, int(20 * self.ui_scale)), max(14, int(18 * self.ui_scale)), max(14, int(20 * self.ui_scale)), max(14, int(18 * self.ui_scale)))
        layout.setSpacing(max(10, int(14 * self.ui_scale)))
        
        # Title
        title_label = QLabel(" Face Recognition Scanner")
        title_label.setProperty("class", "headerLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #8b5cf6; padding: 8px;")
        layout.addWidget(title_label)
        
        # Video container
        video_container = QFrame()
        video_container.setFrameShape(QFrame.NoFrame)
        video_container.setFixedSize(self.video_target_w, self.video_target_h)
        video_container.setStyleSheet("background-color: black; border-radius: 10px;")
        
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(self.video_target_w, self.video_target_h)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        video_layout.addWidget(self.video_label)
        
        layout.addWidget(video_container, alignment=Qt.AlignCenter)
        
        # Status
        self.status_label = QLabel(" Scanning for faces...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {Config.THEME['text']}; font-size: 15px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Cancel button
        cancel_btn = ModernButton("Cancel", button_type="danger")
        cancel_btn.clicked.connect(self.cleanup_and_reject)
        layout.addWidget(cancel_btn)
        
        # Timer for progress
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(30)
    
    def start_camera(self):
        """Start camera capture with Pi Camera support"""
        self.camera_handler = None
        
        try:
            if PICAMERA_AVAILABLE:
                # Try Pi Camera first
                print(" [Attendance] Attempting to use Pi Camera...")
                self.camera_handler = PiCameraHandler()
                
                if self.camera_handler.initialize():
                    self.use_picamera = True
                    self.status_label.setText(" Camera ready")
                    print(" [Attendance] Using Pi Camera")
                else:
                    print(f" [Attendance] Pi Camera failed: {self.camera_handler.error_message}")
                    self.use_picamera = False
                    self.camera_handler = None
                    # Fall back to OpenCV
                    self.init_opencv_camera()
            else:
                self.use_picamera = False
                self.init_opencv_camera()
            
            # Start timer for frame updates
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            
        except Exception as e:
            print(f" [Attendance] Camera error: {e}")
            self.status_label.setText(f" Camera error: {str(e)[:30]}")
            self.status_label.setStyleSheet(f"color: {Config.THEME['error']};")
    
    def init_opencv_camera(self):
        """Fallback to OpenCV camera"""
        print(" [Attendance] Falling back to OpenCV camera...")
        
        # Try multiple indices
        camera_indices = [0, 1, 2, 3, 4]
        
        for idx in camera_indices:
            self.capture = cv2.VideoCapture(idx)
            if self.capture.isOpened():
                # Test frame read
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    print(f" [Attendance] OpenCV camera {idx} working")
                    # Set camera properties
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.capture.set(cv2.CAP_PROP_FPS, 30)
                    self.status_label.setText(f"Camera ready")
                    return
                else:
                    print(f" [Attendance] Camera {idx} opened but couldn't read frame")
                    self.capture.release()
        
        # If no camera works
        print(" [Attendance] No working OpenCV camera found")
        self.capture = None
        self.status_label.setText(" No camera found")
    
    def stop_camera(self):
        """Stop camera safely"""
        if getattr(self, "_camera_stopping", False):
            return
        self._camera_stopping = True

        if self.timer and self.timer.isActive():
            self.timer.stop()

        if self.progress_timer and self.progress_timer.isActive():
            self.progress_timer.stop()

        self.processing = True

        if self.use_picamera and self.camera_handler:
            try:
                self.camera_handler.close()
            except Exception as e:
                print(f" [Attendance] Camera close error: {e}")
            self.camera_handler = None
        elif hasattr(self, 'capture') and self.capture:
            try:
                self.capture.release()
            except Exception as e:
                print(f" [Attendance] OpenCV release error: {e}")
            self.capture = None

        if self.hardware:
            self.hardware.manual_deactivate()

        print(" [Attendance] Camera stopped safely")   
    
    def update_frame(self):
        """Update video frame"""
        if getattr(self, "_camera_stopping", False):
            return

        frame = None

        if self.camera_handler:
            frame = self.camera_handler.capture_frame()
        elif self.capture:
            ret, frame = self.capture.read()
            if not ret:
                frame = None
        
        if frame is None:
            return
        
        # Rest of frame processing remains the same
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Resize for display
        target_w, target_h = self.video_target_w, self.video_target_h
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Center the image
        display_image = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
        rgb_resized = cv2.resize(rgb_image, (new_w, new_h))
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        display_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = rgb_resized
        
        bytes_per_line = ch * target_w
        qt_image = QImage(display_image.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # ── Distance guard (inside scan) ──────────────────────────────────────
        # Check DURING the scan so a person can't bypass by stepping close
        # after the dialog opens.
        if self.distance_fn is not None:
            dist = self.distance_fn()
            if dist is not None and dist <= 0.3:
                if not self.processing:
                    self.process_result("TOO_CLOSE")
                return
        # ─────────────────────────────────────────────────────────────────────

        # Process face recognition
        if not self.processing:
            self.processing = True
            result = self.system.scan_face(frame)
            
            # Handle both old and new return formats
            if len(result) == 2:
                status, name = result
                is_live = True
                liveness_conf = 1.0
            else:
                status, name, is_live, liveness_conf = result
            
            if status == "FOUND" and is_live:
                self.process_result(name)
            elif status == "FOUND" and not is_live:
                self.process_result("SPOOF")
            elif status == "SPOOF":
                self.process_result("SPOOF")
            elif status == "UNKNOWN":
                self.process_result("NOT_IN_DB")
            elif (QDateTime.currentDateTime().toSecsSinceEpoch() - 
                self.scan_start_time.toSecsSinceEpoch()) >= 3:
                self.process_result(None)
            
            self.processing = False
            
    def update_progress(self):
        """Update progress bar"""
        elapsed = QDateTime.currentDateTime().toSecsSinceEpoch() - self.scan_start_time.toSecsSinceEpoch()
        progress = min(100, (elapsed / 3) * 100)
        self.progress_bar.setValue(int(progress))
    
    def process_result(self, name):
        """Process recognition result"""
        self.stop_camera()

        if name == "TOO_CLOSE":
            self.status_label.setText("⚠️ Too Close! Please stand ~1m away from the sensor")
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 14px; font-weight: bold;")
            QApplication.beep()
        elif name == "SPOOF":
            self.status_label.setText("⚠️ SPOOF DETECTED! - Photo or Screen Detected")
            self.status_label.setStyleSheet("color: #ef4444; font-size: 14px; font-weight: bold;")
            QApplication.beep()
        elif name == "NOT_IN_DB":
            self.status_label.setText("❌ Unknown Student - Not in database")
            self.status_label.setStyleSheet("color: #ef4444; font-size: 14px; font-weight: bold;")
        elif name:
            success, status = self.system.mark_attendance(name, self.system.current_schedule)
            if success:
                self.status_label.setText(f"✅ {name} - {status}")
                self.status_label.setStyleSheet("color: #10b981; font-size: 14px; font-weight: bold;")
                self.face_detected.emit(name, status)
            else:
                self.status_label.setText(f"❌ {name} - {status}")
                self.status_label.setStyleSheet("color: #ef4444; font-size: 14px; font-weight: bold;")
        else:
            self.status_label.setText("❌ No Face Detected")
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 14px; font-weight: bold;")
        
        QTimer.singleShot(2000, self.cleanup_and_accept)
    
    def cleanup_and_accept(self):
        """Clean up and accept dialog"""
        self.stop_camera()
        self.accept()
    
    def cleanup_and_reject(self):
        """Clean up and reject dialog"""
        self.stop_camera()
        self.reject()
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.stop_camera()
        event.accept()


# ========== FACE VERIFICATION DIALOG (Documents) ==========

class FaceVerificationDialog(QDialog):
    """Responsive face verification dialog - larger preview, controls lower"""

    verification_result = pyqtSignal(bool, str)

    def __init__(self, system, expected_student, hardware=None, parent=None, distance_fn=None):
        super().__init__(parent)
        self.system = system
        self.expected_student = expected_student
        self.hardware = hardware
        self.distance_fn = distance_fn  # callable → returns last known distance or None
        self.expected_name = f"{expected_student.get('fname', '')} {expected_student.get('lname', '')}".strip()
        self.expected_id = expected_student.get('id', '')

        self.camera_handler = None
        self.use_picamera = False
        self.use_hardware_camera = False
        self.capture = None
        self.timer = QTimer()
        self.scan_start_time = QDateTime.currentDateTime()
        self.processing = False
        self.verification_complete = False
        self._camera_stopping = False



        self.setWindowTitle("Face Verification")
        self.setModal(True)
        self.setStyleSheet(MODERN_STYLE)

        dialog_w, dialog_h, self.ui_scale, _ = responsive_size(660, 620, 500, 470, 0.66, 0.92)
        self.resize(dialog_w, dialog_h)
        self.setMinimumSize(500, 470)

        self.mismatch_count = 0
        self.face_seen = False

        self.init_ui()
        if self.hardware:
            self.hardware.manual_activate()

        self.use_hardware_camera = False
        self.start_camera()

    def _scaled(self, value):
        return max(1, int(value * self.ui_scale))

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self._scaled(14),
            self._scaled(12),
            self._scaled(14),
            self._scaled(12),
        )
        layout.setSpacing(self._scaled(10))

        title = QLabel("VERIFY IDENTITY")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size:{self._scaled(18)}px; font-weight:bold; color:#8b5cf6; padding:{self._scaled(4)}px;"
        )
        layout.addWidget(title)

        # Make preview taller and give it priority
        usable_w = max(440, self.width() - self._scaled(40))
        usable_h = max(380, self.height() - self._scaled(110))

        self.video_w = min(usable_w, self._scaled(600))
        self.video_h = min(
            int(self.video_w * 0.68),   # taller preview than before
            max(self._scaled(240), int(usable_h * 0.58))
        )

        video_container = QFrame()
        video_container.setFixedSize(self.video_w, self.video_h)
        video_container.setStyleSheet(
            "background-color: black; border-radius: 12px; border: 2px solid #475569;"
        )

        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(self.video_w, self.video_h)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        video_layout.addWidget(self.video_label)

        layout.addWidget(video_container, alignment=Qt.AlignCenter)

        # Result text directly under preview
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(self._scaled(28))
        self.result_label.setMaximumHeight(self._scaled(44))
        self.result_label.setStyleSheet(
            f"font-size:{self._scaled(15)}px; font-weight:bold; padding:{self._scaled(2)}px;"
        )
        layout.addWidget(self.result_label)

        # Push progress/hint lower as a separate bottom section
        layout.addSpacing(self._scaled(8))

        bottom_info = QWidget()
        bottom_layout = QVBoxLayout(bottom_info)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(self._scaled(8))

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(self._scaled(10))
        self.progress_bar.setTextVisible(False)
        bottom_layout.addWidget(self.progress_bar)

        self.hint_label = QLabel("Look at camera")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet(
            f"color:#94a3b8; font-size:{self._scaled(11)}px; padding-top:{self._scaled(2)}px;"
        )
        bottom_layout.addWidget(self.hint_label)

        layout.addWidget(bottom_info)

        # Real spacer so controls stay low
        layout.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(self._scaled(44))
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #ef4444;
                color: white;
                font-size: {self._scaled(13)}px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: {self._scaled(8)}px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        cancel_btn.clicked.connect(self.cleanup_and_reject)
        layout.addWidget(cancel_btn)

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(30)
    
    def start_camera(self):
        try:
            if PICAMERA_AVAILABLE:
                self.camera_handler = PiCameraHandler()
                if self.camera_handler.initialize():
                    self.use_picamera = True
                    print("✅ Using Pi Camera")
                else:
                    self.init_opencv_camera()
            else:
                self.init_opencv_camera()

            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
        except Exception as e:
            print(f"Camera error: {e}")
            self.result_label.setText("Camera error")
            self.result_label.setStyleSheet(
                f"color:#ef4444; font-size:{self._scaled(16)}px; font-weight:bold;"
            )

    def init_opencv_camera(self):
        for idx in [0, 1, 2]:
            self.capture = cv2.VideoCapture(idx)
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    print(f"✅ OpenCV camera {idx} working")
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return
                self.capture.release()

        self.capture = None
        self.result_label.setText("No camera found")
        self.result_label.setStyleSheet(
            f"color:#ef4444; font-size:{self._scaled(16)}px; font-weight:bold;"
        )

    def stop_camera(self):
        """Stop camera safely"""
        if getattr(self, "_camera_stopping", False):
            return
        self._camera_stopping = True

        if self.timer and self.timer.isActive():
            self.timer.stop()
        if self.progress_timer and self.progress_timer.isActive():
            self.progress_timer.stop()

        self.processing = True

        if self.use_picamera and self.camera_handler:
            try:
                self.camera_handler.close()
            except Exception as e:
                print(f"Camera close error: {e}")
            self.camera_handler = None
        elif self.capture:
            try:
                self.capture.release()
            except Exception as e:
                print(f"OpenCV release error: {e}")
            self.capture = None

        if self.hardware:
            self.hardware.manual_deactivate()

        print("Camera stopped safely")

    # In FaceVerificationDialog class
    def update_frame(self):
        """Update video frame"""
        if getattr(self, "_camera_stopping", False):
            return
        if self.verification_complete:
            return

        try:
            frame = None

            if self.camera_handler:
                frame = self.camera_handler.capture_frame()
            elif self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    frame = None

            if frame is None:
                return

            # Display frame (same as before)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_image.shape

            scale = min(self.video_w / w, self.video_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            display_image = np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)
            rgb_resized = cv2.resize(rgb_image, (new_w, new_h))

            y_offset = (self.video_h - new_h) // 2
            x_offset = (self.video_w - new_w) // 2
            display_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = rgb_resized

            qt_image = QImage(
                display_image.data,
                self.video_w,
                self.video_h,
                3 * self.video_w,
                QImage.Format_RGB888
            )
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

            # Process face verification - THIS IS THE PART TO UPDATE
            if not self.processing:
                self.processing = True
                self._process_face(frame)
                self.processing = False

        except Exception as e:
            print(f"Frame error: {e}")

    def _process_face(self, frame):
        """Process face for verification with anti-spoofing"""
        # ── Distance guard (inside scan) ──────────────────────────────────────
        if self.distance_fn is not None:
            dist = self.distance_fn()
            if dist is not None and dist <= 0.3:
                self.fail_verification("Too close to sensor — please stand ~1m away")
                return
        # ─────────────────────────────────────────────────────────────────────

        if len(self.system.known_faces) == 0:
            self._load_student_encodings()

        if len(self.system.known_faces) == 0:
            self.fail_verification("No face data")
            return

        # Use the new scan_face method which includes anti-spoofing
        result = self.system.scan_face(frame)
        
        # Handle both old and new return formats
        if len(result) == 2:
            status, name = result
            is_live = True
            liveness_conf = 1.0
        else:
            status, name, is_live, liveness_conf = result
        
        # Check if it's a spoof attack
        if status == "SPOOF" or (status == "FOUND" and not is_live):
            self.fail_verification("Spoof detected! Please use your real face.")
            return
        
        # Check if face was found
        if status == "FOUND" and is_live and name:
            self.face_seen = True
            self.verify_identity(name)
            return
        
        # Handle unknown face
        if status == "UNKNOWN":
            self.mismatch_count += 1
            self.result_label.setText("⚠️ Face detected but does not match selected student")
            self.result_label.setStyleSheet(
                f"color:#f59e0b; font-size:{self._scaled(14)}px; font-weight:bold;"
            )
            if self.mismatch_count >= 2:
                self.fail_verification("Face does not match selected student")
                return
        
        # Check timeout
        elapsed = QDateTime.currentDateTime().toSecsSinceEpoch() - self.scan_start_time.toSecsSinceEpoch()
        if elapsed >= 8:
            self.fail_verification("No face detected" if not self.face_seen else "Face does not match selected student")

    def _load_student_encodings(self):
        try:
            student_id = self.expected_student.get('id')
            section = self.expected_student.get('year-section')

            section_path = os.path.join(Config.LOCAL_STORAGE, "sections", section, "students")
            if os.path.exists(section_path):
                for folder in os.listdir(section_path):
                    if folder.startswith(f"{student_id}_"):
                        cache_file = os.path.join(section_path, folder, "encoding.npy")
                        if os.path.exists(cache_file):
                            cache_data = np.load(cache_file, allow_pickle=True).item()
                            self.system.known_faces = cache_data.get('encodings', [])
                            self.system.known_names = [cache_data.get('name', 'Unknown')] * len(self.system.known_faces)
                            break
        except Exception as e:
            print(f"Error loading encodings: {e}")

    def verify_identity(self, recognized_name):
        self.verification_complete = True
        self.stop_camera()
        self.video_label.clear()

        expected_full = self.expected_name.lower().strip()
        recognized_full = recognized_name.lower().strip()

        if expected_full == recognized_full:
            self.result_label.setText("✅ VERIFIED")
            self.result_label.setStyleSheet(
                f"color:#10b981; font-size:{self._scaled(17)}px; font-weight:bold;"
            )
            self.hint_label.setText("")
            self.verification_result.emit(True, f"Welcome, {self.expected_name}!")
            QTimer.singleShot(1500, self.cleanup_and_accept)
        else:
            self.fail_verification("Face does not match")

    def fail_verification(self, message):
        self.verification_complete = True
        self.stop_camera()
        self.video_label.clear()

        self.result_label.setText(f"❌ {message}")
        self.result_label.setStyleSheet(
            f"color:#ef4444; font-size:{self._scaled(15)}px; font-weight:bold;"
        )
        self.hint_label.setText("")
        self.verification_result.emit(False, message)
        QTimer.singleShot(1500, self.cleanup_and_reject)

    def update_progress(self):
        if self.verification_complete:
            return

        elapsed = QDateTime.currentDateTime().toSecsSinceEpoch() - self.scan_start_time.toSecsSinceEpoch()
        progress = min(100, (elapsed / 8) * 100)
        self.progress_bar.setValue(int(progress))

        if progress > 80:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
        elif progress > 50:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")
        else:
            self.progress_bar.setStyleSheet("")

    def cleanup_and_accept(self):
        self.stop_camera()
        self.accept()

    def cleanup_and_reject(self):
        self.stop_camera()
        self.reject()

    def closeEvent(self, event):
        """Turn off hardware when dialog closes"""
        self.stop_camera()
        event.accept()


# ========== ATTENDANCE TABLE MODEL ==========

class AttendanceTableModel(QAbstractTableModel):
    """Model for attendance data"""
    
    def __init__(self):
        super().__init__()
        self.headers = ["Student Name", "Time In", "Status"]
        self.data = []
    
    def rowCount(self, parent=QModelIndex()):
        return len(self.data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        
        row, col = index.row(), index.column()
        
        if role == Qt.DisplayRole:
            if 0 <= row < len(self.data) and 0 <= col < len(self.headers):
                return self.data[row][col]
        
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        
        elif role == Qt.ForegroundRole:
            if col == 2:  # Status column
                if 0 <= row < len(self.data):
                    status = self.data[row][col]
                    if status == "Present":
                        return QColor(Config.THEME['success'])
                    elif status == "Late":
                        return QColor(Config.THEME['secondary'])
                    else:
                        return QColor(Config.THEME['error'])
        
        return QVariant()
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        elif orientation == Qt.Horizontal and role == Qt.TextAlignmentRole:
            return Qt.AlignCenter  # ← Center headers
        return QVariant()
    
    def update_data(self, new_data):
        """Update table data"""
        self.beginResetModel()
        self.data = new_data
        self.endResetModel()


# ========== UPCOMING CLASSES WINDOW ==========

class UpcomingClassesWindow(QMainWindow):
    """Fullscreen window to display all upcoming classes with force start option"""
    
    def __init__(self, system, parent=None):
        super().__init__(parent)
        self.system = system
        self.parent_app = parent
        self.setWindowTitle("Upcoming Classes - Fullscreen View")
        self.setStyleSheet(MODERN_STYLE)
        
        # Start in fullscreen
        self.showFullScreen()
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # ========== HEADER ==========
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel(" UPCOMING CLASSES TODAY")
        title.setProperty("class", "headerLabel")
        title.setStyleSheet("font-size: 32px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Close button (large, visible)
        close_btn = QPushButton(" CLOSE")
        close_btn.setMinimumSize(150, 60)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # ========== STATS BAR ==========
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(20)
        
        self.total_label = QLabel("Total Classes: 0")
        self.total_label.setStyleSheet(f"color: {Config.THEME['text']}; font-size: 16px; padding: 10px; background-color: {Config.THEME['surface_light']}; border-radius: 8px;")
        stats_layout.addWidget(self.total_label)
        
        self.active_label = QLabel("Active Now: 0")
        self.active_label.setStyleSheet(f"color: {Config.THEME['success']}; font-size: 16px; padding: 10px; background-color: {Config.THEME['surface_light']}; border-radius: 8px;")
        stats_layout.addWidget(self.active_label)
        
        self.upcoming_label = QLabel("Upcoming: 0")
        self.upcoming_label.setStyleSheet(f"color: {Config.THEME['accent']}; font-size: 16px; padding: 10px; background-color: {Config.THEME['surface_light']}; border-radius: 8px;")
        stats_layout.addWidget(self.upcoming_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # ========== CLASSES GRID ==========
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setVerticalSpacing(25)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Initial load
        self.load_classes()
        
        # Auto-refresh timer (30 seconds)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_classes)
        self.refresh_timer.start(30000)
    
    def keyPressEvent(self, event):
        """Handle escape key to exit fullscreen"""
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    
    def load_classes(self):
        """Load and display upcoming classes in a grid"""
        # Clear existing widgets
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        schedules = self.system.get_local_schedules()
        now = datetime.now()
        today_date = now.date()
        
        classes = []
        active_count = 0
        upcoming_count = 0
        
        for sched in schedules:
            try:
                # Skip ended classes
                if sched.get('status') == 'ended':
                    continue
                
                # Get schedule date
                sched_date = sched.get('date')
                if sched_date:
                    if isinstance(sched_date, str):
                        sched_date = datetime.strptime(sched_date, "%Y-%m-%d").date()
                else:
                    sched_date = today_date
                
                # Only show today's classes
                if sched_date != today_date:
                    continue
                
                # Parse time
                sched_time = datetime.strptime(sched['start_time'], "%H:%M:%S").time()
                sched_datetime = datetime.combine(sched_date, sched_time)
                end_datetime = sched_datetime + timedelta(minutes=sched['duration'])
                
                # Determine if class should be shown (current or future)
                if end_datetime >= now - timedelta(minutes=30):  # Show classes that ended within last 30 min
                    classes.append((sched_datetime, sched))
                    
                    # Count stats
                    if sched_datetime <= now <= end_datetime:
                        active_count += 1
                    elif sched_datetime > now:
                        upcoming_count += 1
                        
            except Exception as e:
                print(f"Error processing schedule: {e}")
        
        # Update stats
        self.total_label.setText(f"Total Classes Today: {len(classes)}")
        self.active_label.setText(f" Active Now: {active_count}")
        self.upcoming_label.setText(f"⏳ Upcoming: {upcoming_count}")
        
        # Sort by time
        classes.sort(key=lambda x: x[0])
        
        if not classes:
            # Show empty state
            empty_label = QLabel(" No classes scheduled for today!")
            empty_label.setStyleSheet(f"color: {Config.THEME['text_secondary']}; font-size: 24px; padding: 100px;")
            empty_label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(empty_label, 0, 0, 1, 3)
            return
        
        # Display in grid (3 columns)
        row, col = 0, 0
        for dt, sched in classes:
            card = self.create_class_card(dt, sched)
            self.grid_layout.addWidget(card, row, col)
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
    
    def create_class_card(self, dt, sched):
        """Create a detailed class card with force start button"""
        now = datetime.now()
        end_dt = dt + timedelta(minutes=sched['duration'])
        
        # Determine status
        if dt <= now <= end_dt:
            status = " ACTIVE NOW"
            status_color = Config.THEME['success']
            time_left = end_dt - now
            minutes_left = time_left.seconds // 60
            time_text = f"Ends in {minutes_left} minutes"
        elif dt > now:
            status = "⏳ UPCOMING"
            status_color = Config.THEME['accent']
            time_until = dt - now
            hours = time_until.seconds // 3600
            minutes = (time_until.seconds % 3600) // 60
            if hours > 0:
                time_text = f"Starts in {hours}h {minutes}m"
            else:
                time_text = f"Starts in {minutes}m"
        else:
            status = " ENDED"
            status_color = Config.THEME['text_secondary']
            time_ago = now - dt
            minutes_ago = time_ago.seconds // 60
            time_text = f"Ended {minutes_ago} minutes ago"
        
        # Create card
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setProperty("class", "class-card")
        card.setMinimumHeight(350)
        card.setMaximumWidth(400)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with subject and status
        header = QHBoxLayout()
        
        subject = QLabel(sched['subject'])
        subject.setProperty("class", "class-subject")
        subject.setStyleSheet("font-size: 20px; font-weight: bold;")
        header.addWidget(subject)
        
        status_label = QLabel(status)
        status_label.setStyleSheet(f"color: {status_color}; font-size: 12px; font-weight: bold; padding: 4px 8px; border: 1px solid {status_color}; border-radius: 4px;")
        header.addWidget(status_label)
        header.addStretch()
        layout.addLayout(header)
        
        # Teacher
        teacher = QLabel(f" {sched['teacher']}")
        teacher.setProperty("class", "class-teacher")
        teacher.setStyleSheet("font-size: 16px;")
        layout.addWidget(teacher)
        
        # Section
        section = QLabel(f" Section {sched['section']}")
        section.setStyleSheet(f"color: {Config.THEME['text']}; font-size: 16px;")
        layout.addWidget(section)
        
        # Room
        room = QLabel(f" {sched['classroom']}")
        room.setProperty("class", "class-room")
        room.setStyleSheet("font-size: 16px;")
        layout.addWidget(room)
        
        # Time
        time_label = QLabel(f" {sched['start_time']} - {(dt + timedelta(minutes=sched['duration'])).strftime('%H:%M')}")
        time_label.setProperty("class", "class-time")
        time_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(time_label)
        
        # Countdown/status text
        countdown = QLabel(time_text)
        countdown.setStyleSheet(f"color: {status_color}; font-size: 18px; font-weight: bold; padding-top: 10px; border-top: 1px solid {Config.THEME['border']};")
        countdown.setAlignment(Qt.AlignCenter)
        layout.addWidget(countdown)
        
        # Force Start Button - Only show for upcoming classes
        if dt > now:
            force_start_btn = QPushButton(" FORCE START CLASS NOW")
            force_start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f59e0b;
                    color: white;
                    font-weight: bold;
                    padding: 12px;
                    border-radius: 6px;
                    font-size: 14px;
                    margin-top: 10px;
                }
                QPushButton:hover {
                    background-color: #d97706;
                }
            """)
            force_start_btn.clicked.connect(lambda checked, s=sched: self.force_start_class(s))
            layout.addWidget(force_start_btn)
        
        return card
    
    def force_start_class(self, schedule):
        """Force start a class immediately"""
        print(f" FORCE STARTING CLASS: {schedule['subject']} - {schedule['section']}")
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Force Start Class",
            f"Are you sure you want to FORCE START {schedule['subject']} - {schedule['section']}?\n\n"
            f"This will start the class immediately, regardless of scheduled time.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Update database status to 'active'
                result = self.system.supabase.table("schedule")\
                    .update({"status": "active"})\
                    .eq("id", schedule['id'])\
                    .execute()
                print(f" Database updated to 'active'")
                
            except Exception as db_error:
                print(f" Database error: {db_error}")
            
            # Update local status
            schedule['status'] = 'active'
            
            # Clear old data
            self.system.attendance_log.clear()
            self.system.known_faces.clear()
            self.system.known_names.clear()
            self.system.known_student_ids.clear()
            
            # Set as current schedule
            self.system.current_schedule = schedule
            self.system.current_section = schedule['section']
            
            # Stop background sync
            self.system.stop_background_sync()
            
            # Load students for this section
            face_count = self.system.load_section_students(self.system.current_section)
            
            # Close this window
            self.close()
            
            # Force main UI to refresh
            if self.parent_app:
                # Force a sync to get updated status
                self.parent_app.system.sync_schedules(sync_students=False)
                self.parent_app.refresh_display()
            
            QMessageBox.information(
                self,
                "Class Force Started",
                f" {schedule['subject']} - {schedule['section']} has been force started.\n\n"
                f" Students loaded: {face_count}"
            )
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.refresh_timer.stop()
        event.accept()




# ========== SHARED PRINT ENGINE ==========

def print_attendance_report(parent, data, title_info: dict):
    if data is None or len(data) == 0:
        QMessageBox.information(parent, 'Nothing to Print', 'No attendance records to print.')
        return
    subject  = title_info.get('subject', 'Unknown Subject')
    section  = title_info.get('section', '')
    teacher  = title_info.get('teacher', '')
    date_str = title_info.get('date',    datetime.now().strftime('%Y-%m-%d'))
    present_count = int((data['Status'].str.lower() == 'present').sum()) if 'Status' in data.columns else 0
    late_count    = int((data['Status'].str.lower() == 'late').sum())    if 'Status' in data.columns else 0
    total_count   = len(data)
    columns = ['Student Name', 'Student ID', 'Time In', 'Date', 'Status']
    col_w   = {'Student Name': '34%', 'Student ID': '18%', 'Time In': '13%', 'Date': '16%', 'Status': '11%'}
    th_row  = ''.join(
        f'<th width="{col_w[c]}" style="background-color:#312e81;color:white;padding:7px 8px;text-align:left;">{c}</th>'
        for c in columns)
    row_html = ''
    for i, (_, row) in enumerate(data.iterrows()):
        bg = '#f9fafb' if i % 2 == 0 else '#ffffff'
        sv = str(row.get('Status', '')).lower()
        sc = {'present': '#15803d', 'late': '#b45309', 'absent': '#dc2626'}.get(sv, '#374151')
        cells = ''
        for col in columns:
            val   = str(row.get(col, ''))
            style = f'padding:6px 8px;border:1px solid #e5e7eb;background-color:{bg};'
            if col == 'Status':
                style += f'color:{sc};font-weight:bold;'
            cells += f'<td style="{style}">{val}</td>'
        row_html += f'<tr>{cells}</tr>'
    html = f"""<html><head><meta charset="utf-8">
<style>
  body  {{font-family:Arial,Helvetica,sans-serif;font-size:10pt;color:#111827;margin:0;padding:16px;}}
  h1    {{font-size:14pt;margin:0 0 4px 0;color:#1e1b4b;}}
  .meta {{font-size:9pt;color:#4b5563;margin-bottom:10px;}}
  .badge{{font-size:9pt;font-weight:bold;margin-right:10px;}}
  table {{border-collapse:collapse;width:100%;}}
  tfoot td{{padding:7px 8px;font-weight:bold;font-size:9pt;border-top:2px solid #312e81;background-color:#eef2ff;}}
  .footer{{margin-top:14px;font-size:8pt;color:#9ca3af;text-align:center;}}
</style></head><body>
  <h1>Attendance Report &mdash; {subject}</h1>
  <p class="meta"><b>Section:</b> {section} &nbsp;&nbsp;<b>Teacher:</b> {teacher} &nbsp;&nbsp;<b>Date:</b> {date_str} &nbsp;&nbsp;<b>Printed:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  <p>
    <span class="badge" style="color:#15803d">&#10004; Present: {present_count}</span>
    <span class="badge" style="color:#b45309">&#9888; Late: {late_count}</span>
    <span class="badge" style="color:#374151">Total: {total_count}</span>
  </p>
  <table border="0" cellspacing="0" cellpadding="0" width="100%">
    <thead><tr>{th_row}</tr></thead>
    <tbody>{row_html}</tbody>
    <tfoot><tr><td colspan="5">Total Students: {total_count} &nbsp;&nbsp;({present_count} present, {late_count} late)</td></tr></tfoot>
  </table>
  <p class="footer">Generated by Attendance System &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body></html>"""
    printer = QPrinter(QPrinter.HighResolution)
    printer.setPageSize(QPrinter.A4)
    printer.setOrientation(QPrinter.Portrait)
    doc = QTextDocument()
    doc.setHtml(html)
    preview = QPrintPreviewDialog(printer, parent)
    preview.setWindowTitle(f'Print \u2014 {subject} {section}')
    preview.paintRequested.connect(doc.print_)
    preview.exec_()


# ========== MAIN ATTENDANCE UI ==========


class LocalAttendanceRecordsPage(QWidget):
    """Attendance records page that loads saved local CSV files."""

    def __init__(self):
        super().__init__()
        self.attendance_files = []          # All files
        self.filtered_files = []            # Files that match current filter
        self.current_data = pd.DataFrame()
        self.current_file_name = ""
        self.compact_mode = False           # Set before _detect_ui_scale uses font_px
        self.ui_scale = self._detect_ui_scale()
        self._build_ui()
        self.load_attendance_records()

    def _detect_ui_scale(self):
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                return 1.0
            geo = screen.availableGeometry()
            scale = min(geo.width() / 1280.0, geo.height() / 720.0, 1.4)
            return max(0.92, scale)
        except Exception:
            return 1.0

    def scaled(self, value):
        return max(1, int(value * self.ui_scale))

    def font_px(self, compact_px, regular_px=None):
        """Return a font size tuned for the current window mode."""
        if regular_px is None:
            regular_px = compact_px
        return self.scaled(compact_px if self.compact_mode else regular_px)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)  # Reduced spacing since header is gone

        # 🔥 REMOVED ENTIRE HEADER SECTION
        # No header rectangle with title and description anymore

        # Filter Bar
        filter_bar = QFrame()
        filter_bar.setObjectName('panel')
        filter_layout = QHBoxLayout(filter_bar)
        filter_layout.setContentsMargins(18, 12, 18, 12)
        filter_layout.setSpacing(15)
        fl = QLabel('Filter by:')
        fl.setStyleSheet('font-size:14px; font-weight:600;')
        filter_layout.addWidget(fl)

        self.subject_filter = QComboBox()
        self.subject_filter.setMinimumHeight(38)
        self.subject_filter.addItem('All Subjects')
        self.subject_filter.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.subject_filter)

        self.date_filter = QComboBox()
        self.date_filter.setMinimumHeight(38)
        self.date_filter.addItem('All Dates')
        self.date_filter.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.date_filter)

        filter_layout.addStretch()
        self.summary_label = QLabel('')
        self.summary_label.setStyleSheet('color:#94a3b8;')
        filter_layout.addWidget(self.summary_label)
        print_btn = QPushButton('🖨 Print')
        print_btn.setMinimumHeight(38)
        print_btn.setStyleSheet('QPushButton{background:#4f46e5;color:white;border:none;border-radius:6px;padding:6px 16px;font-weight:600;font-size:13px;}QPushButton:hover{background:#6366f1;}')
        print_btn.clicked.connect(self.print_current)
        filter_layout.addWidget(print_btn)
        root.addWidget(filter_bar)

        # Main content splitter
        content = QSplitter(Qt.Horizontal)
        content.setChildrenCollapsible(False)
        content.setHandleWidth(8)

        # Left panel - File list
        left_panel = QFrame()
        left_panel.setObjectName('panel')
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(self.scaled(12), self.scaled(12), self.scaled(12), self.scaled(12))
        left_layout.setSpacing(12)
        
        file_title = QLabel('Attendance Files')
        file_title.setStyleSheet('font-size:18px; font-weight:800; color:white;')
        left_layout.addWidget(file_title)
        
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        self.file_list.setStyleSheet(f'font-size:{self.scaled(14)}px; padding: {self.scaled(4)}px;')
        left_layout.addWidget(self.file_list, 1)
        content.addWidget(left_panel)

        # Right panel - Data table
        right_panel = QFrame()
        right_panel.setObjectName('panel')
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(12)
        
        data_title = QLabel('Attendance Data')
        data_title.setStyleSheet('font-size:18px; font-weight:800; color:white;')
        right_layout.addWidget(data_title)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            'Subject', 'Teacher', 'Section', 'Student Name', 'Student ID', 'Time In', 'Date', 'Status'
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(self.scaled(38))
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f'font-size:{self.scaled(13)}px;')
        right_layout.addWidget(self.table, 1)
        content.addWidget(right_panel)
        content.setSizes([300, 900])

        root.addWidget(content, 1)

    def update_filters(self):
        """Update filter dropdowns based on loaded data"""
        subjects = set()
        dates = set()
        
        for filepath in self.attendance_files:
            try:
                df = pd.read_csv(filepath)
                if 'Subject' in df.columns:
                    subjects.update(df['Subject'].dropna().unique())
                if 'Date' in df.columns:
                    dates.update(df['Date'].dropna().unique())
            except Exception:
                continue
        
        self.subject_filter.blockSignals(True)
        self.date_filter.blockSignals(True)
        
        self.subject_filter.clear()
        self.subject_filter.addItem('All Subjects')
        for sub in sorted(subjects):
            self.subject_filter.addItem(sub)
        
        self.date_filter.clear()
        self.date_filter.addItem('All Dates')
        for d in sorted(dates, reverse=True):
            self.date_filter.addItem(d)
        
        self.subject_filter.blockSignals(False)
        self.date_filter.blockSignals(False)

    def apply_filter(self):
        """Apply filters to BOTH file list AND data table"""
        subject = self.subject_filter.currentText()
        date = self.date_filter.currentText()
        
        # STEP 1: Filter which files to show
        self.filtered_files = []
        for filepath in self.attendance_files:
            try:
                df = pd.read_csv(filepath)
                file_matches = True
                
                if subject != 'All Subjects' and 'Subject' in df.columns:
                    if subject not in df['Subject'].astype(str).values:
                        file_matches = False
                
                if date != 'All Dates' and 'Date' in df.columns:
                    if date not in df['Date'].astype(str).values:
                        file_matches = False
                
                if file_matches:
                    self.filtered_files.append(filepath)
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
        
        # STEP 2: Update the file list display
        self._update_file_list()
        
        # STEP 3: Auto-select the first filtered file and load its data
        if self.filtered_files:
            first_file = self.filtered_files[0]
            self._load_file_data(first_file)
            
            # Highlight it in the list
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item and item.data(Qt.UserRole) == first_file:
                    self.file_list.setCurrentRow(i)
                    break
        else:
            self.table.setRowCount(0)
            self.summary_label.setText(f'No files contain {subject} data' if subject != 'All Subjects' else 'No files found')

    def _update_file_list(self):
        """Update the file list widget with filtered files"""
        self.file_list.blockSignals(True)
        self.file_list.clear()
        
        if not self.filtered_files:
            msg_item = QListWidgetItem("No files match current filter")
            msg_item.setFlags(msg_item.flags() & ~Qt.ItemIsSelectable)
            msg_item.setForeground(QBrush(QColor('#ef4444')))
            self.file_list.addItem(msg_item)
            self.file_list.blockSignals(False)
            return
        
        for filepath in self.filtered_files:
            name = os.path.basename(filepath).replace('.csv', '')
            parts = name.split('_')
            
            if len(parts) >= 4:
                display_name = f"{parts[1]} - {parts[2]} - {parts[3]}"
            elif len(parts) >= 3:
                display_name = f"{parts[1]} - {parts[2]}"
            else:
                display_name = name
            
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)
        
        self.file_list.blockSignals(False)

    def on_file_selected(self, current, previous):
        """Handle file selection from list"""
        if current is None:
            return
        
        filepath = current.data(Qt.UserRole)
        if filepath and os.path.exists(filepath):
            self._load_file_data(filepath)

    def _load_file_data(self, filepath):
        """Load and display data from selected CSV file"""
        try:
            self.current_data = pd.read_csv(filepath)
            self.current_file_name = os.path.basename(filepath)
            self._apply_filters_to_data()
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            self.current_data = pd.DataFrame()
            self.table.setRowCount(0)
            self.summary_label.setText('Error loading file')

    def _apply_filters_to_data(self):
        """Apply subject and date filters to the loaded data table"""
        if self.current_data is None or len(self.current_data) == 0:
            self.table.setRowCount(0)
            self.summary_label.setText('No data')
            return
        
        filtered = self.current_data.copy()
        subject = self.subject_filter.currentText()
        date = self.date_filter.currentText()
        
        if subject != 'All Subjects' and 'Subject' in filtered.columns:
            filtered = filtered[filtered['Subject'].astype(str) == subject]
        
        if date != 'All Dates' and 'Date' in filtered.columns:
            filtered = filtered[filtered['Date'].astype(str) == date]
        
        columns = ['Subject', 'Teacher', 'Section', 'Student Name', 'Student ID', 'Time In', 'Date', 'Status']
        self.table.setRowCount(len(filtered))
        self.table.clearContents()
        
        for row, (_, data) in enumerate(filtered.iterrows()):
            for col, col_name in enumerate(columns):
                value = '' if col_name not in data else str(data[col_name])
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if col_name == 'Status':
                    lower = value.lower()
                    if lower == 'present':
                        item.setForeground(QBrush(QColor('#10b981')))
                    elif lower == 'late':
                        item.setForeground(QBrush(QColor('#f59e0b')))
                    elif lower == 'absent':
                        item.setForeground(QBrush(QColor('#ef4444')))
                self.table.setItem(row, col, item)
        
        self.summary_label.setText(f'Showing {len(filtered)} records from {self.current_file_name}')

    def load_attendance_records(self):
        """Load all attendance CSV files"""
        self.attendance_files = []
        csv_files = glob.glob('attendance_*.csv')
        csv_files.sort(reverse=True)
        
        for file in csv_files:
            self.attendance_files.append(file)
        
        self.update_filters()
        self.apply_filter()

    def export_attendance(self):
        """Export current data to CSV"""
        if self.current_data is None or len(self.current_data) == 0:
            return
        
        filtered = self.current_data.copy()
        subject = self.subject_filter.currentText()
        if subject != 'All Subjects' and 'Subject' in filtered.columns:
            filtered = filtered[filtered['Subject'].astype(str) == subject]
        
        date = self.date_filter.currentText()
        if date != 'All Dates' and 'Date' in filtered.columns:
            filtered = filtered[filtered['Date'].astype(str) == date]
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Attendance',
            f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            'CSV Files (*.csv)'
        )
        
        if filename:
            filtered.to_csv(filename, index=False)
            QMessageBox.information(self, 'Export Complete', f'Saved to {filename}')

    def print_current(self):
        if self.current_data is None or len(self.current_data) == 0:
            QMessageBox.information(self, 'Nothing to Print', 'Load an attendance file first.')
            return
        filtered = self.current_data.copy()
        subject = self.subject_filter.currentText()
        date    = self.date_filter.currentText()
        if subject != 'All Subjects' and 'Subject' in filtered.columns:
            filtered = filtered[filtered['Subject'].astype(str) == subject]
        if date != 'All Dates' and 'Date' in filtered.columns:
            filtered = filtered[filtered['Date'].astype(str) == date]
        first = filtered.iloc[0] if len(filtered) > 0 else {}
        print_attendance_report(self, filtered, {
            'subject': str(first.get('Subject', subject)),
            'section': str(first.get('Section', '')),
            'teacher': str(first.get('Teacher', '')),
            'date':    str(first.get('Date', date)),
        })

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("⚠️ PyMuPDF not installed. Document viewer will use external viewer.")

class IntegratedDocumentsPage(QWidget):
    """Single-page student document workflow with verification and embedded PDF preview."""
    FITZ_AVAILABLE = FITZ_AVAILABLE 
    def __init__(self, system, main_window=None, parent=None):
        super().__init__(parent)
        self.system = system
        self.main_window = main_window
        self.selected_student = None
        self.verified = False
        self._data_ready = False
        self.current_pdf_path = None
        self.pdf_document = None
        self.pages = []
        self.zoom_factor = 1.0
        self.compact_mode = False

       

        from document_viewer_logic import DocumentViewerLogic
        self.logic = DocumentViewerLogic(system.supabase, Config)
        self.logic.terms_loaded.connect(self.on_terms_loaded)
        self.logic.document_ready.connect(self.on_document_ready)
        self.logic.document_error.connect(self.on_document_error)
        self.logic.progress_update.connect(self.on_progress_update)

        self._build_ui()
        self.load_years()
        self._apply_initial_layout()

    def _build_ui(self):
        """Build the UI once - placeholder created once"""
        self.root_layout = QHBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(8)

        # ========== LEFT PANEL ==========
        self.left_panel = QFrame()
        self.left_panel.setObjectName('panel')
        self.left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        left_layout = QVBoxLayout(self.left_panel)
        self.left_layout = left_layout
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(8)

        self.title_label = QLabel('Student Document Access')
        self.title_label.setStyleSheet('font-size:20px; font-weight:800; color:white;')
        self.title_label.setWordWrap(True)
        left_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel('Select student and verify identity to access documents')
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet('color:#cbd5e1; font-size:11px;')
        left_layout.addWidget(self.subtitle_label)

        # Student info card
        self.student_card = QFrame()
        self.student_card.setObjectName('cameraCard')
        card_layout = QVBoxLayout(self.student_card)
        self.student_card_layout = card_layout
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(4)

        self.student_name_label = QLabel('No student selected')
        self.student_name_label.setStyleSheet('font-size:16px; font-weight:800; color:white;')
        self.student_name_label.setWordWrap(True)
        self.student_meta_label = QLabel('Choose year, section, and student')
        self.student_meta_label.setStyleSheet('color:#94a3b8;')
        self.student_meta_label.setWordWrap(True)
        self.verify_state_label = QLabel('Verification required')
        self.verify_state_label.setStyleSheet('color:#f59e0b; font-weight:700;')
        self.verify_state_label.setWordWrap(True)

        card_layout.addWidget(self.student_name_label)
        card_layout.addWidget(self.student_meta_label)
        card_layout.addWidget(self.verify_state_label)
        left_layout.addWidget(self.student_card)

        # Year Selection
        self.year_label = QLabel('Year Level')
        left_layout.addWidget(self.year_label)
        self.year_combo = QComboBox()
        self.year_combo.setMinimumHeight(36)
        self.year_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.year_combo.currentIndexChanged.connect(self.on_year_changed)
        left_layout.addWidget(self.year_combo)

        # Section Selection
        self.section_label = QLabel('Section')
        left_layout.addWidget(self.section_label)
        self.section_combo = QComboBox()
        self.section_combo.setMinimumHeight(36)
        self.section_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.section_combo.currentIndexChanged.connect(self.on_section_changed)
        left_layout.addWidget(self.section_combo)

        # Student Selection
        self.student_label = QLabel('Student')
        left_layout.addWidget(self.student_label)
        self.student_combo = QComboBox()
        self.student_combo.setMinimumHeight(36)
        self.student_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.student_combo.currentIndexChanged.connect(self.on_student_changed)
        left_layout.addWidget(self.student_combo)

        # Buttons
        btn_layout = QHBoxLayout()
        self.button_row = btn_layout
        btn_layout.setSpacing(8)
        self.verify_btn = QPushButton('Verify Identity')
        self.verify_btn.setEnabled(False)
        self.verify_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.verify_btn.clicked.connect(self.start_verification)
        btn_layout.addWidget(self.verify_btn, 3)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.setObjectName('ghostBtn')
        self.reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.reset_btn.clicked.connect(self.reset_document_flow)
        btn_layout.addWidget(self.reset_btn, 2)
        left_layout.addLayout(btn_layout)

        # Progress
        self.progress_frame = QFrame()
        self.progress_frame.setObjectName('cameraCard')
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(12, 12, 12, 12)
        progress_layout.setSpacing(6)
        self.progress_label = QLabel('')
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.progress_frame)

        # Document Type
        self.doc_type_label = QLabel('Document Type')
        left_layout.addWidget(self.doc_type_label)
        self.doc_type_combo = QComboBox()
        self.doc_type_combo.addItem('Certificate of Grades', 'COG')
        self.doc_type_combo.addItem('Advising Form', 'ADVISING')
        self.doc_type_combo.setMinimumHeight(36)
        self.doc_type_combo.setEnabled(False)
        self.doc_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.doc_type_combo.currentIndexChanged.connect(self._update_fetch_button_state)
        left_layout.addWidget(self.doc_type_combo)

        # Academic Term
        self.term_label = QLabel('Academic Period')
        left_layout.addWidget(self.term_label)
        self.term_combo = QComboBox()
        self.term_combo.addItem('Verify student first', None)
        self.term_combo.setMinimumHeight(36)
        self.term_combo.setEnabled(False)
        self.term_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.term_combo.currentIndexChanged.connect(self._update_fetch_button_state)
        left_layout.addWidget(self.term_combo)

        # Fetch Button
        self.fetch_btn = QPushButton('Fetch Document')
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.fetch_btn.clicked.connect(self.fetch_document)
        left_layout.addWidget(self.fetch_btn)

        self.info_label = QLabel('Documents are fetched live and cleaned automatically.')
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet('color:#94a3b8;')
        left_layout.addWidget(self.info_label)
        left_layout.addStretch()

        # ========== RIGHT PANEL - PDF VIEWER ==========
        self.right_panel = QFrame()
        self.right_panel.setObjectName('panel')
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(self.right_panel)
        self.right_layout = right_layout
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(8)

        self.preview_title = QLabel('Document Preview')
        self.preview_title.setStyleSheet('font-weight:700; color:white;')

        self.zoom_out_btn = QPushButton('-')
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_out_btn.setEnabled(False)
        self.zoom_out_btn.setStyleSheet('font-size: 14px; font-weight: bold; padding: 0px;')

        self.zoom_label = QLabel('100%')
        self.zoom_label.setFixedWidth(48)
        self.zoom_label.setAlignment(Qt.AlignCenter)

        self.zoom_in_btn = QPushButton('+')
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_in_btn.setEnabled(False)
        self.zoom_in_btn.setStyleSheet('font-size: 14px; font-weight: bold; padding: 0px;')

        self.print_btn = QPushButton('Print')
        self.print_btn.setObjectName('ghostBtn')
        self.print_btn.setEnabled(False)
        self.print_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.print_btn.clicked.connect(self.print_document)

        self.close_doc_btn = QPushButton('Close')
        self.close_doc_btn.setObjectName('ghostBtn')
        self.close_doc_btn.setEnabled(False)
        self.close_doc_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.close_doc_btn.clicked.connect(self.cleanup_document)

        self.toolbar_bar = QFrame()
        toolbar_bar_layout = QVBoxLayout(self.toolbar_bar)
        toolbar_bar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_bar_layout.setSpacing(6)

        top_toolbar = QHBoxLayout()
        top_toolbar.setContentsMargins(0, 0, 0, 0)
        top_toolbar.setSpacing(8)
        top_toolbar.addWidget(self.preview_title)
        top_toolbar.addStretch()
        top_toolbar.addWidget(self.zoom_out_btn)
        top_toolbar.addWidget(self.zoom_label)
        top_toolbar.addWidget(self.zoom_in_btn)
        toolbar_bar_layout.addLayout(top_toolbar)

        bottom_toolbar = QHBoxLayout()
        bottom_toolbar.setContentsMargins(0, 0, 0, 0)
        bottom_toolbar.setSpacing(8)
        bottom_toolbar.addStretch()
        bottom_toolbar.addWidget(self.print_btn)
        bottom_toolbar.addWidget(self.close_doc_btn)
        toolbar_bar_layout.addLayout(bottom_toolbar)

        right_layout.addWidget(self.toolbar_bar)

        # Scroll area for PDF
        self.pdf_scroll = QScrollArea()
        self.pdf_scroll.setWidgetResizable(True)
        self.pdf_scroll.setStyleSheet('QScrollArea { border: 1px solid #253046; border-radius: 14px; background: #0b1220; }')

        self.pdf_container = QWidget()
        self.pdf_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pdf_layout = QVBoxLayout(self.pdf_container)
        self.pdf_layout.setContentsMargins(10, 10, 10, 10)
        self.pdf_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.pdf_layout.setSpacing(12)

        # Placeholder created ONCE
        self.placeholder_label = QLabel('Select a student and click Verify Identity')
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setMinimumHeight(320)
        self.placeholder_label.setStyleSheet('border:1px dashed #475569; border-radius:18px; color:#94a3b8; font-size:13px;')
        self.pdf_layout.addWidget(self.placeholder_label)

        self.pdf_scroll.setWidget(self.pdf_container)
        right_layout.addWidget(self.pdf_scroll, 1)

        self.root_layout.addWidget(self.left_panel, 3)
        self.root_layout.addWidget(self.right_panel, 4)

    # ========== RESIZE HANDLING ==========

    def _set_compact_document_layout(self, compact):
        self.compact_mode = compact
        self.root_layout.setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)
        self.root_layout.setSpacing(8 if compact else 10)

        self.left_panel.setMinimumWidth(0 if compact else 280)
        self.left_panel.setMaximumWidth(16777215)
        self.right_panel.setMinimumWidth(0)
        self.right_panel.setMaximumWidth(16777215)

        self.left_layout.setContentsMargins(12 if compact else 14, 12 if compact else 14, 12 if compact else 14, 12 if compact else 14)
        self.left_layout.setSpacing(7 if compact else 8)
        self.right_layout.setContentsMargins(12 if compact else 14, 12 if compact else 14, 12 if compact else 14, 12 if compact else 14)
        self.right_layout.setSpacing(7 if compact else 8)
        self.student_card_layout.setContentsMargins(10 if compact else 12, 10 if compact else 12, 10 if compact else 12, 10 if compact else 12)

        combo_h = 34 if compact else 36
        for combo in (self.year_combo, self.section_combo, self.student_combo, self.doc_type_combo, self.term_combo):
            combo.setMinimumHeight(combo_h)

        btn_h = 36 if compact else 40
        for btn in (self.verify_btn, self.reset_btn, self.fetch_btn, self.print_btn, self.close_doc_btn):
            btn.setMinimumHeight(btn_h)

        self.zoom_out_btn.setFixedSize(28, 28) if compact else self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_in_btn.setFixedSize(28, 28) if compact else self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_label.setFixedWidth(44 if compact else 48)

        self.title_label.setStyleSheet(f'font-size:{18 if compact else 20}px; font-weight:800; color:white;')
        self.subtitle_label.setStyleSheet(f'color:#cbd5e1; font-size:{10 if compact else 11}px;')
        self.student_name_label.setStyleSheet(f'font-size:{15 if compact else 16}px; font-weight:800; color:white;')
        self.info_label.setStyleSheet(f'color:#94a3b8; font-size:{10 if compact else 11}px;')
        self.preview_title.setStyleSheet(f'font-weight:700; color:white; font-size:{11 if compact else 12}px;')
        self.placeholder_label.setMinimumHeight(220 if compact else 320)

    def _apply_initial_layout(self):
        """Set initial layout based on screen size"""
        try:
            self._set_compact_document_layout(self.width() < 1200)
        except:
            pass

    def resizeEvent(self, event):
        """Handle resize - simple and safe"""
        super().resizeEvent(event)
        try:
            self._set_compact_document_layout(self.width() < 1200)
        except Exception as e:
            print(f"Resize error: {e}")


    def sync_student_photos_with_loading(self, student_id, student_name, section):
        """Sync student photos with loading screen"""
        if not self.main_window:
            # No main window, just do it directly
            self.system.mirror_student_photos(student_id, student_name, section)
            return
        
        loading = ModernLoadingDialog(
            f"Preparing {student_name}",
            f"Downloading photos for {student_name}...",
            self.main_window
        )
        
        import threading
        def worker():
            try:
                # Call the existing method
                self.system.mirror_student_photos(student_id, student_name, section)
            finally:
                QMetaObject.invokeMethod(loading, "accept", Qt.QueuedConnection)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        loading.exec_()

    def download_student_images_with_loading(self, student_data):
        """Download student images with loading screen"""
        if not self.main_window:
            return self.system.download_student_images(student_data)
        
        student_name = f"{student_data.get('fname', '')} {student_data.get('lname', '')}".strip()
        
        loading = ModernLoadingDialog(
            f"Processing {student_name}",
            f"Building face recognition data...",
            self.main_window
        )
        
        result = [False]
        
        import threading
        def worker():
            try:
                result[0] = self.system.download_student_images(student_data)
            finally:
                QMetaObject.invokeMethod(loading, "accept", Qt.QueuedConnection)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        loading.exec_()
        return result[0]


    # ========== PDF VIEWER METHODS ==========

    def clear_pdf_widgets(self):
        """Clear PDF pages - keep placeholder"""
        while self.pdf_layout.count():
            item = self.pdf_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget != self.placeholder_label:
                widget.deleteLater()
        
        self.placeholder_label.setVisible(True)
        self.placeholder_label.setText('Select a student and click Verify Identity')
        self.pages = []

    def display_pdf(self, pdf_path):
        """Display PDF - replaces placeholder with pages"""
        self.clear_pdf_widgets()
        self.pages = []
        self.zoom_factor = 1.0
        self.zoom_label.setText('100%')
        
        self.placeholder_label.setVisible(False)

        if not self.FITZ_AVAILABLE:
            self.placeholder_label.setVisible(True)
            self.placeholder_label.setText('PyMuPDF not installed. Use Open Externally button.')
            return

        try:
            self.pdf_document = fitz.open(pdf_path)
            for page_num in range(len(self.pdf_document)):
                page_label = QLabel(f'Page {page_num + 1} of {len(self.pdf_document)}')
                page_label.setAlignment(Qt.AlignCenter)
                page_label.setStyleSheet('color:#94a3b8; font-size:11px;')
                self.pdf_layout.addWidget(page_label)

                img_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                self.pdf_layout.addWidget(img_label)
                self.pages.append({'label': img_label, 'page_num': page_num})
            
            self.refresh_pdf_display()
            
        except Exception as e:
            self.placeholder_label.setVisible(True)
            self.placeholder_label.setText(f'Error loading PDF: {str(e)[:100]}')

    def refresh_pdf_display(self):
        """Refresh all pages with current zoom"""
        if not self.pages or not self.pdf_document:
            return
        
        zoom = (1.2 if self.compact_mode else 1.5) * self.zoom_factor
        for page_data in self.pages:
            try:
                page = self.pdf_document[page_data['page_num']]
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                qimage = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                page_data['label'].setPixmap(QPixmap.fromImage(qimage))
            except Exception as e:
                print(f"Page render error: {e}")

    def zoom_in(self):
        """Increase zoom"""
        self.zoom_factor = min(self.zoom_factor + 0.2, 3.0)
        self.zoom_label.setText(f'{int(self.zoom_factor * 100)}%')
        self.refresh_pdf_display()

    def zoom_out(self):
        """Decrease zoom"""
        self.zoom_factor = max(self.zoom_factor - 0.2, 0.5)
        self.zoom_label.setText(f'{int(self.zoom_factor * 100)}%')
        self.refresh_pdf_display()

    def print_document(self):
        """Print the current document directly - no confirmation dialog"""
        if not self.current_pdf_path or not os.path.exists(self.current_pdf_path):
            QMessageBox.warning(self, 'No Document', 'No document loaded to print')
            return
        
        if not self.FITZ_AVAILABLE:
            QMessageBox.critical(self, 'Missing Library', 'PyMuPDF not installed for printing')
            return
        
        from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
        from PyQt5.QtGui import QPainter
        
        # Show printing status
        self._show_status('Printing document...')
        QApplication.processEvents()
        
        # Create printer
        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageSize(QPrinter.Letter)
        printer.setOrientation(QPrinter.Portrait)
        
        # Show printer selection dialog
        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle('Print Document')
        
        if dialog.exec_() == QDialog.Accepted:
            self._print_pdf_with_fitz(printer)
        else:
            self._show_status('Print cancelled')

    def _print_pdf_with_fitz(self, printer):
        """Print PDF using fitz - called by print preview or direct print"""
        try:
            self._show_status('Processing pages...')
            QApplication.processEvents()
            
            # Open PDF with fitz
            doc = fitz.open(self.current_pdf_path)
            num_pages = len(doc)
            
            painter = QPainter()
            painter.begin(printer)
            
            for page_num in range(num_pages):
                if page_num % 5 == 0:
                    self._show_status(f'Printing page {page_num + 1} of {num_pages}...')
                    QApplication.processEvents()
                
                if page_num > 0:
                    printer.newPage()
                
                page = doc[page_num]
                
                # Render with higher resolution for printing
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to QImage
                img_format = QImage.Format_RGB888
                qimage = QImage(
                    pix.samples, pix.width, pix.height,
                    pix.stride, img_format
                )
                
                # Convert to QPixmap for painting
                pixmap = QPixmap.fromImage(qimage)
                
                # Scale to fit page while preserving aspect ratio
                page_rect = printer.pageRect(QPrinter.DevicePixel)
                scaled_pixmap = pixmap.scaled(
                    page_rect.size().toSize(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                # Center on page
                x = (page_rect.width() - scaled_pixmap.width()) / 2
                y = (page_rect.height() - scaled_pixmap.height()) / 2
                
                painter.drawPixmap(int(x), int(y), scaled_pixmap)
            
            painter.end()
            doc.close()
            
            self._show_status('Print completed')
            
        except Exception as e:
            print(f"Print error: {e}")
            self._show_status('Print failed')
            QMessageBox.critical(self, 'Print Error', f'Failed to print document:\n{str(e)}')

    def cleanup_document(self):
        """Clean up and show placeholder"""
        if hasattr(self, 'logic'):
            self.logic.cleanup_current_document()
        self.current_pdf_path = None
        if self.pdf_document:
            try:
                self.pdf_document.close()
            except:
                pass
        self.pdf_document = None
        self.clear_pdf_widgets()
        self.print_btn.setEnabled(False)
        self.close_doc_btn.setEnabled(False)
        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.zoom_factor = 1.0
        self.zoom_label.setText('100%')

    # ========== STUDENT SELECTION METHODS ==========

    def load_years(self):
        """Load available years from database"""
        try:
            response = self.system.supabase.table('student').select('year-section').execute()
            years = set()
            for student in response.data or []:
                ys = student.get('year-section')
                if ys:
                    years.add(str(ys)[0])
            
            self.year_combo.blockSignals(True)
            self.year_combo.clear()
            self.year_combo.addItem('-- Select Year --', None)
            for year in sorted(years):
                mapping = {'1': '1st Year', '2': '2nd Year', '3': '3rd Year', '4': '4th Year'}
                self.year_combo.addItem(mapping.get(year, f'{year}th Year'), year)
            self.year_combo.blockSignals(False)
            
        except Exception as e:
            print(f"Error loading years: {e}")

    def on_year_changed(self, index):
        """Handle year selection"""
        self.section_combo.clear()
        self.section_combo.addItem('-- Select Section --', None)
        self.student_combo.clear()
        self.student_combo.addItem('-- Select Student --', None)
        self.selected_student = None
        self.verify_btn.setEnabled(False)
        self._lock_document_controls()
        self.verified = False
        self._update_student_card()
        
        if index <= 0:
            return
        
        year = self.year_combo.currentData()
        
        if self.main_window:
            try:
                # This will show loading dialog and close when done
                self.main_window.run_with_loading(
                    "Loading Sections",
                    f"Loading sections for {self.year_combo.currentText()}...",
                    self._load_sections,
                    year
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load sections: {e}")
        else:
            self._load_sections(year)

    def _load_sections(self, year):
        """Load sections for selected year"""
        try:
            response = self.system.supabase.table('student').select('year-section').execute()
            sections = set()
            for student in response.data or []:
                ys = student.get('year-section')
                if ys and str(ys).startswith(str(year)):
                    sections.add(ys)
            
            self.section_combo.blockSignals(True)
            self.section_combo.clear()
            self.section_combo.addItem('-- Select Section --', None)
            for section in sorted(sections):
                self.section_combo.addItem(f'Section {section}', section)
            self.section_combo.blockSignals(False)
            
        except Exception as e:
            print(f"Error loading sections: {e}")

    def on_section_changed(self, index):
        """Handle section selection"""
        self.student_combo.clear()
        self.student_combo.addItem('-- Select Student --', None)
        self.selected_student = None
        self.verify_btn.setEnabled(False)
        self._lock_document_controls()
        self._update_student_card()
        self.verified = False
        
        if index <= 0:
            return
        
        section = self.section_combo.currentData()
        
        if self.main_window:
            try:
                # This will show loading dialog and close when done
                self.main_window.run_with_loading(
                    "Loading Students",
                    f"Loading students from section {section}...",
                    self._load_students,
                    section
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load students: {e}")
        else:
            self._load_students(section)

    def _load_students(self, section):
        """Load students for selected section"""
        try:
            students = self.system.get_students_by_section(section)
            students = sorted(students, key=lambda s: (s.get('lname', ''), s.get('fname', '')))
            
            self.student_combo.blockSignals(True)
            self.student_combo.clear()
            self.student_combo.addItem('-- Select Student --', None)
            for student in students:
                name = f"{student.get('lname', '')}, {student.get('fname', '')}".strip(', ')
                display = f"{name} ({student.get('id', '')})"
                self.student_combo.addItem(display, student)
            self.student_combo.blockSignals(False)
            
        except Exception as e:
            print(f"Error loading students: {e}")

    def on_student_changed(self, index):
        """Handle student selection"""
        new_student = self.student_combo.currentData() if index > 0 else None

        if new_student and self.selected_student:
            if new_student.get('id') != self.selected_student.get('id'):
                print(f"🔄 Student changed from {self.selected_student.get('id')} to {new_student.get('id')}")
                self.verified = False
                self._data_ready = False
                self._lock_document_controls()
                self.cleanup_document()

        if index <= 0:
            self.selected_student = None
            self.verify_btn.setEnabled(False)
            self._data_ready = False
            self._lock_document_controls()
            self.verified = False
        else:
            self.selected_student = new_student
            self._data_ready = False
            self.verify_btn.setEnabled(False)  # disable while data is being prepared
            # Kick off photo sync + cache build in background immediately
            self._prepare_student_data()

        self._update_student_card()

    def _update_student_card(self):
        """Update the student info card"""
        if not self.selected_student:
            self.student_name_label.setText('No student selected')
            self.student_meta_label.setText('Choose year, section, and student')
            self.verify_state_label.setText('Verification required')
            self.verify_state_label.setStyleSheet('color:#f59e0b; font-weight:700;')
            return
        
        name = f"{self.selected_student.get('fname', '')} {self.selected_student.get('lname', '')}".strip()
        sid = self.selected_student.get('id', '')
        section = self.selected_student.get('year-section', '')
        self.student_name_label.setText(name)
        self.student_meta_label.setText(f'ID: {sid} | Section: {section}')
        
        if self.verified:
            self.verify_state_label.setText('Verified')
            self.verify_state_label.setStyleSheet('color:#10b981; font-weight:700;')
        else:
            self.verify_state_label.setText('Verification required')
            self.verify_state_label.setStyleSheet('color:#f59e0b; font-weight:700;')

    def _lock_document_controls(self):
        """Lock document controls until verification"""
        self.verified = False
        self.doc_type_combo.setEnabled(False)
        self.term_combo.setEnabled(False)
        self.term_combo.clear()
        self.term_combo.addItem('Verify student first', None)
        self.fetch_btn.setEnabled(False)

    def _update_fetch_button_state(self):
        """Update fetch button state"""
        if self.verified and self.doc_type_combo.currentData() and self.term_combo.currentData():
            self.fetch_btn.setEnabled(True)
        else:
            self.fetch_btn.setEnabled(False)

    def _prepare_student_data(self):
        """Background sync of photos + cache build triggered on student selection.
        Runs in a thread with a loading dialog so the UI stays responsive.
        Sets self._data_ready and re-enables the Verify button when done.
        """
        if not self.selected_student:
            return

        student_name    = f"{self.selected_student.get('fname', '')} {self.selected_student.get('lname', '')}".strip()
        student_id      = self.selected_student.get('id', '')
        student_section = self.selected_student.get('year-section', '')

        if not self.main_window:
            # No loading dialog available — run synchronously
            self.system.mirror_student_photos(student_id, student_name, student_section)
            student_folder = self.system.get_student_folder_path(student_id, student_name, student_section)
            if not self.system.validate_student_cache(student_folder):
                self.system.build_student_cache(student_folder)
            self._data_ready = True
            self.verify_btn.setEnabled(True)
            return

        loading = ModernLoadingDialog(
            f"Preparing {student_name}",
            "Syncing photos and building face data...",
            self.main_window
        )

        result = [True]  # assume success

        import threading
        def worker():
            try:
                self.system.mirror_student_photos(student_id, student_name, student_section)
                student_folder = self.system.get_student_folder_path(student_id, student_name, student_section)
                if not self.system.validate_student_cache(student_folder):
                    built = self.system.build_student_cache(student_folder)
                    if not built:
                        result[0] = False
            except Exception as e:
                print(f"⚠️ Data prep error: {e}")
                result[0] = False
            finally:
                QMetaObject.invokeMethod(loading, "accept", Qt.QueuedConnection)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        loading.exec_()

        self._data_ready = result[0]
        if self._data_ready:
            self.verify_btn.setEnabled(True)
            self._show_status(f"✅ {student_name} ready for verification")
        else:
            # No photos or cache build failed — still let them try but warn in status bar
            self._data_ready = False
            self.verify_btn.setEnabled(True)  # let them try; scanner will catch it
            self._show_status(f"⚠️ No face data found for {student_name} — upload photos first")

    # ========== VERIFICATION METHODS ==========

    def check_student_cache(self, student):
        """Check if student has face encodings cached"""
        try:
            student_id = student.get('id')
            student_name = f"{student.get('fname', '')} {student.get('lname', '')}".strip()
            section = student.get('year-section')
            student_folder = self.system.get_student_folder_path(student_id, student_name, section)
            return self.system.validate_student_cache(student_folder)
        except Exception:
            return False

    def start_verification(self):
        """Open the face scanner. Photos and cache are already prepared on student selection."""
        if not self.selected_student:
            return

        # Fast cache check — if background prep already ran this is instant
        if not self.check_student_cache(self.selected_student):
            student_name    = f"{self.selected_student.get('fname', '')} {self.selected_student.get('lname', '')}".strip()
            self._show_status(f"⚠️ No face data for {student_name} — make sure photos are uploaded")
            return

        self._proceed_to_scanner()

    def _proceed_to_scanner(self):
        """Open face scanner for verification"""
        self.system.known_faces.clear()
        self.system.known_names.clear()
        self.system.known_student_ids.clear()

        student_id = self.selected_student.get('id')
        section = self.selected_student.get('year-section')
        
        section_path = os.path.join(Config.LOCAL_STORAGE, 'sections', section, 'students')
        if os.path.exists(section_path):
            for folder in os.listdir(section_path):
                if folder.startswith(f'{student_id}_'):
                    cache_file = os.path.join(section_path, folder, 'encoding.npy')
                    if os.path.exists(cache_file):
                        cache_data = np.load(cache_file, allow_pickle=True).item()
                        self.system.known_faces = cache_data.get('encodings', [])
                        self.system.known_names = [cache_data.get('name', 'Unknown')] * len(self.system.known_faces)
                        break

        self.progress_frame.setVisible(False)
        if len(self.system.known_faces) == 0:
            QMessageBox.critical(self, 'Error', 'Could not load face data. Please try again.')
            return

        # Build distance callable from main window's cached value
        mw = self.main_window
        distance_fn = (lambda: mw._last_sensor_distance) if (mw and mw.hardware) else None

        # Pass hardware to verification dialog
        self.verification_dialog = FaceVerificationDialog(
            self.system,
            self.selected_student,
            mw.hardware if mw else None,
            self,
            distance_fn=distance_fn
        )
        self.verification_dialog.verification_result.connect(self._on_verification_result)
        self.verification_dialog.exec_()

    def _on_verification_result(self, success, message):
        """Handle verification result"""
        if not success:
            self.verified = False
            self._update_student_card()
            self._lock_document_controls()
            if self.main_window and not self.main_window.system.current_schedule:
                for btn in self.main_window.nav_buttons:
                    btn.setEnabled(True)
            # Result already shown inline in the dialog — no popup needed
            self._show_status(f"❌ Verification failed: {message}")
            return

        self.verified = True
        self._update_student_card()
        self.doc_type_combo.setEnabled(True)
        
        # Load student terms
        self.term_combo.setEnabled(False)
        self.term_combo.clear()
        self.term_combo.addItem('Loading academic periods...', None)
        self.logic.load_student_terms(self.selected_student)
        
        self._show_status(message)

    def _show_status(self, message):
        """Show status message in main window if available"""
        if self.main_window and hasattr(self.main_window, 'status_bar'):
            self.main_window.status_bar.showMessage(message, 3000)
        print(message)

    # ========== DOCUMENT FETCHING ==========

    def on_terms_loaded(self, formatted_terms):
        """Handle loaded terms"""
        self.term_combo.clear()
        if not self.verified:
            self.term_combo.addItem('Verify student first', None)
            self.term_combo.setEnabled(False)
            return
        
        if not formatted_terms:
            self.term_combo.addItem('No academic periods found', None)
            self.term_combo.setEnabled(False)
            self.fetch_btn.setEnabled(False)
            return
        
        for term in formatted_terms:
            self.term_combo.addItem(term['display'], term['raw'])
        self.term_combo.setEnabled(True)
        self.term_combo.setCurrentIndex(0)
        self._update_fetch_button_state()

    def fetch_document(self):
        """Fetch the selected document"""
        if not self.verified or not self.selected_student:
            return
        
        doc_type = self.doc_type_combo.currentData()
        term_raw = self.term_combo.currentData()
        if not doc_type or not term_raw:
            return
        
        self.fetch_btn.setEnabled(False)
        self.logic.fetch_document(doc_type, term_raw)

    def on_progress_update(self, progress, message):
        """Handle progress updates"""
        self.progress_frame.setVisible(True)
        if self.progress_bar.maximum() == 0 and progress >= 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        if progress >= 100:
            QTimer.singleShot(1500, lambda: self.progress_frame.setVisible(False))

    def on_document_error(self, error):
        """Handle document error"""
        self.progress_frame.setVisible(False)
        self.fetch_btn.setEnabled(True)
        self._show_status(error)
        QMessageBox.warning(self, 'Document Error', error)

    def on_document_ready(self, pdf_path):
        """Handle document ready"""
        self.current_pdf_path = pdf_path
        self.display_pdf(pdf_path)
        self.fetch_btn.setEnabled(True)
        self.print_btn.setEnabled(True)
        self.close_doc_btn.setEnabled(True)
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self._show_status('Document ready')

    def reset_document_flow(self):
        """Reset the entire document workflow"""
        self.verified = False
        self.selected_student = None
        self.year_combo.setCurrentIndex(0)
        self.section_combo.clear()
        self.section_combo.addItem('-- Select Section --', None)
        self.student_combo.clear()
        self.student_combo.addItem('-- Select Student --', None)
        self.verify_btn.setEnabled(False)
        self._lock_document_controls()
        self.cleanup_document()
        self._update_student_card()

class ModernAttendanceUI(QMainWindow):
    """Main modern Qt UI for attendance system"""

    sensor_scan_requested = pyqtSignal(float)
    refresh_ui_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.system = AttendanceSystem()
        self.attendance_model = AttendanceTableModel()
        self.upcoming_window = None
        self.documents_window = None
        self.doc_access_window = None
        self.ui_scale = 1.0
        self.compact_mode = False
        self._scan_dialog_open = False

        # ===== ADD THIS: Initialize hardware controller =====
        self.hardware = None
        self._hardware_active = False
        self._last_sensor_distance = None   # cache — updated by sensor thread & debug timer
        self._sensor_auto_enabled = False   # controlled by the toggle button only
        
        if HARDWARE_AVAILABLE:
            try:
                self.hardware = HardwareController(
                    trig_pin=23,
                    echo_pin=24,
                    ir_led_pin=18,
                    detection_distance=0.5,      # 50cm detection range
                    lost_target_delay=2.0,       # Wait 2 seconds before turning off
                    cooldown_seconds=2.0         # 2 second cooldown after each scan
                )
                print("✅ Hardware controller initialized (sensor DISABLED by default)")
                
                # Wire both callbacks
                # Connect hardware signals to UI slots
                self.hardware.person_detected_signal.connect(self.on_hardware_person_detected)
                self.hardware.person_left_signal.connect(self.on_hardware_person_left)
                self.hardware.scan_complete_signal.connect(self.on_hardware_scan_complete)
                
            except Exception as e:
                print(f"⚠️ Hardware controller init failed: {e}")
                self.hardware = None
        # ===== END ADD =====

        self.init_ui()
        self.start_timers()
        self.system.start_schedule_sync_loop(self.on_schedules_synced)
        self.system.sync_schedules(sync_students=True)
        self.sensor_scan_requested.connect(self._handle_sensor_scan_request)
        self.refresh_ui_requested.connect(self.refresh_display)

        # ===== Sensor debug overlay (bottom-left of status bar) =====
        self.sensor_debug_label = QLabel("⬤ Sensor: OFF")
        self.sensor_debug_label.setStyleSheet("color: #475569; font-size: 11px; padding: 2px 8px;")
        self.status_bar.addWidget(self.sensor_debug_label)  # addWidget = left side

        self.sensor_debug_timer = QTimer()
        self.sensor_debug_timer.timeout.connect(self.update_sensor_debug)
        self.sensor_debug_timer.start(250)  # Update 4x per second
        # ===== END sensor debug =====

        self.refresh_display()

    def on_schedules_synced(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}]  Schedule sync callback received")
        self.refresh_ui_requested.emit()
        # Re-arm the precise timer whenever schedules are refreshed
        self.schedule_precise_start_timer()

    def run_with_loading(self, title, message, func, *args, **kwargs):
        """Run a function with loading screen - FIXED VERSION"""
        loading = ModernLoadingDialog(title, message, self)
        
        result = [None]
        error = [None]
        finished = [False]
        
        # Create a thread to run the actual function
        import threading
        def worker():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                error[0] = e
            finally:
                finished[0] = True
                # Use QTimer to safely close dialog from main thread
                QMetaObject.invokeMethod(loading, "accept", Qt.QueuedConnection)
        
        # Start the worker thread
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        # Show dialog - this blocks until loading.accept() is called
        loading.exec_()
        
        # Wait a bit for thread to finish if needed
        if not finished[0]:
            thread.join(timeout=0.5)
        
        if error[0]:
            raise error[0]
        return result[0]


    def calculate_ui_scale(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1.0, False, QRect(0, 0, 1024, 600)
        geo = screen.availableGeometry()
        # Tune the UI around a 1024x600 Raspberry Pi display first, then allow mild scaling upward.
        scale = min(geo.width() / 1024.0, geo.height() / 600.0, 1.18)
        scale = max(0.88, scale)
        compact = geo.width() <= 1100 or geo.height() <= 650
        return scale, compact, geo

    def scaled(self, value):
        return max(1, int(value * self.ui_scale))

    def font_px(self, compact_px, regular_px=None):
        """Return a font size tuned for the current window mode."""
        if regular_px is None:
            regular_px = compact_px
        return self.scaled(compact_px if self.compact_mode else regular_px)

    def build_dynamic_stylesheet(self):
        base_font = self.font_px(12, 14)
        radius = self.scaled(14 if self.compact_mode else 16)
        btn_pad_y = self.scaled(9 if self.compact_mode else 11)
        btn_pad_x = self.scaled(14 if self.compact_mode else 18)
        field_pad_y = self.scaled(8 if self.compact_mode else 10)
        field_pad_x = self.scaled(10 if self.compact_mode else 12)
        min_btn_h   = self.scaled(40 if self.compact_mode else 44)
        min_field_h = self.scaled(34 if self.compact_mode else 36)
        return MODERN_STYLE + f"""
            QMainWindow {{ background: #0b1220; }}
            QWidget {{ background: #0f172a; color: #e5e7eb; font-family: 'Segoe UI', 'Inter', 'Arial'; font-size: {base_font}px; }}
            QFrame#sidebar {{ background: #111827; border-right: 1px solid #253046; }}
            QFrame#topbar, QFrame#panel, QFrame#heroCard, QFrame#classCard, QFrame#cameraCard, QFrame#brandCard {{
                background: #111827; border: 1px solid #253046; border-radius: {radius}px;
            }}
            QFrame#accentBar {{ background: #8b5cf6; border: none; border-radius: 4px; }}
            QPushButton {{
                background: #7c3aed; border: none; border-radius: {self.scaled(10)}px;
                color: white; padding: {btn_pad_y}px {btn_pad_x}px;
                font-weight: 700; font-size: {base_font}px; min-height: {min_btn_h}px;
            }}
            QPushButton:hover {{ background: #8b5cf6; }}
            QPushButton:pressed {{ background: #6d28d9; }}
            QPushButton#ghostBtn, QPushButton#navBtn {{
                background: transparent; border: 1px solid #253046; color: #d1d5db;
                text-align: left; padding: {btn_pad_y}px {btn_pad_x}px;
                font-size: {base_font}px; min-height: {min_btn_h}px;
            }}
            QPushButton#ghostBtn:hover, QPushButton#navBtn:hover {{ background: #182235; border: 1px solid #334155; }}
            QPushButton#navBtn[active='true'] {{ background: #1e293b; border: 1px solid #8b5cf6; color: white; font-weight: 800; }}
            QListWidget {{ background: transparent; border: none; }}
            QListWidget::item {{ margin-bottom: {self.scaled(6)}px; padding: {self.scaled(6)}px; font-size: {base_font}px; }}
            QLineEdit, QComboBox {{
                background: #0b1220; border: 1px solid #253046;
                border-radius: {self.scaled(10)}px; padding: {field_pad_y}px {field_pad_x}px;
                min-height: {min_field_h}px; font-size: {base_font}px;
            }}
            QLineEdit:focus, QComboBox:focus {{ border: 1px solid #8b5cf6; }}
            QComboBox::drop-down {{ width: {self.scaled(28)}px; border: none; }}
            QTableWidget, QTreeView {{
                background: #0b1220; border: 1px solid #253046;
                border-radius: {self.scaled(12)}px; gridline-color: #1f2937;
                alternate-background-color: #111827; font-size: {base_font}px;
            }}
            QTableWidget::item, QTreeView::item {{ padding: {self.scaled(10)}px; min-height: {self.scaled(32)}px; }}
            QHeaderView::section {{
                background: #111827; color: #dbeafe; border: none;
                padding: {self.scaled(10)}px; font-weight: 700; font-size: {base_font}px;
            }}
            QProgressBar {{
                background: #0b1220; border: 1px solid #253046;
                border-radius: {self.scaled(10)}px; text-align: center; min-height: {self.scaled(14)}px;
            }}
            QProgressBar::chunk {{ background: #8b5cf6; border-radius: {self.scaled(9)}px; }}
            QStatusBar {{ background: #111827; color: #cbd5e1; border-top: 1px solid #253046; font-size: {base_font}px; }}
            QScrollBar:vertical {{ background: #111827; width: {self.scaled(14)}px; border-radius: {self.scaled(7)}px; }}
            QScrollBar::handle:vertical {{ background: #334155; border-radius: {self.scaled(7)}px; min-height: {self.scaled(30)}px; }}
            QScrollBar::handle:vertical:hover {{ background: #8b5cf6; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QLabel {{ font-size: {base_font}px; }}
        """

    def init_ui(self):
        self.ui_scale, self.compact_mode, geo = self.calculate_ui_scale()
        self.setWindowTitle('ADV-Attendify - Face Recognition Attendance System')
        self.setMinimumSize(800, 480)
        self.resize(max(1024, int(geo.width() * 0.98)), max(600, int(geo.height() * 0.98)))
        self.setStyleSheet(self.build_dynamic_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        self.root_layout = QHBoxLayout(central)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        self.sidebar = self.build_sidebar()
        self.content = self.build_content_area()
        self.root_layout.addWidget(self.sidebar)
        self.root_layout.addWidget(self.content, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('System ready')

        self.set_page(1)
        self.apply_responsive_layout()

    def build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar_width = self.scaled(210) if self.compact_mode else self.scaled(250)
        sidebar.setMinimumWidth(sidebar_width)
        sidebar.setMaximumWidth(sidebar_width)
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(self.scaled(14), self.scaled(14), self.scaled(14), self.scaled(14))
        layout.setSpacing(self.scaled(12))

        brand = QFrame()
        brand.setObjectName('brandCard')
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(16, 16, 16, 16)
        accent = QFrame()
        accent.setObjectName('accentBar')
        accent.setFixedSize(self.scaled(34), max(3, self.scaled(4)))
        title = QLabel('ADV-Attendify')
        title.setWordWrap(False)
        title.setStyleSheet(f'font-size:{self.font_px(17, 21)}px; font-weight:900; color:white;')
        sub = QLabel('Face Recognition System')
        sub.setWordWrap(False)
        sub.setStyleSheet(f'color:#94a3b8; font-size:{self.font_px(10, 12)}px;')
        brand_layout.addWidget(accent)
        brand_layout.addWidget(title)
        brand_layout.addWidget(sub)
        layout.addWidget(brand)

        self.nav_buttons = []
        labels = ['Attendance Records', 'Class Schedule', 'Student Documents']
        for index, text in enumerate(labels):
            btn = QPushButton(text)
            btn.setObjectName('navBtn')
            btn.setProperty('active', False)
            btn.setMinimumHeight(self.scaled(48) if self.compact_mode else self.scaled(56))
            btn.clicked.connect(lambda checked=False, i=index: self.set_page(i))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()
        return sidebar

    def wrap_in_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def build_content_area(self):
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(self.scaled(10), self.scaled(10), self.scaled(10), self.scaled(10))
        outer.setSpacing(self.scaled(10))

        topbar = QFrame()
        topbar.setObjectName('topbar')
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(self.scaled(12), self.scaled(8), self.scaled(12), self.scaled(8))
        top_layout.setSpacing(self.scaled(8))

        self.page_title = QLabel('Class Schedule')
        self.page_title.setStyleSheet(f'font-size:{self.font_px(16, 20)}px; font-weight:800; color:white;')

        clock_container = QWidget()
        clock_layout = QVBoxLayout(clock_container)
        clock_layout.setContentsMargins(0, 0, 0, 0)
        clock_layout.setSpacing(0)
        self.time_label = QLabel()
        self.time_label.setStyleSheet(f'font-size:{self.font_px(18, 24)}px; font-weight:800; color:#67e8f9;')
        self.date_label = QLabel()
        self.date_label.setWordWrap(False)
        self.date_label.setStyleSheet(f'color:#cbd5e1; font-size:{self.font_px(11, 13)}px;')
        clock_layout.addWidget(self.time_label, alignment=Qt.AlignRight)
        clock_layout.addWidget(self.date_label, alignment=Qt.AlignRight)

        self.fullscreen_btn = QPushButton('Fullscreen')
        self.fullscreen_btn.setMinimumWidth(self.scaled(96) if self.compact_mode else self.scaled(120))
        self.fullscreen_btn.setObjectName('ghostBtn')
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        top_layout.addWidget(self.page_title)
        top_layout.addStretch()
        top_layout.addWidget(clock_container)
        top_layout.addWidget(self.fullscreen_btn)
        outer.addWidget(topbar)

        self.main_stack = QStackedWidget()
        self.attendance_records_page = LocalAttendanceRecordsPage()
        self.main_stack.addWidget(self.wrap_in_scroll(self.attendance_records_page))

        self.classes_page = QWidget()
        classes_layout = QVBoxLayout(self.classes_page)
        classes_layout.setContentsMargins(0, 0, 0, 0)
        classes_layout.setSpacing(16)
        self.classes_stack = QStackedWidget()
        self.classes_overview_page = self.create_classes_overview_page()
        self.active_class_widget = self.create_active_class_widget()
        self.classes_stack.addWidget(self.classes_overview_page)
        self.classes_stack.addWidget(self.active_class_widget)
        classes_layout.addWidget(self.classes_stack)
        self.main_stack.addWidget(self.wrap_in_scroll(self.classes_page))

        self.documents_page = IntegratedDocumentsPage(self.system, self, self)
        self.main_stack.addWidget(self.wrap_in_scroll(self.documents_page))

        outer.addWidget(self.main_stack, 1)
        return wrapper

    def create_classes_overview_page(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self.scaled(14))

        left = QFrame()
        left.setObjectName('panel')
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(self.scaled(18), self.scaled(18), self.scaled(18), self.scaled(18))
        left_layout.setSpacing(self.scaled(12))

        title = QLabel('Class Schedule')
        title.setStyleSheet(f'font-size:{self.font_px(18, 24)}px; font-weight:800; color:white;')
        desc = QLabel("Today's class overview and quick actions")
        desc.setStyleSheet(f'color:#cbd5e1; font-size:{self.font_px(11, 13)}px;')
        left_layout.addWidget(title)
        left_layout.addWidget(desc)





        self.manual_refresh_btn = QPushButton('Refresh Schedules')
        self.manual_refresh_btn.setObjectName('ghostBtn')
        self.manual_refresh_btn.setMinimumHeight(self.scaled(44) if self.compact_mode else self.scaled(50))
        self.manual_refresh_btn.clicked.connect(self.manual_refresh_with_loading)
        left_layout.addWidget(self.manual_refresh_btn)
        left_layout.addStretch()

        right = QFrame()
        right.setObjectName('panel')
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(self.scaled(18), self.scaled(18), self.scaled(18), self.scaled(18))
        right_layout.setSpacing(self.scaled(12))

        next_title = QLabel('Next Class')
        next_title.setStyleSheet(f'font-size:{self.font_px(17, 22)}px; font-weight:800; color:white;')
        right_layout.addWidget(next_title)

        self.next_class_card = QFrame()
        self.next_class_card.setObjectName('classCard')
        card_layout = QVBoxLayout(self.next_class_card)
        card_layout.setContentsMargins(self.scaled(16), self.scaled(16), self.scaled(16), self.scaled(16))
        card_layout.setSpacing(self.scaled(10))

        self.next_subject = QLabel('No upcoming classes')
        self.next_subject.setStyleSheet(f'font-size:{self.font_px(18, 24)}px; font-weight:800; color:white;')
        self.next_teacher = QLabel('')
        self.next_teacher.setStyleSheet(f'color:#cbd5e1; font-size:{self.font_px(12, 15)}px;')
        self.next_room = QLabel('')
        self.next_room.setStyleSheet(f'color:#cbd5e1; font-size:{self.font_px(12, 15)}px;')
        self.next_time = QLabel('')
        self.next_time.setStyleSheet(f'color:#67e8f9; font-weight:700; font-size:{self.font_px(13, 16)}px;')
        self.next_countdown = QLabel('')
        self.next_countdown.setStyleSheet(f'color:#8b5cf6; font-size:{self.font_px(22, 30)}px; font-weight:800;')

        card_layout.addWidget(self.next_subject)
        card_layout.addWidget(self.next_teacher)
        card_layout.addWidget(self.next_room)
        card_layout.addWidget(self.next_time)
        card_layout.addSpacing(8)
        card_layout.addWidget(self.next_countdown)
        right_layout.addWidget(self.next_class_card)
        right_layout.addStretch()

        layout.addWidget(left, 2)
        layout.addWidget(right, 3)
        return widget

    def check_hardware_monitoring(self):
        """Deactivate hardware when class ends or page changes.
        The sensor is ONLY started by the toggle button — never automatically.
        The IR LED is managed inside the scan dialogs themselves.
        """
        if not self.hardware:
            return

        on_classes_page = self.main_stack.currentIndex() == 1
        class_active    = bool(self.system.current_schedule)

        # Deactivate if we left the classes page or the class ended
        if self._hardware_active and (not on_classes_page or not class_active):
            reason = "no active class" if not class_active else "not on classes page"
            print(f"🔇 Deactivating hardware ({reason})...")
            self.hardware.stop_monitoring()
            self.hardware.disable_sensor()
            self._hardware_active = False
            self._sensor_auto_enabled = False
            self._update_sensor_toggle_btn()
            print("✅ Hardware monitoring INACTIVE")

    def toggle_sensor(self):
        """Enable / disable the 1m auto-scan sensor via the button."""
        if not self.hardware:
            return
        if self._sensor_auto_enabled:
            # Turn OFF
            self.hardware.stop_monitoring()
            self.hardware.disable_sensor()
            self._hardware_active = False
            self._sensor_auto_enabled = False
            print("🔇 Auto-scan sensor disabled by user")
            self.status_bar.showMessage("⬤ Auto-scan sensor OFF", 2000)
        else:
            # Turn ON — only when a class is active
            if not self.system.current_schedule:
                QMessageBox.warning(self, 'No Active Class',
                                    'Start a class before enabling the sensor.')
                return
            self.hardware.enable_sensor()
            self.hardware.start_monitoring()
            self._hardware_active = True
            self._sensor_auto_enabled = True
            print("🔊 Auto-scan sensor enabled by user")
            self.status_bar.showMessage("🟢 Auto-scan sensor ON — detecting within 1m", 2000)
        self._update_sensor_toggle_btn()

    def _update_sensor_toggle_btn(self):
        """Sync the toggle button label and colour to current state."""
        if not hasattr(self, 'sensor_toggle_btn'):
            return
        if not self.hardware:
            self.sensor_toggle_btn.setVisible(False)
            return
        if self._sensor_auto_enabled:
            self.sensor_toggle_btn.setText('🟢 Auto-Scan: ON')
            self.sensor_toggle_btn.setStyleSheet(
                f'QPushButton {{ background:#065f46; color:white; border:none;'
                f' border-radius:{self.scaled(10)}px; padding:{self.scaled(10)}px {self.scaled(16)}px;'
                f' font-weight:700; font-size:{self.font_px(12, 14)}px; }}'
                f'QPushButton:hover {{ background:#047857; }}')
        else:
            self.sensor_toggle_btn.setText('🔴 Auto-Scan: OFF')
            self.sensor_toggle_btn.setStyleSheet(
                f'QPushButton {{ background:#374151; color:#9ca3af; border:1px solid #4b5563;'
                f' border-radius:{self.scaled(10)}px; padding:{self.scaled(10)}px {self.scaled(16)}px;'
                f' font-weight:700; font-size:{self.font_px(12, 14)}px; }}'
                f'QPushButton:hover {{ background:#4b5563; color:white; }}')

    def on_hardware_person_detected(self, distance):
        """Called from the sensor thread. Never open Qt dialogs here."""
        self._last_sensor_distance = distance  # cache for main-thread use
        print(f"👤 Student detected at {distance:.2f}m")

        if not self.system.current_schedule:
            print("⚠️ Sensor detected a person, but there is no active class")
            return

        if self._scan_dialog_open:
            print("⚠️ Scan dialog already open - sensor trigger ignored")
            return

        # Queue the exact same scan function used by the Scan a Student button
        self.sensor_scan_requested.emit(distance)

    @pyqtSlot(float)
    def _handle_sensor_scan_request(self, distance):
        """Runs on the Qt main thread."""
        if self._scan_dialog_open:
            return
        if not self.system.current_schedule:
            return
        if self.main_stack.currentIndex() != 1:
            print("⚠️ Sensor trigger ignored - not on classes page")
            return
        print(f"📷 Sensor trigger at {distance:.2f}m - opening scanner")
        self.status_bar.showMessage(f"👤 Student detected at {distance:.2f}m - opening scanner", 2000)
        self.scan_face()

    def _on_scanner_closed(self):
        """Called when scanner dialog closes"""
        self._scan_dialog_open = False

    def on_hardware_person_left(self):
        """Called when person leaves sensor range"""
        print("👤 Student left sensor range")
        self.status_bar.showMessage(f"👤 Student left - waiting for next student", 2000)

    def on_hardware_scan_complete(self):
        """Called when a face scan is completed - starts hardware cooldown"""
        print("⏱️ Scan complete - hardware entering cooldown")
        self.status_bar.showMessage("✅ Scan complete - sensor cooling down (2 seconds)", 2000)

    def update_sensor_debug(self):
        """Update the live sensor debug label in the status bar (runs every 250ms).
        This is the ONLY place that calls get_distance() directly — it's safe here
        because it runs on the main thread and is the dedicated polling loop.
        scan_face() and dialogs use _last_sensor_distance to avoid thread races.
        """
        if not hasattr(self, 'sensor_debug_label'):
            return

        if not self.hardware:
            self.sensor_debug_label.setText("⬤ Sensor: N/A")
            self.sensor_debug_label.setStyleSheet("color: #475569; font-size: 11px; padding: 2px 8px;")
            return

        if self.hardware.is_in_cooldown():
            self.sensor_debug_label.setText("⬤ Sensor: COOLDOWN")
            self.sensor_debug_label.setStyleSheet("color: #f59e0b; font-size: 11px; padding: 2px 8px;")
            return

        try:
            dist = self.hardware.get_distance()
            self._last_sensor_distance = dist  # keep cache fresh for dialogs

            too_close = dist <= 0.3
            detected  = not too_close and dist <= self.hardware.detection_distance

            if not self._hardware_active:
                label = f"⬤ Sensor: OFF  {dist:.2f}m"
                color = "#475569"
            elif too_close:
                label = f"🔴 TOO CLOSE  {dist:.2f}m"
                color = "#ef4444"
            elif detected:
                label = f"🟢 DETECTED  {dist:.2f}m"
                color = "#10b981"
            else:
                label = f"⬤ CLEAR  {dist:.2f}m"
                color = "#06b6d4"

            self.sensor_debug_label.setText(label)
            self.sensor_debug_label.setStyleSheet(
                f"color: {color}; font-size: 11px; padding: 2px 8px;")
        except Exception:
            self.sensor_debug_label.setText("⬤ Sensor: ERR")
            self.sensor_debug_label.setStyleSheet("color: #ef4444; font-size: 11px; padding: 2px 8px;")



    def manual_refresh_with_loading(self):
        """Manual refresh with loading screen"""
        loading = ModernLoadingDialog(
            "Refreshing Schedules",
            "Checking for updated schedules...",
            self
        )
        
        import threading
        def worker():
            try:
                # Call the existing method directly
                self.system.sync_schedules(sync_students=False)
                self.refresh_ui_requested.emit()
            finally:
                # Close loading dialog when done
                QMetaObject.invokeMethod(loading, "accept", Qt.QueuedConnection)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        loading.exec_()

    def load_section_with_loading(self, section, allow_build=False):
        """Load section students with loading screen"""
        loading = ModernLoadingDialog(
            f"Loading Section {section}",
            f"Preparing face data for {section}...",
            self
        )
        
        result = [0]
        
        import threading
        def worker():
            try:
                # Pass allow_build to the core method
                result[0] = self.system.load_section_students(section, allow_build=allow_build)
            finally:
                QMetaObject.invokeMethod(loading, "accept", Qt.QueuedConnection)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        loading.exec_()
        return result[0]

    def create_active_class_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self.scaled(14))

        left_panel = QFrame()
        left_panel.setObjectName('panel')
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(self.scaled(16), self.scaled(16), self.scaled(16), self.scaled(16))
        left_layout.setSpacing(self.scaled(12))

        title = QLabel('Active Class')
        title.setStyleSheet(f'font-size:{self.font_px(18, 24)}px; font-weight:800; color:white;')
        subtitle = QLabel('Live attendance session')
        subtitle.setStyleSheet(f'color:#cbd5e1; font-size:{self.font_px(11, 13)}px;')
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)

        self.class_info_card = QFrame()
        self.class_info_card.setObjectName('panel')
        info_layout = QVBoxLayout(self.class_info_card)
        info_layout.setContentsMargins(self.scaled(15), self.scaled(15), self.scaled(15), self.scaled(15))
        info_layout.setSpacing(self.scaled(10))

        self.class_info_labels = {}
        info_items = [
            ' Teacher',
            ' Room',
            ' Time',
            ' Duration',
            ' Students Loaded',
            ' Present'
        ]
        for item in info_items:
            label = QLabel(f'{item}: -')
            label.setStyleSheet(f'font-size:{self.font_px(11, 14)}px; color:#e5e7eb;')
            info_layout.addWidget(label)
            self.class_info_labels[item] = label
        left_layout.addWidget(self.class_info_card)

        countdown_card = QFrame()
        countdown_card.setObjectName('panel')
        countdown_layout = QVBoxLayout(countdown_card)
        countdown_layout.setContentsMargins(self.scaled(15), self.scaled(15), self.scaled(15), self.scaled(15))
        countdown_title = QLabel('Class Ends In')
        countdown_title.setStyleSheet(f'font-size:{self.font_px(13, 16)}px; font-weight:700; color:white;')
        self.countdown_label = QLabel('Calculating...')
        self.countdown_label.setStyleSheet(f'font-size:{self.font_px(24, 36)}px; font-weight:800; color:#8b5cf6;')
        self.countdown_label.setAlignment(Qt.AlignCenter)
        countdown_layout.addWidget(countdown_title)
        countdown_layout.addWidget(self.countdown_label)
        left_layout.addWidget(countdown_card)

        self.scan_btn = QPushButton('Scan Student')
        self.scan_btn.setMinimumHeight(self.scaled(44) if self.compact_mode else self.scaled(50))
        self.scan_btn.setStyleSheet(f'QPushButton {{ background:#7c3aed; color:white; border:none; border-radius:{self.scaled(10)}px; padding:{self.scaled(10)}px {self.scaled(16)}px; font-weight:700; font-size:{self.font_px(12, 14)}px; }} QPushButton:hover {{ background:#8b5cf6; }}')
        self.scan_btn.clicked.connect(self.scan_face)
        left_layout.addWidget(self.scan_btn)

        # Toggle button — only visible when hardware is connected
        self.sensor_toggle_btn = QPushButton('🔴 Auto-Scan: OFF')
        self.sensor_toggle_btn.setMinimumHeight(self.scaled(44) if self.compact_mode else self.scaled(46))
        self.sensor_toggle_btn.clicked.connect(self.toggle_sensor)
        self.sensor_toggle_btn.setVisible(bool(self.hardware))
        left_layout.addWidget(self.sensor_toggle_btn)
        self._update_sensor_toggle_btn()

        self.end_class_btn = QPushButton('End Class')
        self.end_class_btn.setMinimumHeight(self.scaled(44) if self.compact_mode else self.scaled(50))
        self.end_class_btn.clicked.connect(self.end_class)
        self.end_class_btn.setStyleSheet(f'QPushButton {{ background:#ef4444; color:white; border:none; border-radius:{self.scaled(10)}px; padding:{self.scaled(10)}px {self.scaled(16)}px; font-weight:700; font-size:{self.font_px(12, 14)}px; }} QPushButton:hover {{ background:#dc2626; }}')
        left_layout.addWidget(self.end_class_btn)

        back_btn = QPushButton('Back to Schedule')
        back_btn.setObjectName('ghostBtn')
        back_btn.setMinimumHeight(self.scaled(42) if self.compact_mode else self.scaled(46))
        back_btn.clicked.connect(lambda: self.set_page(1))
        left_layout.addWidget(back_btn)
        left_layout.addStretch()

        right_panel = QFrame()
        right_panel.setObjectName('panel')
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(self.scaled(16), self.scaled(16), self.scaled(16), self.scaled(16))
        right_layout.setSpacing(self.scaled(10))

        attendance_header = QLabel('Attendance Register')
        attendance_header.setStyleSheet(f'font-size:{self.font_px(16, 20)}px; font-weight:800; color:white;')
        right_layout.addWidget(attendance_header)

        self.attendance_table = QTreeView()
        self.attendance_table.setModel(self.attendance_model)
        self.attendance_table.setAlternatingRowColors(True)
        self.attendance_table.setSelectionBehavior(QTreeView.SelectRows)
        self.attendance_table.setEditTriggers(QTreeView.NoEditTriggers)
        self.attendance_table.setRootIsDecorated(False)
        self.attendance_table.setItemsExpandable(False)
        self.attendance_table.setColumnWidth(0, self.scaled(220))
        self.attendance_table.setColumnWidth(1, self.scaled(110))
        self.attendance_table.setColumnWidth(2, self.scaled(100))
        right_layout.addWidget(self.attendance_table, 1)

        layout.addWidget(left_panel, 40)
        layout.addWidget(right_panel, 60)
        return widget

    def create_documents_shell_page(self):
        widget = QWidget()
        root = QHBoxLayout(widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        left = QFrame()
        left.setObjectName('panel')
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(14)

        title = QLabel('Student Document Access')
        title.setStyleSheet('font-size:24px; font-weight:800; color:white;')
        left_layout.addWidget(title)

        note = QLabel('Student ID search removed. Use the document section layout below.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#cbd5e1;')
        left_layout.addWidget(note)

        combos = [
            ('Year Level', ['1st Year', '2nd Year', '3rd Year', '4th Year']),
            ('Section', ['Section 1A', 'Section 1B', 'Section 2A', 'Section 3A']),
            ('Student', ['Select from the verification window'])
        ]
        for label_text, values in combos:
            lbl = QLabel(label_text)
            lbl.setStyleSheet('font-weight:700; color:#dbeafe;')
            combo = QComboBox()
            combo.addItems(values)
            combo.setEnabled(False)
            left_layout.addWidget(lbl)
            left_layout.addWidget(combo)

        verify_btn = QPushButton('Verify Identity')
        verify_btn.clicked.connect(lambda: self.set_page(2))
        open_btn = QPushButton('Open Record')
        open_btn.setObjectName('ghostBtn')
        open_btn.clicked.connect(lambda: self.set_page(2))
        left_layout.addWidget(verify_btn)
        left_layout.addWidget(open_btn)
        left_layout.addStretch()

        right = QFrame()
        right.setObjectName('cameraCard')
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)
        cam_title = QLabel('Verification Preview')
        cam_title.setStyleSheet('font-size:22px; font-weight:800; color:white;')
        preview = QLabel('Face Verification Placeholder')
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumHeight(420)
        preview.setStyleSheet('border:1px dashed #475569; border-radius:18px; color:#94a3b8; font-size:15px;')
        info = QLabel('The document section keeps the original document logic in the codebase while using this updated layout.')
        info.setWordWrap(True)
        info.setStyleSheet('color:#cbd5e1;')
        right_layout.addWidget(cam_title)
        right_layout.addWidget(preview, 1)
        right_layout.addWidget(info)

        root.addWidget(left, 2)
        root.addWidget(right, 3)
        return widget

    def apply_responsive_layout(self):
        compact_window = self.width() < 1100 or self.height() < 650
        sidebar_width = self.scaled(210 if compact_window else 250)
        self.sidebar.setMinimumWidth(sidebar_width)
        self.sidebar.setMaximumWidth(sidebar_width)
        if hasattr(self, 'fullscreen_btn'):
            self.fullscreen_btn.setText('Full' if self.width() < 1120 else 'Fullscreen')
            self.fullscreen_btn.setMinimumWidth(self.scaled(84) if compact_window else self.scaled(116))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = self.width() < 1100 or self.height() < 650
        new_scale = min(self.width() / 1024.0, self.height() / 600.0, 1.18)
        new_scale = max(0.88, new_scale)
        if compact != self.compact_mode or abs(new_scale - self.ui_scale) > 0.04:
            self.compact_mode = compact
            self.ui_scale = new_scale
            self.setStyleSheet(self.build_dynamic_stylesheet())
        self.apply_responsive_layout()

    def set_page(self, index: int):
        """Set the current page, respecting active class lock"""
        # Index 0 = Attendance Records, 1 = Classes (active class lives here), 2 = Documents
        # During active class, only the classes page (index 1) is allowed
        if self.system.current_schedule and index != 1:
            QMessageBox.warning(self, 'Class Active',
                            'Cannot navigate during active class. End the class first.')
            return
        
        current = self.main_stack.currentIndex()
        if current == 2 and index != 2 and hasattr(self, 'documents_page'):
            self.documents_page.cleanup_document()
        self.main_stack.setCurrentIndex(index)
        titles = ['Attendance Records', 'Class Schedule', 'Student Documents']
        self.page_title.setText(titles[index])
        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty('active', i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
        self.check_hardware_monitoring()
        if index == 0 and hasattr(self, 'attendance_records_page'):
            self.attendance_records_page.load_attendance_records()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def update_active_class_display(self):
        """Update active class display with current data"""
        if not self.system.current_schedule:
            return
        
        schedule = self.system.current_schedule
        
        # Update class info
        self.class_info_labels[' Teacher'].setText(f" Teacher: {schedule['teacher']}")
        self.class_info_labels[' Room'].setText(f" Room: {schedule['classroom']}")
        self.class_info_labels[' Time'].setText(f" Time: {schedule['start_time']}")
        self.class_info_labels[' Duration'].setText(f" Duration: {schedule['duration']} minutes")
        
        # 🔥 Count UNIQUE students, not encodings
        unique_student_count = len(set(self.system.known_student_ids))
        self.class_info_labels[' Students Loaded'].setText(f" Students Loaded: {unique_student_count}")
        self.class_info_labels[' Present'].setText(f" Present: {len(self.system.attendance_log)}")
        
        # Update attendance table
        records = self.system.load_class_attendance(schedule)
        table_data = [[r['name'], r['time'], r['status'].capitalize()] for r in records]
        self.attendance_model.update_data(table_data)
        
        # 🔥 FORCE COLUMN WIDTHS to fix alignment
        # Need to access the view that shows the model
        # In active class, it's a QTreeView (self.attendance_table)
        
        # Set column widths to make them align properly
        self.attendance_table.setColumnWidth(0, 250)  # Name column
        self.attendance_table.setColumnWidth(1, 120)  # Time column
        self.attendance_table.setColumnWidth(2, 100)  # Status column
        
        # Also ensure headers are visible and stretched properly
        self.attendance_table.header().setStretchLastSection(True)
        self.attendance_table.header().setDefaultAlignment(Qt.AlignCenter)

    def refresh_display(self):
        """Refresh the display based on current state"""
        active_class = None
        for sched in self.system.local_schedules:
            if sched.get('status') == 'active':
                active_class = sched
                break

        if active_class:
            self.system.current_schedule = active_class
            self.system.current_section = active_class['section']

            # Load students if not already loaded (startup or page reload)
            if not self.system.known_faces:
                print(f"📚 Loading students for active class: {active_class['section']}")
                count = self.load_section_with_loading(self.system.current_section, allow_build=True)
                self.status_bar.showMessage(f"✅ Loaded {count} students for {self.system.current_section}")

            self.check_hardware_monitoring()
            for btn in self.nav_buttons:
                btn.setEnabled(False)
            self.classes_stack.setCurrentWidget(self.active_class_widget)
            self.update_active_class_display()
            return

        # No active class - re-enable navigation
        for btn in self.nav_buttons:
            btn.setEnabled(True)

        self.check_hardware_monitoring()

        self.system.current_schedule = self.system.get_current_schedule()
        if self.system.current_schedule:
            self.classes_stack.setCurrentWidget(self.active_class_widget)
            self.update_active_class_display()
            if self.system.current_section != self.system.current_schedule['section']:
                self.system.current_section = self.system.current_schedule['section']
                count = self.load_section_with_loading(self.system.current_section, allow_build=True)
                self.status_bar.showMessage(f"✅ Loaded {count} students for {self.system.current_section}")
        else:
            self.classes_stack.setCurrentWidget(self.classes_overview_page)
            self.update_main_menu()

            # Pre-load students for the upcoming section so they're ready at class start
            upcoming_section = self.get_upcoming_section()
            if upcoming_section and not self.system.known_faces:
                print(f"📚 Pre-loading students for upcoming section: {upcoming_section}")
                self.load_section_with_loading(upcoming_section, allow_build=True)

    def get_upcoming_section(self):
        """Get the section of the next upcoming class"""
        schedules = self.system.get_local_schedules()
        now = datetime.now()
        today_date = now.date()
        
        next_class = None
        next_datetime = None
        
        for sched in schedules:
            try:
                sched_date = sched.get('date')
                if sched_date:
                    if isinstance(sched_date, str):
                        sched_date = datetime.strptime(sched_date, '%Y-%m-%d').date()
                else:
                    sched_date = today_date
                    
                if sched_date < today_date:
                    continue
                    
                sched_time = datetime.strptime(sched['start_time'], '%H:%M:%S').time()
                sched_datetime = datetime.combine(sched_date, sched_time)
                
                if sched_datetime > now and (next_datetime is None or sched_datetime < next_datetime):
                    next_datetime = sched_datetime
                    next_class = sched
            except Exception:
                continue
        
        return next_class['section'] if next_class else None

    def update_main_menu(self):
        schedules = self.system.get_local_schedules()
        now = datetime.now()
        today_date = now.date()
        next_class = None
        next_datetime = None
        for sched in schedules:
            try:
                sched_date = sched.get('date')
                if sched_date:
                    if isinstance(sched_date, str):
                        sched_date = datetime.strptime(sched_date, '%Y-%m-%d').date()
                else:
                    sched_date = today_date
                if sched_date < today_date:
                    continue
                sched_time = datetime.strptime(sched['start_time'], '%H:%M:%S').time()
                sched_datetime = datetime.combine(sched_date, sched_time)
                if sched_datetime > now and (next_datetime is None or sched_datetime < next_datetime):
                    next_datetime = sched_datetime
                    next_class = sched
            except Exception:
                continue
        if next_class and next_datetime:
            time_diff = next_datetime - now
            total_seconds = max(int(time_diff.total_seconds()), 0)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            countdown = f'{hours}h {minutes}m' if hours > 0 else f'{minutes}m'
            self.next_subject.setText(next_class['subject'])
            self.next_teacher.setText(f"Teacher: {next_class['teacher']}")
            self.next_room.setText(f"Room: {next_class['classroom']}")
            self.next_time.setText(f"Time: {next_class['start_time']}")
            self.next_countdown.setText(f'Starts in {countdown}')
            self.system.start_background_sync(next_class['section'], self.on_students_synced)
        else:
            self.next_subject.setText('No upcoming classes')
            self.next_teacher.setText('')
            self.next_room.setText('')
            self.next_time.setText('')
            self.next_countdown.setText('')


    def open_document_access(self):
        self.set_page(2)

    def on_students_synced(self, section):
        self.status_bar.showMessage(f' Students synced for {section}')

    def scan_face(self):
        """Open the face scanner. The 0.3m distance check happens inside the dialog."""
        if not self.system.current_schedule:
            QMessageBox.warning(self, 'Error', 'No active class schedule')
            return
        if self._scan_dialog_open:
            return

        self._scan_dialog_open = True
        try:
            # Pass a callable so the dialog can read the cached distance on every
            # frame — never call get_distance() here to avoid thread races.
            dist_fn = (lambda: self._last_sensor_distance) if self.hardware else None
            dialog = FaceScannerDialog(self.system, self.hardware, self,
                                       distance_fn=dist_fn)
            dialog.face_detected.connect(self.on_face_detected)
            dialog.finished.connect(self._on_scanner_closed)
            dialog.exec_()
        finally:
            pass

    def on_face_detected(self, name, status):
        self.update_active_class_display()
        self.status_bar.showMessage(f' {name} marked {status}')

        if self.hardware:
            self.hardware.mark_scan_complete()

    def end_class(self):
        if not self.system.current_schedule:
            return
        reply = QMessageBox.question(
            self, 'End Class',
            f"Are you sure you want to end {self.system.current_schedule['subject']} - {self.system.current_schedule['section']}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            success, data = self.system.end_class()
            if success:
                if self.hardware:
                    self.hardware.stop_monitoring()
                    self.hardware.disable_sensor()
                    self._hardware_active = False
                    self._sensor_auto_enabled = False
                    self._update_sensor_toggle_btn()
                    print("🔇 Hardware disabled - class ended")
                QMessageBox.information(self, 'Class Ended',
                    f" {data['subject']} - {data['section']} ended.\n\n"
                    f" Total present: {data['total_students']} students")
                self.refresh_ui_requested.emit()
                self._print_class_attendance(data['subject'], data['section'])
            else:
                QMessageBox.critical(self, 'Error', f'Failed to end class: {data}')

    def _print_class_attendance(self, subject, section, teacher='', date_str=None):
        import glob as _glob
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        elif not isinstance(date_str, str):
            date_str = str(date_str)
        clean_subj = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_sect = "".join(c for c in section if c.isalnum() or c in ('-', '_')).rstrip()
        matches = sorted(_glob.glob(f"attendance_{clean_subj}_{clean_sect}_{date_str}_*.csv"), reverse=True)
        if not matches:
            for f in sorted(_glob.glob(f"attendance_*_{date_str}_*.csv"), reverse=True):
                try:
                    df = pd.read_csv(f)
                    if ('Subject' in df.columns and 'Section' in df.columns
                            and subject in df['Subject'].values
                            and section in df['Section'].values):
                        matches = [f]; break
                except Exception:
                    continue
        if not matches:
            reply = QMessageBox.question(self, 'Print Attendance',
                f'No attendance file found for {subject} — {section}.\n'
                'Open the Attendance Records page instead?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.set_page(0)
            return
        try:
            df = pd.read_csv(matches[0])
            first = df.iloc[0] if len(df) > 0 else {}
            print_attendance_report(self, df, {
                'subject': subject, 'section': section,
                'teacher': teacher or str(first.get('Teacher', '')),
                'date':    date_str,
            })
        except Exception as e:
            QMessageBox.warning(self, 'Print Error', f'Could not load attendance file:\n{e}')

    def start_timers(self):
        # Clock — every 1 second (also drives the countdown label)
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        # Lifecycle check — every 5 seconds (class end, DB status changes)
        self.class_timer = QTimer()
        self.class_timer.timeout.connect(self.check_class_lifecycle)
        self.class_timer.start(5000)

        # Periodic UI refresh — every 30 seconds
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.refresh_display)
        self.ui_timer.start(30000)

        self.update_clock()
        # Schedule a precise single-shot timer for the next class
        self.schedule_precise_start_timer()

    def schedule_precise_start_timer(self):
        """Set a single-shot QTimer that fires exactly when the next scheduled class starts.
        This removes the polling delay so the class begins within ~1 second of its start time."""
        schedules = self.system.get_local_schedules()
        now = datetime.now()
        today = now.date()

        soonest_ms = None
        for sched in schedules:
            if sched.get('status') in ('active', 'ended'):
                continue
            sched_date = sched.get('date')
            if isinstance(sched_date, str):
                try:
                    sched_date = datetime.strptime(sched_date, '%Y-%m-%d').date()
                except Exception:
                    sched_date = today
            if sched_date != today:
                continue
            try:
                sched_time = datetime.strptime(sched['start_time'], '%H:%M:%S').time()
                sched_dt = datetime.combine(today, sched_time)
                ms_until = int((sched_dt - now).total_seconds() * 1000)
                if ms_until > 0:
                    if soonest_ms is None or ms_until < soonest_ms:
                        soonest_ms = ms_until
            except Exception:
                continue

        if soonest_ms is not None:
            mins = soonest_ms // 60000
            secs = (soonest_ms % 60000) // 1000
            print(f"⏰ Precise start timer set — fires in {mins}m {secs}s")
            QTimer.singleShot(soonest_ms, self._on_precise_class_start)

    def _on_precise_class_start(self):
        """Fires exactly at class start time — starts the class immediately."""
        print("⏰ PRECISE TIMER: Class start time reached!")
        if self.system.check_class_start():
            self.refresh_display()
        else:
            # Already active or start was handled by lifecycle check — just refresh
            self.refresh_display()

    def update_clock(self):
        now = QDateTime.currentDateTime()
        self.time_label.setText(now.toString('hh:mm:ss'))
        self.date_label.setText(now.toString('dddd, MMMM d, yyyy'))
        # Update countdown every second instead of every 20 seconds
        if self.system.current_schedule and hasattr(self, 'countdown_label'):
            remaining = self.system.get_class_time_remaining()
            self.countdown_label.setText(remaining)

    def check_class_lifecycle(self):
        """Runs every 5 seconds — handles class end and DB-driven status changes."""
        started = self.system.check_class_start()
        if started:
            self.refresh_display()
            self.schedule_precise_start_timer()

        schedule_snapshot = None
        if self.system.current_schedule:
            schedule_snapshot = dict(self.system.current_schedule)

        if self.system.check_class_end():
            # Reset sensor toggle when class auto-ends
            if self.hardware and self._sensor_auto_enabled:
                self.hardware.stop_monitoring()
                self.hardware.disable_sensor()
                self._hardware_active = False
                self._sensor_auto_enabled = False
                self._update_sensor_toggle_btn()
            self.refresh_display()
            self.schedule_precise_start_timer()
            if schedule_snapshot:
                self._print_class_attendance(
                    schedule_snapshot.get('subject', ''),
                    schedule_snapshot.get('section', ''),
                    teacher=schedule_snapshot.get('teacher', ''),
                    date_str=schedule_snapshot.get('date', None))

        if not self.system.current_schedule:
            if self.classes_stack.currentWidget() == self.classes_overview_page:
                self.update_main_menu()

    def update_class_timer_interval(self):
        # Kept for backward compatibility — interval is now fixed at 5s
        pass

    def closeEvent(self, event):
        """Clean up when closing application"""
        # Clean up hardware
        if self.hardware:
            self.hardware.cleanup()
        
        self.system.stop_background_sync()
        event.accept()


# ========== MAIN ENTRY POINT ==========

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon (optional)
    app.setWindowIcon(QIcon())
    
    window = ModernAttendanceUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
