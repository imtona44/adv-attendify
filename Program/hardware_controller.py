"""
Hardware Controller for Ultrasonic Sensor and IR LED Only
Uses Qt signals for thread-safe communication
"""

import time
import threading
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal

try:
    from gpiozero import DistanceSensor, LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ GPIO libraries not available - running in simulation mode")


class HardwareController(QObject):
    """Controls ultrasonic sensor and IR LED - Camera handled separately"""
    
    # Qt signals for thread-safe communication
    person_detected_signal = pyqtSignal(float)  # distance
    person_left_signal = pyqtSignal()
    scan_complete_signal = pyqtSignal()
    
    def __init__(self, 
                 trig_pin=23, 
                 echo_pin=24, 
                 ir_led_pin=18,
                 detection_distance=1.0,
                 lost_target_delay=2.0,
                 cooldown_seconds=2.0):
        super().__init__()
        
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.ir_led_pin = ir_led_pin
        self.detection_distance = detection_distance
        self.lost_target_delay = lost_target_delay
        self.cooldown_seconds = cooldown_seconds
        
        self.sensor = None
        self.ir_led = None
        self.last_seen_time = 0
        self.last_scan_time = 0
        self.is_active = False
        self.monitoring = False
        self.monitor_thread = None
        self.sensor_enabled = False
        
        self._init_hardware()
    
    def _init_hardware(self):
        """Initialize GPIO hardware only"""
        if not GPIO_AVAILABLE:
            print("⚠️ Running in simulation mode - hardware not initialized")
            return
        
        try:
            self.sensor = DistanceSensor(
                echo=self.echo_pin,
                trigger=self.trig_pin,
                max_distance=2.5
            )
            self.ir_led = LED(self.ir_led_pin)
            print("✅ Hardware initialized - Ultrasonic sensor and IR LED ready")
            print(f"Settings: Detection={self.detection_distance}m, Cooldown={self.cooldown_seconds}s")
        except Exception as e:
            print(f"❌ Hardware initialization failed: {e}")
            self.sensor = None
            self.ir_led = None
    
    def enable_sensor(self):
        """Enable ultrasonic sensor monitoring"""
        if not GPIO_AVAILABLE:
            self.sensor_enabled = True
            return
        
        if self.sensor_enabled:
            return
        
        self.sensor_enabled = True
        self.last_scan_time = 0
        print("Ultrasonic sensor ENABLED")
    
    def disable_sensor(self):
        """Disable ultrasonic sensor"""
        if not GPIO_AVAILABLE:
            self.sensor_enabled = False
            return
        
        if not self.sensor_enabled:
            return
        
        self.sensor_enabled = False
        print("🔇 Ultrasonic sensor DISABLED")
        
        # Turn off IR LED
        if self.ir_led and self.ir_led.is_lit:
            self.turn_ir_led_off()
    
    def turn_ir_led_on(self):
        """Turn on IR LED"""
        if not GPIO_AVAILABLE:
            return
        
        if self.ir_led:
            self.ir_led.on()
            print("IR LED ON")
    
    def turn_ir_led_off(self):
        """Turn off IR LED"""
        if not GPIO_AVAILABLE:
            return
        
        if self.ir_led:
            self.ir_led.off()
            print("IR LED OFF")
    
    def get_distance(self):
        """Get distance from ultrasonic sensor"""
        if not GPIO_AVAILABLE or not self.sensor:
            import random
            return random.uniform(0.3, 1.2)
        
        try:
            return self.sensor.distance * self.sensor.max_distance
        except Exception as e:
            print(f"⚠️ Distance reading error: {e}")
            return 999.0
    
    def is_in_cooldown(self):
        """Check if system is in cooldown period"""
        if self.last_scan_time == 0:
            return False
        elapsed = time.time() - self.last_scan_time
        return elapsed < self.cooldown_seconds
    
    def mark_scan_complete(self):
        """Call this when a face scan is completed to start cooldown."""
        self.last_scan_time = time.time()
        self.is_active = False
        print(f"Scan completed - cooldown active for {self.cooldown_seconds}s")
        self.scan_complete_signal.emit()
    
    def start_monitoring(self):
        """Start background monitoring for person detection"""
        if self.monitoring:
            return

        if not self.sensor_enabled:
            print("⚠️ Cannot start monitoring - sensor is disabled")
            return

        self.monitoring = True
        self.is_active = False
        self.last_seen_time = 0

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Started monitoring for person detection")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("Stopped monitoring")
    
    def _monitor_loop(self):
        """Background thread that only triggers a scan request when an object enters range."""
        while self.monitoring and self.sensor_enabled:
            try:
                if self.is_in_cooldown():
                    time.sleep(0.2)
                    continue

                distance = self.get_distance()

                if distance <= self.detection_distance:
                    if not self.is_active:
                        self.is_active = True
                        self.last_seen_time = time.time()
                        print(f"👤 Object detected at {distance:.2f}m")
                        self.person_detected_signal.emit(distance)
                else:
                    if self.is_active:
                        self.is_active = False
                        self.person_left_signal.emit()

                time.sleep(0.1)

            except Exception as e:
                print(f"⚠️ Monitor loop error: {e}")
                time.sleep(0.5)
    
    def manual_activate(self):
        """Manually turn on IR LED (for document verification)"""
        print("Manual activation - turning on IR LED")
        self.turn_ir_led_on()
        self.is_active = True
        self.last_seen_time = time.time()
    
    def manual_deactivate(self):
        """Manually turn off IR LED"""
        print("Manual deactivation - turning off IR LED")
        self.turn_ir_led_off()
        self.is_active = False
    
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up hardware...")
        self.disable_sensor()
        self.stop_monitoring()
        self.turn_ir_led_off()
        
        if GPIO_AVAILABLE and self.sensor:
            try:
                self.sensor.close()
            except:
                pass
        print("✅ Hardware cleanup complete")