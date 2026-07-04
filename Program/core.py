# ===== MUST BE FIRST: prevent DLL conflicts between cv2, ONNX Runtime, and OpenMP =====
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = 'CPUExecutionProvider'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# ==================================================================================

import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv
import json
from supabase import create_client
import shutil
import requests
import threading
import pickle

 
try:
    from uniface import RetinaFace, ArcFace
    from uniface.spoofing import create_spoofer, MiniFASNetWeights
    UNIFACE_AVAILABLE = True
    print("✅ UniFace anti-spoofing available")
except ImportError as e:
    UNIFACE_AVAILABLE = False
    print(f"⚠️ UniFace not available: {e}")
except Exception as e:
    UNIFACE_AVAILABLE = False
    print(f"⚠️ UniFace import error: {e}")
# ========== CONFIGURATION ==========
class Config:
    """Configuration settings for the application"""
    THEME = {
        'bg': '#0f172a',
        'surface': '#1e293b',
        'surface_light': '#334155',
        'primary': '#8b5cf6',
        'primary_dark': '#7c3aed',
        'secondary': '#f59e0b',
        'accent': '#06b6d4',
        'success': '#10b981',
        'error': '#ef4444',
        'text': '#f8fafc',
        'text_secondary': '#cbd5e1',
        'border': '#475569',
        'gradient': ['#8b5cf6', '#ec4899', '#f59e0b']
    }
    
    SUPABASE_URL = "your_supabase_url"
    SUPABASE_KEY = "your_key"
    
    LOCAL_STORAGE = "local_storage"
    CACHE_VERSION = "2.0"
    CACHE_STRUCTURE = "sections"


class AttendanceSystem:
    """Core attendance system with all business logic for new schema"""
        
    def __init__(self):
        # State variables
        self.known_faces = []
        self.known_names = []
        self.known_student_ids = []
        self.attendance_log = {}
        self.local_schedules = []
        self.current_schedule = None
        self.current_section = None
        self.background_sync_running = False
        self.background_cache_builder_running = False
        self.schedule_last_sync = None
        
        # Cache variables
        self.cache_index = {}
        
        # Initialize UniFace models
        self._detector = None
        self._recognizer = None
        self._spoof_detector = None
        
        # Initialize system
        self.create_local_storage()
        self.get_cache_index()
        self.cleanup_orphaned_caches()
        
        # Initialize Supabase client
        self.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        
        # Initialize UniFace models
        self._init_uniface_models()
        
        print("✅ Attendance System initialized with new schema")

    def _init_uniface_models(self):
        """Initialize UniFace models for face detection, recognition, and anti-spoofing"""
        if not UNIFACE_AVAILABLE:
            print("⚠️ UniFace not available - system will not work properly")
            return
        
        try:
            print("🔧 Initializing UniFace models...")

            
            try:
                import onnxruntime as ort
                ort.set_default_logger_severity(3)
                providers = ['CPUExecutionProvider']
            except Exception:
                providers = None

            if providers is not None:
                try:
                    self._detector = RetinaFace(providers=providers)
                    self._recognizer = ArcFace(providers=providers)
                    self._spoof_detector = create_spoofer(
                        model_name=MiniFASNetWeights.V2,
                        providers=providers
                    )
                    print("✅ UniFace models ready (CPU-pinned: Detection + Recognition + Anti-spoofing)")
                    return
                except TypeError:
                    # Older UniFace versions don't accept providers kwarg — fall through
                    print("🔧 UniFace doesn't accept providers kwarg, retrying without...")

            # Fallback: no explicit providers
            self._detector = RetinaFace()
            self._recognizer = ArcFace()
            self._spoof_detector = create_spoofer(model_name=MiniFASNetWeights.V2)
            print("✅ UniFace models ready (Detection + Recognition + Anti-spoofing)")

        except Exception as e:
            print(f"❌ UniFace initialization error: {e}")
            import traceback
            traceback.print_exc()
            self._detector = None
            self._recognizer = None
            self._spoof_detector = None

    def mirror_student_photos(self, student_id, student_name, student_section):
        """
        Mirror local photos to match database exactly
        - Downloads missing photos
        - Deletes local photos not in database
        - Creates metadata.json if missing
        - Rebuilds cache when needed
        """
        import requests
        import os
        import json
        
        print(f"\n🔄 Syncing photos for {student_name} (ID: {student_id})...")
        
        # ===== GET STUDENT FOLDER PATHS =====
        student_folder = self.get_student_folder_path(student_id, student_name, student_section)
        images_path = self.get_student_images_path(student_folder)
        os.makedirs(images_path, exist_ok=True)
        
        # ===== GET DATABASE STATE =====
        try:
            db_response = self.supabase.table("face-images")\
                .select("*")\
                .eq("student-ID", student_id)\
                .execute()
            
            # Map of filename -> image_url
            db_files = {}
            for img in db_response.data:
                filename = img.get('file-name')
                image_url = img.get('image-url')
                if filename and image_url:
                    db_files[filename] = image_url
            
            print(f"📊 Database has: {len(db_files)} photos")
            if db_files:
                print(f"   Files: {list(db_files.keys())}")
            
        except Exception as e:
            print(f"❌ Error fetching from database: {e}")
            return False
        
        # ===== GET LOCAL STATE =====
        local_files = {}
        if os.path.exists(images_path):
            for f in os.listdir(images_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    local_files[f] = os.path.join(images_path, f)
        
        print(f"📁 Local has:    {len(local_files)} photos")
        if local_files:
            print(f"   Files: {list(local_files.keys())}")
        
        # ===== CHECK IF SYNC NEEDED =====
        files_to_delete = []
        files_to_download = []
        
        # Find local files not in database (to delete)
        for local_file in local_files:
            if local_file not in db_files:
                files_to_delete.append(local_file)
        
        # Find database files not in local (to download)
        for db_file in db_files:
            if db_file not in local_files:
                files_to_download.append(db_file)
        
        # ===== EARLY EXIT IF ALREADY SYNCED =====
        if not files_to_delete and not files_to_download:
            print(f"✅ Photos already in sync")
            
            # Check and create metadata.json if missing
            metadata_path = os.path.join(student_folder, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"📝 Creating missing metadata.json...")
                # Get student data from database for metadata
                try:
                    student_data_response = self.supabase.table("student").select("*").eq("id", student_id).execute()
                    if student_data_response.data:
                        student_data = student_data_response.data[0]
                        metadata = {
                            'student_id': student_id,
                            'name': student_name,
                            'first_name': student_data.get('fname', ''),
                            'last_name': student_data.get('lname', ''),
                            'section': student_section,
                            'email': student_data.get('email', ''),
                            'image_count': len(db_files),
                            'last_sync': datetime.now().isoformat(),
                            'folder_structure': 'organized_v2'
                        }
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        print(f"✅ Metadata created at: {metadata_path}")
                    else:
                        print(f"⚠️ Could not fetch student data for metadata")
                except Exception as e:
                    print(f"⚠️ Failed to create metadata: {e}")
            
            # Check cache validity and rebuild if needed
            if not self.validate_student_cache(student_folder):
                print(f"⚠️ Cache invalid, rebuilding...")
                if self.build_student_cache(student_folder):
                    print(f"✅ Cache rebuilt successfully")
                    self.update_section_cache_with_student(student_section, student_folder)
                else:
                    print(f"❌ Failed to rebuild cache")
            else:
                print(f"✅ Cache is valid")
            
            return True
        
        # ===== DELETE PHASE =====
        if files_to_delete:
            print(f"\n Deleting {len(files_to_delete)} local photos not in database:")
            for filename in files_to_delete:
                file_path = local_files[filename]
                try:
                    os.remove(file_path)
                    print(f"   ✅ Deleted: {filename}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {filename}: {e}")
        
        # ===== DOWNLOAD PHASE =====
        if files_to_download:
            print(f"\n Downloading {len(files_to_download)} new photos from database:")
            for filename in files_to_download:
                url = db_files[filename]
                save_path = os.path.join(images_path, filename)
                
                print(f"    Downloading: {filename}")
                try:
                    # Add headers for authentication if needed
                    headers = {
                        "apikey": Config.SUPABASE_KEY,
                        "Authorization": f"Bearer {Config.SUPABASE_KEY}"
                    }
                    
                    response = requests.get(url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        print(f"      ✅ Saved ({len(response.content)} bytes)")
                    else:
                        print(f"      ❌ Failed (HTTP {response.status_code})")
                except requests.exceptions.ConnectionError:
                    print(f"      ❌ Connection error")
                except requests.exceptions.Timeout:
                    print(f"      ❌ Timeout")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
        
        # ===== CREATE METADATA.JSON =====
        print(f"\n Creating metadata.json...")
        metadata_path = os.path.join(student_folder, "metadata.json")
        
        try:
            # Get student data from database
            student_data_response = self.supabase.table("student").select("*").eq("id", student_id).execute()
            if student_data_response.data:
                student_data = student_data_response.data[0]
            else:
                student_data = {}
            
            metadata = {
                'student_id': student_id,
                'name': student_name,
                'first_name': student_data.get('fname', ''),
                'last_name': student_data.get('lname', ''),
                'section': student_section,
                'email': student_data.get('email', ''),
                'image_count': len(db_files),
                'last_sync': datetime.now().isoformat(),
                'folder_structure': 'organized_v2'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Failed to create metadata: {e}")
        
        # ===== REBUILD CACHE =====
        print(f"\n🔄 Photos changed - rebuilding cache...")
        
        if self.build_student_cache(student_folder):
            print(f"✅ Cache rebuilt successfully")
            self.update_section_cache_with_student(student_section, student_folder)
            return True
        else:
            print(f"❌ Failed to rebuild cache")
            return False


    def sync_section_photos(self, section):
        """Quick sync for all students in a section when class is detected"""
        print(f"\n🔄 Pre-class photo sync for section {section}")
        
        # Get all students in this section
        students = self.get_students_by_section(section)
        
        if not students:
            print(f"📭 No students found in section {section}")
            return
        
        changes_detected = False
        
        for student in students:
            student_id = student['id']
            student_name = f"{student.get('fname', '')} {student.get('lname', '')}".strip()
            
            # Quick check if photos need updating
            if self.photos_need_sync(student_id, student_name, section):
                print(f"   ⚠️ {student_name}: Photos changed - updating...")
                self.mirror_student_photos(student_id, student_name, section)
                changes_detected = True
        
        if changes_detected:
            print(f" Invalidating section cache for {section}")
            self.invalidate_section_cache(section)
        
        print(f"✅ Section {section} photo sync complete")


    def photos_need_sync(self, student_id, student_name, section):
        """Quick check if student photos need updating"""
        try:
            # Get database count
            db_response = self.supabase.table("face-images")\
                .select("*", count="exact")\
                .eq("student-ID", student_id)\
                .execute()
            
            db_count = len(db_response.data)
            
            # Get local count
            student_folder = self.get_student_folder_path(student_id, student_name, section)
            images_path = self.get_student_images_path(student_folder)
            
            local_count = 0
            if os.path.exists(images_path):
                local_count = len([f for f in os.listdir(images_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Also check if cache is valid
            cache_valid = self.validate_student_cache(student_folder) if os.path.exists(student_folder) else False
            
            return (db_count != local_count) or not cache_valid
            
        except Exception as e:
            print(f"⚠️ Error checking sync status: {e}")
            return False

    # ========== FILE UTILITIES ==========
    
    def get_attendance_filename(self, subject, section, date=None):
        """Generate attendance filename including hour so same-day classes get separate files"""
        if date is None:
            date = datetime.now()
        today = date.strftime("%Y-%m-%d")
        hour = date.strftime("%Hh")         
        clean_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_section = "".join(c for c in section if c.isalnum() or c in ('-', '_')).rstrip()
        return f"attendance_{clean_subject}_{clean_section}_{today}_{hour}.csv"
    
    def initialize_attendance_file(self, subject, section):
        """Initialize attendance file for a specific subject and section"""
        filename = self.get_attendance_filename(subject, section)
        if not os.path.exists(filename):
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Subject", "Teacher", "Section",
                    "Student Name", "Student ID",
                    "Time In", "Date", "Status", "Schedule ID"
                ])
            print(f"📁 Created new attendance file: {filename}")
        return filename
    
    # ========== CACHE MANAGEMENT ==========
    
    def create_local_storage(self):
        """Create local storage with cache structure"""
        if not os.path.exists(Config.LOCAL_STORAGE):
            os.makedirs(Config.LOCAL_STORAGE)
            print(f"📁 Created local storage directory: {Config.LOCAL_STORAGE}")
        
        # Create cache structure
        self.create_cache_structure()
    
    def create_cache_structure(self):
        """Create the organized cache folder structure"""
        base_dirs = ["sections", "students", "cache_metadata", "face_images"]
        
        for base_dir in base_dirs:
            dir_path = os.path.join(Config.LOCAL_STORAGE, base_dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"📁 Created directory: {dir_path}")
        
        # Initialize cache index if it doesn't exist
        cache_index_file = os.path.join(Config.LOCAL_STORAGE, "cache_metadata", "cache_index.json")
        if not os.path.exists(cache_index_file):
            initial_index = {
                "version": Config.CACHE_VERSION,
                "sections": {},
                "students": {},
                "last_updated": datetime.now().isoformat(),
                "cache_hits": 0,
                "cache_misses": 0
            }
            with open(cache_index_file, 'w') as f:
                json.dump(initial_index, f, indent=2)
            print("📁 Initialized cache index")
        
        print("✅ Cache structure ready")
    
    def get_cache_index(self):
        """Load the cache index into memory"""
        cache_index_file = os.path.join(Config.LOCAL_STORAGE, "cache_metadata", "cache_index.json")
        try:
            with open(cache_index_file, 'r') as f:
                self.cache_index = json.load(f)
            return self.cache_index
        except Exception as e:
            print(f"❌ Error loading cache index: {e}")
            self.cache_index = {"sections": {}, "students": {}, "cache_hits": 0, "cache_misses": 0}
            return self.cache_index
    
    def save_cache_index(self):
        """Save the cache index to disk"""
        cache_index_file = os.path.join(Config.LOCAL_STORAGE, "cache_metadata", "cache_index.json")
        try:
            self.cache_index["last_updated"] = datetime.now().isoformat()
            with open(cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving cache index: {e}")
            return False
    
    def get_student_folder_path(self, student_id, student_name, section):
        """Get the organized path for a student's data"""
        clean_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        folder_name = f"{student_id}_{clean_name}"
        return os.path.join(Config.LOCAL_STORAGE, "sections", section, "students", folder_name)
    
    def get_student_images_path(self, student_folder):
        """Get the images subfolder path for a student"""
        return os.path.join(student_folder, "images")
    
    def get_student_cache_path(self, student_folder):
        """Get the cache file path for a student"""
        return os.path.join(student_folder, "encoding.npy")
    
    def get_section_cache_path(self, section):
        """Get the pre-compiled section cache path"""
        return os.path.join(Config.LOCAL_STORAGE, "sections", section, "section_cache.npy")
    
    def section_exists(self, section):
        """Check if a section folder exists"""
        return os.path.exists(os.path.join(Config.LOCAL_STORAGE, "sections", section))
    
    def compute_image_hash(self, image_path):
        """Compute hash of an image file to detect changes"""
        try:
            with open(image_path, 'rb') as f:
                return hash(f.read())
        except:
            return None
    
    def compute_image_hashes(self, student_folder):
        """Compute hashes for all images in student folder"""
        images_path = self.get_student_images_path(student_folder)
        hashes = []
        
        if os.path.exists(images_path):
            # Sort files to ensure consistent order
            for filename in sorted(os.listdir(images_path)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(images_path, filename)
                    try:
                        # Use a consistent hash method
                        import hashlib
                        with open(image_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        hashes.append(file_hash)
                    except Exception as e:
                        print(f"   Error hashing {filename}: {e}")
        
        return sorted(hashes)  # Sort to ensure consistent order
    
    def validate_student_cache(self, student_folder):
        """Check if a student's cache is still valid - DEBUG VERSION"""
        cache_file = self.get_student_cache_path(student_folder)
        
        if not os.path.exists(cache_file):
            print(f"   ❌ Cache file missing: {cache_file}")
            return False
        
        try:
            cache_data = np.load(cache_file, allow_pickle=True).item()
            
            # Check version
            if cache_data.get('version') != Config.CACHE_VERSION:
                print(f"   ❌ Version mismatch: cache={cache_data.get('version')}, expected={Config.CACHE_VERSION}")
                return False
            
            # Check encodings
            if not cache_data.get('encodings'):
                print(f"   ❌ No encodings in cache")
                return False
            
            # Check image hashes
            current_hashes = self.compute_image_hashes(student_folder)
            cached_hashes = cache_data.get('image_hashes', [])
            
            if current_hashes != cached_hashes:
                print(f"   ❌ Image hashes mismatch")
                print(f"      Current: {current_hashes}")
                print(f"      Cached:  {cached_hashes}")
                return False
            
            print(f"   ✅ Cache valid")
            return True
            
        except Exception as e:
            print(f"❌ Cache validation error: {e}")
            return False
    
    def build_student_cache(self, student_folder):
        """Build or rebuild cache for a single student using UniFace"""
        try:
            student_id = os.path.basename(student_folder).split('_')[0]
            images_path = self.get_student_images_path(student_folder)
            
            print(f"🔄 Building cache for student: {student_id}")
            
            if self._detector is None or self._recognizer is None:
                print(f"❌ UniFace models not initialized")
                return False
            
            encodings = []
            valid_images = 0
            
            # Process images in sorted order for consistency
            image_files = sorted([f for f in os.listdir(images_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            for filename in image_files:
                image_path = os.path.join(images_path, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  ⚠️ Could not read {filename}")
                        continue
                    
                    # Detect face using UniFace RetinaFace
                    faces = self._detector.detect(image)
                    
                    if faces:
                        # Get embedding for first face
                        face = faces[0]
                        embedding = self._recognizer.get_embedding(image, face.landmarks)
                        embedding = embedding.flatten()
                        encodings.append(embedding)
                        valid_images += 1
                        print(f"  ✅ Processed {filename}")
                    else:
                        print(f"  ⚠️ No face found in {filename}")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
                    continue
            
            if not encodings:
                print(f"  ⚠️ No valid face encodings found for student {student_id}")
                return False
            
            # Load metadata
            metadata_file = os.path.join(student_folder, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Store image hashes for validation
            image_hashes = self.compute_image_hashes(student_folder)
            
            cache_data = {
                'student_id': student_id,
                'name': metadata.get('name', 'Unknown'),
                'full_name': metadata.get('full_name', 'Unknown'),
                'section': metadata.get('section', 'Unknown'),
                'encodings': encodings,
                'image_hashes': image_hashes,
                'valid_images': valid_images,
                'total_faces': len(encodings),
                'created_at': datetime.now().isoformat(),
                'version': Config.CACHE_VERSION,
                'engine': 'uniface_arcface'  # Mark as UniFace cache
            }
            
            cache_file = self.get_student_cache_path(student_folder)
            np.save(cache_file, cache_data)
            
            print(f"✅ Built cache for {metadata.get('name', 'Unknown')}: {len(encodings)} face encodings (512-dim)")
            return True
            
        except Exception as e:
            print(f"❌ Error building cache for {student_folder}: {e}")
            return False
    

    def rebuild_all_caches(self):
        """Rebuild all caches with ArcFace embeddings (run once after upgrading)"""
        import os
        
        print("\n" + "="*50)
        print("🔄 REBUILDING ALL CACHES WITH ARCFACE")
        print("="*50)
        
        sections_dir = os.path.join(Config.LOCAL_STORAGE, "sections")
        
        if not os.path.exists(sections_dir):
            print(f"❌ Sections directory not found: {sections_dir}")
            return
        
        print(f"📂 Scanning: {sections_dir}")
        
        total_rebuilt = 0
        total_failed = 0
        total_found = 0
        
        # Look for ALL student folders (they contain encoding.npy)
        for section in os.listdir(sections_dir):
            section_path = os.path.join(sections_dir, section)
            
            # Skip if not a directory
            if not os.path.isdir(section_path):
                continue
                
            students_path = os.path.join(section_path, "students")
            
            # Skip if no students folder
            if not os.path.exists(students_path):
                print(f"⚠️ No 'students' folder in section: {section}")
                continue
            
            print(f"\n📂 Processing section: {section}")
            
            # Look for student folders
            for student_folder in os.listdir(students_path):
                student_path = os.path.join(students_path, student_folder)
                
                # Skip if not a directory
                if not os.path.isdir(student_path):
                    continue
                
                cache_file = os.path.join(student_path, "encoding.npy")
                
                # Only process if cache file exists
                if os.path.exists(cache_file):
                    total_found += 1
                    print(f"\n   📁 Student: {student_folder}")
                    print(f"      Cache found at: {cache_file}")
                    
                    # Delete old cache
                    try:
                        os.remove(cache_file)
                        print(f"      🗑️ Removed old cache")
                    except Exception as e:
                        print(f"      ⚠️ Could not remove: {e}")
                    
                    # Build new cache
                    if self.build_student_cache(student_path):
                        total_rebuilt += 1
                        print(f"      ✅ Rebuilt successfully")
                    else:
                        total_failed += 1
                        print(f"      ❌ Failed to rebuild")
                else:
                    print(f"\n   📁 Student: {student_folder} - No cache file found")
        
        print("\n" + "="*50)
        print(f"✅ CACHE REBUILD COMPLETE")
        print(f"   Found: {total_found} cache files")
        print(f"   Rebuilt: {total_rebuilt} students")
        print(f"   Failed: {total_failed} students")
        
        if total_found == 0:
            print("\n⚠️ No cache files found! This means:")
            print("   1. No students have been synced yet")
            print("   2. The caches haven't been built")
            print("   3. Run the UI and sync a section first to create caches")
        
        print("="*50 + "\n")


    def load_student_cache(self, student_folder):
        """Load face encodings from student cache"""
        try:
            cache_file = self.get_student_cache_path(student_folder)
            if not os.path.exists(cache_file):
                return None
                
            cache_data = np.load(cache_file, allow_pickle=True).item()
            return cache_data
            
        except Exception as e:
            print(f"❌ Error loading student cache: {e}")
            return None
    
    def build_section_cache(self, section):
        """Build the fast-loading pre-compiled section cache"""
        try:
            section_path = os.path.join(Config.LOCAL_STORAGE, "sections", section)
            students_path = os.path.join(section_path, "students")
            
            if not os.path.exists(students_path):
                print(f"❌ No students folder found for section {section}")
                return 0
            
            all_encodings = []
            all_names = []
            all_student_ids = []
            processed_students = 0
            
            print(f"🔄 Building section cache for {section}...")
            
            for student_folder in os.listdir(students_path):
                student_path = os.path.join(students_path, student_folder)
                
                if not os.path.isdir(student_path):
                    continue
                    
                if not self.validate_student_cache(student_path):
                    if not self.build_student_cache(student_path):
                        continue
                
                cache_data = self.load_student_cache(student_path)
                if cache_data and cache_data.get('encodings'):
                    all_encodings.extend(cache_data['encodings'])
                    all_names.extend([cache_data['name']] * len(cache_data['encodings']))
                    all_student_ids.extend([cache_data['student_id']] * len(cache_data['encodings']))
                    processed_students += 1
                    print(f"  ✅ Added {cache_data['name']}: {len(cache_data['encodings'])} encodings")
            
            if not all_encodings:
                print(f"❌ No valid encodings found for section {section}")
                return 0
            
            unique_student_ids = list(set(all_student_ids))
            
            section_cache_data = {
                'encodings': all_encodings,
                'names': all_names,
                'student_ids': all_student_ids,
                'metadata': {
                    'section': section,
                    'student_count': processed_students,
                    'student_ids': unique_student_ids,
                    'total_encodings': len(all_encodings),
                    'last_updated': datetime.now().isoformat(),
                    'cache_version': Config.CACHE_VERSION
                }
            }
            
            section_cache_file = self.get_section_cache_path(section)
            np.save(section_cache_file, section_cache_data)
            
            self.cache_index["sections"][section] = {
                "cache_valid": True,
                "last_updated": datetime.now().isoformat(),
                "student_count": processed_students,
                "total_encodings": len(all_encodings)
            }
            self.save_cache_index()
            
            self.cache_index["cache_misses"] = self.cache_index.get("cache_misses", 0) + 1
            self.save_cache_index()
            
            print(f"✅ Built section cache for {section}: {processed_students} students, {len(all_encodings)} total encodings")
            return len(all_encodings)
            
        except Exception as e:
            print(f"❌ Error building section cache for {section}: {e}")
            return 0
    
    def validate_section_cache(self, section):
        """Check if section cache is valid"""
        section_cache_file = self.get_section_cache_path(section)
        
        if not os.path.exists(section_cache_file):
            return False
        
        try:
            cache_data = np.load(section_cache_file, allow_pickle=True).item()
            
            if cache_data.get('metadata', {}).get('cache_version') != Config.CACHE_VERSION:
                return False
                
            if not cache_data.get('encodings'):
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Section cache validation error: {e}")
            return False
    

    def load_section_cache(self, section):
        """Load pre-compiled section cache"""
        try:
            section_cache_file = self.get_section_cache_path(section)
            
            if not os.path.exists(section_cache_file):
                return None
                
            cache_data = np.load(section_cache_file, allow_pickle=True).item()
            
            self.cache_index["cache_hits"] = self.cache_index.get("cache_hits", 0) + 1
            self.save_cache_index()
            
            return cache_data
            
        except Exception as e:
            print(f"❌ Error loading section cache: {e}")
            return None

    def load_section_students(self, section, allow_build=False):
        """Load face encodings for a section safely.

        Priority:
        1) load individual student caches
        2) optionally rebuild section cache for future speed
        """
        self.known_faces.clear()
        self.known_names.clear()
        self.known_student_ids.clear()

        if not self.section_exists(section):
            print(f"❌ No data found for section {section}")
            return 0

        section_students_path = os.path.join(Config.LOCAL_STORAGE, "sections", section, "students")
        loaded_students = 0

        if os.path.exists(section_students_path):
            for folder in sorted(os.listdir(section_students_path)):
                student_folder = os.path.join(section_students_path, folder)
                if not os.path.isdir(student_folder):
                    continue

                cache_file = self.get_student_cache_path(student_folder)
                if not os.path.exists(cache_file):
                    print(f"⚠️ No cache file for {folder}")
                    continue

                try:
                    cache_data = np.load(cache_file, allow_pickle=True).item()
                    encodings = cache_data.get("encodings", [])
                    if not encodings:
                        continue

                    valid_encodings = []
                    for emb in encodings:
                        arr = np.array(emb).flatten()
                        if arr.shape[0] == 512:
                            valid_encodings.append(arr)
                        else:
                            print(f"⚠️ Skipping invalid encoding in {folder}: shape={arr.shape}")

                    if not valid_encodings:
                        continue

                    student_name = cache_data.get("name", folder)
                    student_id = str(cache_data.get("student_id", folder.split("_")[0]))

                    self.known_faces.extend(valid_encodings)
                    self.known_names.extend([student_name] * len(valid_encodings))
                    self.known_student_ids.extend([student_id] * len(valid_encodings))
                    loaded_students += 1
                    print(f"  ✅ Loaded {student_name}: {len(valid_encodings)} encodings")

                except Exception as e:
                    print(f"⚠️ Failed loading cache for {folder}: {e}")

        if self.known_faces:
            if self.current_schedule and str(self.current_schedule.get("section", "")).strip() == str(section).strip():
                 self.append_prefetched_cross_section_students_for_active_schedule()
            print(f"✅ Loaded {len(self.known_faces)} encodings from {loaded_students} student caches for {section}")

            if allow_build:
                try:
                    self.invalidate_section_cache(section)
                except Exception:
                    pass
                try:
                    self.build_section_cache(section)
                except Exception as e:
                    print(f"⚠️ Section cache rebuild skipped: {e}")

            return len(self.known_faces)

        if allow_build:
            print(f"🔄 No usable student caches found for {section}, attempting sync/build...")
            try:
                self.sync_section_students(section)
            except Exception as e:
                print(f"⚠️ sync_section_students failed: {e}")

            if os.path.exists(section_students_path):
                for folder in sorted(os.listdir(section_students_path)):
                    student_folder = os.path.join(section_students_path, folder)
                    if not os.path.isdir(student_folder):
                        continue

                    cache_file = self.get_student_cache_path(student_folder)
                    if not os.path.exists(cache_file):
                        continue

                    try:
                        cache_data = np.load(cache_file, allow_pickle=True).item()
                        encodings = cache_data.get("encodings", [])
                        if not encodings:
                            continue

                        valid_encodings = []
                        for emb in encodings:
                            arr = np.array(emb).flatten()
                            if arr.shape[0] == 512:
                                valid_encodings.append(arr)

                        if not valid_encodings:
                            continue

                        student_name = cache_data.get("name", folder)
                        student_id = str(cache_data.get("student_id", folder.split("_")[0]))

                        self.known_faces.extend(valid_encodings)
                        self.known_names.extend([student_name] * len(valid_encodings))
                        self.known_student_ids.extend([student_id] * len(valid_encodings))
                    except Exception:
                        pass

            if self.known_faces:
                if self.current_schedule and str(self.current_schedule.get("section", "")).strip() == str(section).strip():
                 self.append_prefetched_cross_section_students_for_active_schedule()
                try:
                    self.invalidate_section_cache(section)
                except Exception:
                    pass
                try:
                    self.build_section_cache(section)
                except Exception:
                    pass

                print(f"✅ Loaded {len(self.known_faces)} encodings after sync/build for {section}")
                return len(self.known_faces)

        print(f"❌ No valid face data loaded for section {section}")
        return 0
    
    def _normalize_name(self, name):
        """Normalize a full name for reliable comparison."""
        return " ".join(str(name or "").split()).strip().lower()


    def resolve_subject_id_for_schedule(self, schedule):
        """
        Resolve the real enrollment key from the current schedule by matching:
        subjects.sub == schedule['subject']
        subjects.section == schedule['section']
        """
        try:
            subject_name = str(schedule.get("subject", "")).strip()
            section = str(schedule.get("section", "")).strip()

            response = (
                self.supabase
                .table("subjects")
                .select("id, sub, section")
                .eq("sub", subject_name)
                .eq("section", section)
                .execute()
            )

            if response.data:
                subject_id = response.data[0]["id"]
                print(f"✅ Resolved subject_id={subject_id} for {subject_name} [{section}]")
                return subject_id

            print(f"⚠️ Could not resolve subject_id for subject='{subject_name}', section='{section}'")
            return None

        except Exception as e:
            print(f"❌ Error resolving subject_id for schedule: {e}")
            return None



    def prefetch_cross_section_students_for_schedule(self, schedule):
        """
        Ensure cross-section enrolled students for this schedule are already local and encoded
        before class starts.
        """
        try:
            if not schedule:
                return 0

            base_section = str(schedule.get("section", "")).strip()
            enrolled_students = self.get_enrolled_students_for_schedule(schedule)
            if not enrolled_students:
                return 0

            prepared = 0

            for student in enrolled_students:
                student_id = str(student.get("id", "")).strip()
                home_section = str(student.get("year-section", "")).strip()
                student_name = f"{student.get('fname', '')} {student.get('lname', '')}".strip()

                if not student_id or not home_section or not student_name:
                    continue

                # Only prepare cross-section students here
                if home_section == base_section:
                    continue

                student_folder = self.find_student_folder(student_id, home_section)

                if not student_folder or not os.path.exists(student_folder):
                    print(f"📥 Prefetch: local folder missing for {student_name} ({student_id}) - syncing...")
                    ok = self.mirror_student_photos(student_id, student_name, home_section)
                    if not ok:
                        print(f"❌ Prefetch failed for {student_name}")
                        continue
                    student_folder = self.find_student_folder(student_id, home_section)

                if not student_folder or not os.path.exists(student_folder):
                    continue

                if not self.validate_student_cache(student_folder):
                    print(f"🔄 Prefetch: rebuilding cache for {student_name} ({student_id})...")
                    if not self.build_student_cache(student_folder):
                        print(f"❌ Prefetch cache build failed for {student_name}")
                        continue

                prepared += 1
                print(f"✅ Prefetch ready: {student_name} ({student_id}) from {home_section}")

            print(f"📦 Prefetch complete for schedule '{schedule.get('subject')}' [{base_section}] -> {prepared} cross-section students ready")
            return prepared

        except Exception as e:
            print(f"❌ Error prefetching cross-section students for schedule: {e}")
            return 0

    def prefetch_cross_section_students_for_schedules(self, schedules):
        """Prepare cross-section enrolled students for a list of schedules."""
        total = 0
        seen = set()

        for schedule in schedules or []:
            subject = str(schedule.get("subject", "")).strip()
            section = str(schedule.get("section", "")).strip()
            key = (subject, section)

            # Avoid repeating identical schedule prefetch in one sync pass
            if key in seen:
                continue
            seen.add(key)

            total += self.prefetch_cross_section_students_for_schedule(schedule)

        print(f"📦 Total prefetched cross-section students across schedules: {total}")
        return total


    def get_enrolled_students_for_schedule(self, schedule):
        """
        Get all students enrolled in the current schedule's real subject,
        including cross-section/retaker students.
        """
        try:
            subject_id = self.resolve_subject_id_for_schedule(schedule)
            if subject_id is None:
                return []

            enrollments = (
                self.supabase
                .table("enrollment_sub")
                .select("stud_id")
                .eq("sub_id", subject_id)
                .execute()
            )

            stud_ids = list({
                str(row.get("stud_id", "")).strip()
                for row in (enrollments.data or [])
                if row.get("stud_id")
            })

            if not stud_ids:
                print(f"⚠️ No enrolled students found for subject_id={subject_id}")
                return []

            students_resp = (
                self.supabase
                .table("student")
                .select('id, fname, lname, "year-section"')
                .in_("id", stud_ids)
                .execute()
            )

            students = students_resp.data or []
            print(f"📚 Found {len(students)} enrolled students for active schedule")
            return students

        except Exception as e:
            print(f"❌ Error getting enrolled students for schedule: {e}")
            return []


    def ensure_student_local_cache(self, student):
        """
        Ensure the student exists locally and has a valid cache.
        Uses the student's real home section.
        """
        try:
            student_id = str(student.get("id", "")).strip()
            fname = str(student.get("fname", "")).strip()
            lname = str(student.get("lname", "")).strip()
            section = str(student.get("year-section", "")).strip()
            student_name = f"{fname} {lname}".strip()

            if not student_id or not section or not student_name:
                return None

            student_folder = self.find_student_folder(student_id, section)

            # If folder missing, create/sync from DB
            if not student_folder or not os.path.exists(student_folder):
                print(f"📥 Local folder missing for {student_name} ({student_id}) - syncing...")
                ok = self.mirror_student_photos(student_id, student_name, section)
                if not ok:
                    print(f"❌ Failed syncing student {student_name}")
                    return None
                student_folder = self.find_student_folder(student_id, section)

            if not student_folder or not os.path.exists(student_folder):
                return None

            # Validate or rebuild cache if needed
            if not self.validate_student_cache(student_folder):
                print(f"🔄 Rebuilding cache for {student_name} ({student_id})...")
                if not self.build_student_cache(student_folder):
                    print(f"❌ Failed building cache for {student_name}")
                    return None

            return student_folder

        except Exception as e:
            print(f"❌ Error ensuring local cache for student: {e}")
            return None


    def append_prefetched_cross_section_students_for_active_schedule(self):
        """
        Append already-prefetched cross-section students into the active
        recognition pool. No downloading or encoding here.
        """
        try:
            if not self.current_schedule:
                return 0

            base_section = str(self.current_schedule.get("section", "")).strip()
            enrolled_students = self.get_enrolled_students_for_schedule(self.current_schedule)
            if not enrolled_students:
                return 0

            already_loaded_ids = set(str(x) for x in self.known_student_ids)
            added = 0

            for student in enrolled_students:
                student_id = str(student.get("id", "")).strip()
                home_section = str(student.get("year-section", "")).strip()

                if not student_id or not home_section:
                    continue

                # only add cross-section students here
                if home_section == base_section:
                    continue

                if student_id in already_loaded_ids:
                    continue

                student_folder = self.find_student_folder(student_id, home_section)
                if not student_folder:
                    print(f"⚠️ Active pool: no local folder for {student_id} ({home_section})")
                    continue

                cache_file = self.get_student_cache_path(student_folder)
                if not os.path.exists(cache_file):
                    print(f"⚠️ Active pool: no cache file for {student_id} ({home_section})")
                    continue

                try:
                    cache_data = np.load(cache_file, allow_pickle=True).item()
                    encodings = cache_data.get("encodings", [])
                    if not encodings:
                        continue

                    valid_encodings = []
                    for emb in encodings:
                        arr = np.array(emb).flatten()
                        if arr.shape[0] == 512:
                            valid_encodings.append(arr)
                        else:
                            print(f"⚠️ Active pool: skipping invalid encoding for {student_id}: shape={arr.shape}")

                    if not valid_encodings:
                        continue

                    student_name = cache_data.get(
                        "name",
                        f"{student.get('fname', '')} {student.get('lname', '')}".strip()
                    )

                    self.known_faces.extend(valid_encodings)
                    self.known_names.extend([student_name] * len(valid_encodings))
                    self.known_student_ids.extend([student_id] * len(valid_encodings))
                    already_loaded_ids.add(student_id)
                    added += len(valid_encodings)

                    print(f"✅ Active pool added: {student_name} ({student_id}) from {home_section}")

                except Exception as e:
                    print(f"⚠️ Active pool load failed for {student_id}: {e}")

            print(f"Active pool total extra encodings added: {added}")
            return added

        except Exception as e:
            print(f"❌ Error appending prefetched cross-section students: {e}")
            return 0


    def append_cross_section_students_for_active_schedule(self):
        """
        After normal section load, append valid enrolled students from OTHER sections
        into the active recognition pool.
        """
        if not self.current_schedule:
            return 0

        base_section = str(self.current_schedule.get("section", "")).strip()
        enrolled_students = self.get_enrolled_students_for_schedule(self.current_schedule)

        if not enrolled_students:
            return 0

        already_loaded_ids = set(str(x) for x in self.known_student_ids)
        added = 0

        for student in enrolled_students:
            student_id = str(student.get("id", "")).strip()
            home_section = str(student.get("year-section", "")).strip()

            # Only append cross-section extras here
            if not student_id or not home_section or home_section == base_section:
                continue

            if student_id in already_loaded_ids:
                continue

            student_folder = self.ensure_student_local_cache(student)
            if not student_folder:
                continue

            cache_file = self.get_student_cache_path(student_folder)
            if not os.path.exists(cache_file):
                continue

            try:
                cache_data = np.load(cache_file, allow_pickle=True).item()
                encodings = cache_data.get("encodings", [])
                if not encodings:
                    continue

                valid_encodings = []
                for emb in encodings:
                    arr = np.array(emb).flatten()
                    if arr.shape[0] == 512:
                        valid_encodings.append(arr)
                    else:
                        print(f"⚠️ Skipping invalid encoding for {student_id}: shape={arr.shape}")

                if not valid_encodings:
                    continue

                student_name = cache_data.get(
                    "name",
                    f"{student.get('fname', '')} {student.get('lname', '')}".strip()
                )

                self.known_faces.extend(valid_encodings)
                self.known_names.extend([student_name] * len(valid_encodings))
                self.known_student_ids.extend([student_id] * len(valid_encodings))
                already_loaded_ids.add(student_id)
                added += len(valid_encodings)

                print(f"✅ Added cross-section enrolled student: {student_name} ({student_id}) from {home_section}")

            except Exception as e:
                print(f"⚠️ Failed loading extra enrolled student {student_id}: {e}")

        print(f"Added {added} extra encodings from cross-section enrolled students")
        return added





    def clear_section_cache(self, section):
        """Clear cache for a specific section"""
        section_cache_file = self.get_section_cache_path(section)
        if os.path.exists(section_cache_file):
            os.remove(section_cache_file)
            print(f"Cleared cache for section: {section}")
        
        if section in self.cache_index.get("sections", {}):
            self.cache_index["sections"][section]["cache_valid"] = False
        self.save_cache_index()
    
    def invalidate_section_cache(self, section):
        """Mark a section cache as invalid"""
        if section in self.cache_index.get("sections", {}):
            self.cache_index["sections"][section]["cache_valid"] = False
        self.save_cache_index()
    
    def invalidate_student_cache(self, student_id, section):
        """Mark a student's cache as invalid"""
        if student_id in self.cache_index.get("students", {}):
            self.cache_index["students"][student_id]["cache_valid"] = False
            self.cache_index["students"][student_id]["last_invalidated"] = datetime.now().isoformat()
        
        if section in self.cache_index.get("sections", {}):
            self.cache_index["sections"][section]["cache_valid"] = False
        
        self.save_cache_index()
        print(f"🔄 Invalidated cache for student {student_id} in section {section}")
    
    def find_student_folder(self, student_id, section):
        """Find student folder by ID and section"""
        if not section:
            return None
        
        section_path = os.path.join(Config.LOCAL_STORAGE, "sections", section, "students")
        if not os.path.exists(section_path):
            return None
        
        for folder in os.listdir(section_path):
            if folder.startswith(f"{student_id}_"):
                return os.path.join(section_path, folder)
        
        return None
    
    def cleanup_orphaned_caches(self):
        """Remove cache entries for students that no longer exist"""
        print("🧹 Cleaning up orphaned caches...")
        
        students_to_remove = []
        
        for student_id, student_data in self.cache_index.get("students", {}).items():
            section = student_data.get("section")
            student_folder = self.find_student_folder(student_id, section)
            
            if not student_folder or not os.path.exists(student_folder):
                students_to_remove.append(student_id)
        
        for student_id in students_to_remove:
            if student_id in self.cache_index["students"]:
                del self.cache_index["students"][student_id]
                print(f"🗑️ Removed orphaned cache: {student_id}")
        
        sections_to_remove = []
        for section, section_data in self.cache_index.get("sections", {}).items():
            section_path = os.path.join(Config.LOCAL_STORAGE, "sections", section)
            students_path = os.path.join(section_path, "students") if os.path.exists(section_path) else None
            if not os.path.exists(section_path) or not students_path or not os.listdir(students_path):
                sections_to_remove.append(section)
        
        for section in sections_to_remove:
            if section in self.cache_index["sections"]:
                del self.cache_index["sections"][section]
                print(f"🗑️ Removed orphaned section cache: {section}")
        
        self.save_cache_index()
    
    # ========== STUDENT MANAGEMENT ==========
    
    def get_students_by_section(self, section):
        """Get students from a specific section only using new schema"""
        try:
            response = self.supabase.table("student").select("*").eq("year-section", section).execute()
            
            if response.data:
                print(f"Found {len(response.data)} students in section {section}")
                return response.data
            else:
                print(f"No students found in section {section}")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching section {section} students: {e}")
            return []
    
    def download_student_images(self, student_data):
        """Download student images from public storage bucket via face-images table"""
        try:
            student_id = student_data['id']
            student_name = f"{student_data.get('fname', '')} {student_data.get('lname', '')}".strip()
            student_section = student_data.get('year-section', 'Unknown')
            
            student_folder = self.get_student_folder_path(student_id, student_name, student_section)
            images_path = self.get_student_images_path(student_folder)
            
            if not os.path.exists(student_folder):
                os.makedirs(student_folder)
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            
            # Get image records from face-images table
            response = self.supabase.table("face-images").select("*").eq("student-ID", student_id).execute()
            
            if not response.data:
                print(f" No images found for student {student_name}")
                return False
            
            print(f" Found {len(response.data)} images for {student_name}")
            
            downloaded_count = 0
            for i, image_record in enumerate(response.data):
                image_url = image_record.get('image-url')
                filename = image_record.get('file-name', f"image_{i+1}.jpg")
                
                if not image_url:
                    continue
                
                # Get file extension from filename or URL
                ext = os.path.splitext(filename)[1]
                if not ext:
                    url_path = image_url.split('?')[0]
                    ext = os.path.splitext(url_path)[1]
                if not ext:
                    ext = '.jpg'
                
                safe_filename = f"image_{i+1}{ext}"
                image_path = os.path.join(images_path, safe_filename)
                
                # Skip if already exists
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    print(f" Image already exists: {safe_filename}")
                    downloaded_count += 1
                    continue
                
                # Download image from public URL
                try:
                    print(f" Downloading from: {image_url}")
                    img_response = requests.get(image_url, timeout=30)
                    
                    if img_response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            f.write(img_response.content)
                        downloaded_count += 1
                        print(f"  ✅ Saved as: {safe_filename} ({len(img_response.content)} bytes)")
                    else:
                        print(f"  ❌ Failed to download: HTTP {img_response.status_code}")
                        
                except Exception as e:
                    print(f"  ❌ Error downloading: {e}")
            
            # Save metadata
            metadata = {
                'student_id': student_id,
                'name': student_name,
                'first_name': student_data.get('fname', ''),
                'last_name': student_data.get('lname', ''),
                'section': student_section,
                'email': student_data.get('email', ''),
                'image_count': downloaded_count,
                'last_sync': datetime.now().isoformat(),
                'folder_structure': 'organized_v2'
            }
            
            with open(os.path.join(student_folder, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if downloaded_count > 0:
                print(f"Building cache for student: {student_name}")
                if self.build_student_cache(student_folder):
                    print(f"✅ Cache built successfully for {student_name}")
                    self.update_section_cache_with_student(student_section, student_folder)
                else:
                    print(f"❌ Failed to build cache for {student_name}")
            else:
                print(f"⚠️ No images downloaded for {student_name}")
            
            return downloaded_count > 0
            
        except Exception as e:
            print(f"❌ Download error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_section_cache_with_student(self, section, student_folder):
        """Update section cache to include a new student"""
        try:
            section_cache_file = self.get_section_cache_path(section)
            
            if not os.path.exists(section_cache_file):
                print(f"📝 Section cache doesn't exist yet, will build on next load")
                return False
            
            section_cache_data = np.load(section_cache_file, allow_pickle=True).item()
            
            student_cache_data = self.load_student_cache(student_folder)
            if not student_cache_data or not student_cache_data.get('encodings'):
                print(f"❌ No student cache data to add to section")
                return False
            
            section_cache_data['encodings'].extend(student_cache_data['encodings'])
            section_cache_data['names'].extend([student_cache_data['name']] * len(student_cache_data['encodings']))
            section_cache_data['student_ids'].extend([student_cache_data['student_id']] * len(student_cache_data['encodings']))
            
            section_cache_data['metadata']['student_count'] += 1
            section_cache_data['metadata']['total_encodings'] += len(student_cache_data['encodings'])
            section_cache_data['metadata']['last_updated'] = datetime.now().isoformat()
            
            np.save(section_cache_file, section_cache_data)
            
            self.cache_index["sections"][section] = {
                "cache_valid": True,
                "last_updated": datetime.now().isoformat(),
                "student_count": section_cache_data['metadata']['student_count'],
                "total_encodings": section_cache_data['metadata']['total_encodings']
            }
            self.save_cache_index()
            
            print(f"✅ Updated section cache for {section}: added {student_cache_data['name']} with {len(student_cache_data['encodings'])} encodings")
            return True
            
        except Exception as e:
            print(f"❌ Error updating section cache with new student: {e}")
            self.invalidate_section_cache(section)
            return False
    
    def remove_student_data(self, student_id, section):
        """Remove all data for a student"""
        try:
            student_folder = None
            section_path = os.path.join(Config.LOCAL_STORAGE, "sections", section, "students")
            
            if os.path.exists(section_path):
                for folder in os.listdir(section_path):
                    if folder.startswith(f"{student_id}_"):
                        student_folder = os.path.join(section_path, folder)
                        break
            
            if student_folder and os.path.exists(student_folder):
                shutil.rmtree(student_folder)
                print(f" Removed student data: {student_id}")
            
            if student_id in self.cache_index.get("students", {}):
                del self.cache_index["students"][student_id]
            
            self.invalidate_section_cache(section)
            self.save_cache_index()
            
        except Exception as e:
            print(f"Error removing student data: {e}")
    
    def sync_section_students(self, section):
        """Sync students from a specific section only using new schema"""
        try:
            print(f"🔄 Syncing students for section {section}...")
            
            web_students = self.get_students_by_section(section)
            
            section_path = os.path.join(Config.LOCAL_STORAGE, "sections", section, "students")
            existing_students = set()
            
            if os.path.exists(section_path):
                for folder in os.listdir(section_path):
                    if os.path.isdir(os.path.join(section_path, folder)):
                        student_id = folder.split('_')[0]
                        existing_students.add(student_id)
            
            new_students = 0
            updated_students = 0
            
            for web_student in web_students:
                student_id = web_student.get('id')
                student_name = f"{web_student.get('fname', '')} {web_student.get('lname', '')}".strip()
                
                if not student_id:
                    continue
                    
                if student_id not in existing_students:
                    print(f"New student in section {section}: {student_name}")
                    if self.download_student_images(web_student):
                        new_students += 1
                        print(f"✅ Successfully downloaded and cached: {student_name}")
                else:
                    student_folder = self.get_student_folder_path(student_id, student_name, section)
                    metadata_file = os.path.join(student_folder, "metadata.json")
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        old_name = metadata.get('name', '')
                        if old_name != student_name:
                            print(f"🔄 Updating student name: {old_name} -> {student_name}")
                            metadata['name'] = student_name
                            metadata['first_name'] = web_student.get('fname', '')
                            metadata['last_name'] = web_student.get('lname', '')
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            self.invalidate_student_cache(student_id, section)
                            updated_students += 1
            
            web_student_ids = {s['id'] for s in web_students if s.get('id')}
            for existing_id in existing_students:
                if existing_id not in web_student_ids:
                    print(f"Removing student no longer in section: {existing_id}")
                    self.remove_student_data(existing_id, section)
            
            print(f"✅ Section {section} sync complete: {new_students} new, {updated_students} updated")
            
            if new_students > 0 or updated_students > 0:
                print(f"🔄 Students changed, rebuilding section cache for {section}")
                self.clear_section_cache(section)
                self.build_section_cache(section)
            elif not self.validate_section_cache(section):
                print(f"🔄 Building missing/invalid cache for {section}")
                self.build_section_cache(section)
            
            return new_students > 0 or updated_students > 0
            
        except Exception as e:
            print(f"❌ Error syncing section {section}: {e}")
            return False
    
    # ========== SECTION LOADING ==========
    


    # ========== SCHEDULE MANAGEMENT ==========
    
    def sync_schedules(self, sync_students=False):
        """Sync all schedules for today from the schedule table - ONLY ACTIVE/SCHEDULED"""
        try:
            # Get today's date
            today = datetime.now().date()
            
            # CRITICAL: Only fetch schedules that are NOT ended
            # Also fetch professor details
            response = self.supabase.table("schedule")\
                .select("*, professor!inner(fname, lname)")\
                .eq("date", today.isoformat())\
                .neq("status", "ended")\
                .execute()
            
            if response.data:
                old_count = len(self.local_schedules)
                
                # Transform schedule data to match expected format
                self.local_schedules = []
                for sched in response.data:
                    # Get professor name from joined data
                    professor = sched.get('professor', {})
                    first_name = professor.get('fname', '')
                    last_name = professor.get('lname', '')
                    
                    # Combine first and last name
                    if first_name and last_name:
                        teacher_name = f"{first_name} {last_name}"
                    elif first_name:
                        teacher_name = first_name
                    elif last_name:
                        teacher_name = last_name
                    else:
                        teacher_name = "Unknown Professor"
                    
                    self.local_schedules.append({
                        'id': sched['id'],
                        'subject': sched['sub'],
                        'teacher': teacher_name,
                        'section': sched['year-section'],
                        'classroom': 'TBA',
                        'start_time': self.format_time_for_display(sched['time-in']),
                        'duration': self.calculate_duration(sched['time-in'], sched['time-out']),
                        'late_threshold': self.time_to_minutes(sched.get('late', '00:15:00')),
                        'status': sched.get('status', 'scheduled'),  # Store status
                        'date': sched.get('date')
                    })
                
                self.schedule_last_sync = datetime.now()
                
                if sync_students:
                    print(f"📅 Synced {len(self.local_schedules)} active schedules for today")
                    
                else:
                    print(f"📅 Light sync: {len(self.local_schedules)} active schedules")

                self.prefetch_cross_section_students_for_schedules(self.local_schedules)
                changed = len(self.local_schedules) != old_count
                
                if changed and sync_students:
                    # Find sections that need syncing
                    sections_to_sync = set()
                    for sched in self.local_schedules:
                        sections_to_sync.add(sched['section'])
                    
                    for section in sections_to_sync:
                        print(f"🔄 Syncing students for section: {section}")
                        self.sync_section_students(section)
                
                return changed
                
            else:
                self.local_schedules = []
                if sync_students:
                    print("📅 No active schedules found for today")
                return False
                
        except Exception as e:
            print(f"❌ Schedule sync error: {e}")
            return False
    
    def format_time_for_display(self, time_value):
        """Format time from database for display"""
        if time_value is None:
            return "00:00:00"
        if isinstance(time_value, str):
            return time_value
        elif hasattr(time_value, 'strftime'):
            return time_value.strftime("%H:%M:%S")
        else:
            return str(time_value)
    
    def calculate_duration(self, start_time, end_time):
        """Calculate duration in minutes between start and end times"""
        try:
            if isinstance(start_time, str):
                start_parts = start_time.split(':')
                start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
            else:
                start_minutes = start_time.hour * 60 + start_time.minute
            
            if isinstance(end_time, str):
                end_parts = end_time.split(':')
                end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
            else:
                end_minutes = end_time.hour * 60 + end_time.minute
            
            duration = end_minutes - start_minutes
            if duration < 0:
                duration += 24 * 60
            return max(duration, 1)
        except:
            return 60
    
    def time_to_minutes(self, time_value):
        """Convert TIME value to minutes"""
        try:
            if time_value is None:
                return 15
            if isinstance(time_value, str):
                parts = time_value.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            elif hasattr(time_value, 'hour'):
                return time_value.hour * 60 + time_value.minute
            else:
                return 15
        except:
            return 15
    
    def get_local_schedules(self):
        """Get schedules from local storage"""
        return self.local_schedules
    
    def get_current_schedule(self):
        """Find which schedule is currently active - PRIORITIZES 'active' status over time"""
        now = datetime.now()
        today_date = now.date()
        
        # FIRST PRIORITY: Look for any class marked as 'active' in status
        for schedule in self.local_schedules:
            if schedule.get('status') == 'active':
                print(f"🔍 Found class with 'active' status: {schedule['subject']}")
                return schedule
        
        # SECOND PRIORITY: Look for any class within time window that's not ended
        for schedule in self.local_schedules:
            try:
                # Skip if ended
                if schedule.get('status') == 'ended':
                    continue
                
                # Check if for today
                sched_date = schedule.get('date')
                if sched_date:
                    if isinstance(sched_date, str):
                        sched_date = datetime.strptime(sched_date, "%Y-%m-%d").date()
                else:
                    sched_date = today_date
                
                if sched_date != today_date:
                    continue
                
                # Check time window
                schedule_time = datetime.strptime(schedule['start_time'], "%H:%M:%S").time()
                start_datetime = datetime.combine(now.date(), schedule_time)
                end_datetime = start_datetime + timedelta(minutes=schedule['duration'])
                
                if start_datetime <= now <= end_datetime:
                    # Auto-mark as active
                    print(f"🔍 Class within time window: {schedule['subject']}")
                    return schedule
                    
            except Exception as e:
                continue
        
        return None
    
    def auto_update_completed_schedules(self):
        """Automatically update status of completed schedules"""
        try:
            now = datetime.now()
            today_date = now.date()
            
            for schedule in self.local_schedules[:]:
                try:
                    sched_date = schedule.get('date')
                    if sched_date:
                        if isinstance(sched_date, str):
                            sched_date = datetime.strptime(sched_date, "%Y-%m-%d").date()
                    else:
                        sched_date = today_date
                    
                    if sched_date != today_date:
                        continue
                    
                    schedule_time = datetime.strptime(schedule['start_time'], "%H:%M:%S").time()
                    class_start = datetime.combine(now.date(), schedule_time)
                    class_end = class_start + timedelta(minutes=schedule['duration'])
                    
                    if now > class_end + timedelta(minutes=1):
                        # Update status in database
                        try:
                            self.supabase.table("schedule").update({"status": "ended"}).eq("id", schedule['id']).execute()
                            print(f"🔄 Auto-updated: {schedule['subject']} - {schedule['section']} to ended")
                        except Exception as e:
                            print(f"❌ Error updating status: {e}")
                        self.local_schedules.remove(schedule)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ Error in auto-update: {e}")
    
    # ========== ATTENDANCE MARKING ==========
    
    def mark_attendance(self, name, schedule):
        """Mark attendance using enrollment-based authorization for the active schedule."""
        if name == "Unknown":
            return False, "Unknown student"

        try:
            if not schedule:
                return False, "No active schedule"

            target_name = self._normalize_name(name)

            # Get allowed students for this schedule from enrollment
            enrolled_students = self.get_enrolled_students_for_schedule(schedule)
            if not enrolled_students:
                return False, "No enrolled students found"

            student_data = None
            for student in enrolled_students:
                full_name = f"{student.get('fname', '')} {student.get('lname', '')}".strip()
                if self._normalize_name(full_name) == target_name:
                    student_data = student
                    break

            if not student_data:
                print(f"❌ Recognized name '{name}' is not enrolled in this active class")
                return False, "Not enrolled in class"

            student_id = student_data["id"]
            schedule_id = schedule["id"]

            composite_key = (schedule_id, student_id)

            # In-memory duplicate check
            if composite_key in self.attendance_log:
                print(f"⚠️ Student {name} already marked for schedule {schedule_id}")
                return False, "Already marked in this class"

            # Database duplicate check
            duplicate_check = (
                self.supabase
                .table("schedule_list")
                .select("*")
                .eq("sched_id", schedule_id)
                .eq("student_id", student_id)
                .execute()
            )

            if duplicate_check.data:
                print(f"⚠️ Student {name} already marked for schedule {schedule_id}")
                return False, "Already marked"

            current_time = datetime.now()

            # Calculate lateness
            schedule_time = datetime.strptime(schedule["start_time"], "%H:%M:%S").time()
            class_start = datetime.combine(current_time.date(), schedule_time)
            minutes_late = (current_time - class_start).total_seconds() / 60
            status = "late" if minutes_late > schedule["late_threshold"] else "present"

            attendance_data = {
                "sched_id": schedule_id,
                "student_id": student_id,
                "status": status,
                "time_in": current_time.strftime("%H:%M:%S")
            }

            self.supabase.table("schedule_list").insert(attendance_data).execute()

            self.attendance_log[composite_key] = current_time.strftime("%H:%M:%S")

            attendance_file = self.initialize_attendance_file(schedule["subject"], schedule["section"])
            with open(attendance_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    schedule["subject"],
                    schedule["teacher"],
                    schedule["section"],
                    f"{student_data.get('fname', '')} {student_data.get('lname', '')}".strip(),
                    student_id,
                    current_time.strftime("%H:%M:%S"),
                    current_time.strftime("%Y-%m-%d"),
                    status,
                    schedule_id
                ])

            print(f"✅ Marked attendance: {name} -> {status}")
            return True, status

        except Exception as e:
            print(f"❌ Error marking attendance for {name}: {e}")
            import traceback
            traceback.print_exc()
            return False, "Error"
    

    
    def load_class_attendance(self, schedule):
        """Load attendance for specific schedule using schedule_id"""
        if not schedule:
            return []
        
        try:
            schedule_id = schedule['id']
            
            # Fetch attendance records with student details for THIS SCHEDULE
            attendance_response = self.supabase.table("schedule_list")\
                .select("*, student!inner(fname, lname)")\
                .eq("sched_id", schedule_id)\
                .execute()
            
            attendance_records = []
            if attendance_response.data:
                for record in attendance_response.data:
                    student = record.get('student', {})
                    composite_key = (schedule_id, record['student_id'])
                    
                    # Store in memory if not already
                    if composite_key not in self.attendance_log:
                        self.attendance_log[composite_key] = record.get('time_in', '')
                    
                    attendance_records.append({
                        'name': f"{student.get('fname', '')} {student.get('lname', '')}".strip(),
                        'time': record.get('time_in', ''),
                        'status': record.get('status', 'absent')
                    })
            
            print(f"📊 Loaded {len(attendance_records)} attendance records for schedule {schedule_id}")
            return attendance_records
            
        except Exception as e:
            print(f"❌ Error loading attendance: {e}")
            return []
    
    # ========== BACKGROUND TASKS ==========
    
    def start_background_sync(self, target_section, callback=None):
        """Start background student sync for a specific section"""
        if self.background_sync_running:
            return
            
        self.background_sync_running = True
        print(f"🔄 Starting student sync for section {target_section}")
        
        def student_sync_loop():
            if not self.background_sync_running:
                return
                
            try:
                new_students = self.sync_section_students(target_section)
                
                if new_students and callback:
                    callback(target_section)
                    
            except Exception as e:
                print(f"❌ Student sync error: {e}")
            
            if self.background_sync_running:
                threading.Timer(60, student_sync_loop).start()
        
        student_sync_loop()
    
    def stop_background_sync(self):
        """Stop background synchronization"""
        self.background_sync_running = False
        print("⏹️ Background sync stopped")
    
    def start_background_cache_preparation(self):
        """Start background cache preparation for upcoming sections"""
        if self.background_cache_builder_running:
            return
            
        self.background_cache_builder_running = True
        print("🔄 Starting background cache preparation...")
        
        def cache_preparation_loop():
            if not self.background_cache_builder_running:
                return
                
            try:
                if not self.current_schedule:
                    schedules = self.get_local_schedules()
                    now = datetime.now()
                    today_date = now.date()
                    
                    for schedule in schedules:
                        sched_date = schedule.get('date')
                        if sched_date:
                            if isinstance(sched_date, str):
                                sched_date = datetime.strptime(sched_date, "%Y-%m-%d").date()
                        else:
                            sched_date = today_date
                        
                        if sched_date < today_date:
                            continue
                            
                        schedule_time = datetime.strptime(schedule['start_time'], "%H:%M:%S").time()
                        schedule_datetime = datetime.combine(sched_date, schedule_time)
                        
                        time_until = (schedule_datetime - now).total_seconds()
                        if 0 < time_until <= 1800:
                            next_section = schedule['section']
                            if not self.validate_section_cache(next_section):
                                print(f"🔄 Pre-building cache for upcoming section: {next_section}")
                                self.build_section_cache(next_section)
                            break
                            
            except Exception as e:
                print(f"⚠️ Background cache preparation error: {e}")
            
            if self.background_cache_builder_running:
                threading.Timer(300, cache_preparation_loop).start()
        
        cache_preparation_loop()
    
    def start_schedule_sync_loop(self, schedule_callback=None, student_callback=None):
        """Continuous schedule sync - FIXED to actually run continuously"""
        
        print("Starting background schedule sync loop")
        
        # Flag to control the loop
        self.sync_loop_running = True
        
        def schedule_sync_loop():
            # This runs in a separate thread
            while self.sync_loop_running:
                try:
                    if not self.current_schedule:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Full sync (no active class)...")
                        changed = self.sync_schedules(sync_students=True)
                        if changed and schedule_callback:
                            schedule_callback()
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏸️ Light sync (active class) - schedules only...")
                        changed = self.sync_schedules(sync_students=False)
                        if changed and schedule_callback:
                            schedule_callback()
                        
                except Exception as e:
                    print(f"❌ Schedule sync loop error: {e}")
                
                # Wait 60 seconds before next sync
                for _ in range(60):
                    if not self.sync_loop_running:
                        break
                    import time
                    time.sleep(1)
            
            print("⏹️ Schedule sync loop stopped")
        
        # Start the thread
        import threading
        self.sync_thread = threading.Thread(target=schedule_sync_loop, daemon=True)
        self.sync_thread.start()
        print(f"✅ Sync thread started (daemon: {self.sync_thread.daemon})")

    def stop_schedule_sync_loop(self):
        """Stop the background sync loop"""
        self.sync_loop_running = False
        print("⏹️ Stopping schedule sync loop...")
    
    # ========== FACE RECOGNITION ==========
    
    def scan_face(self, frame):
        """Scan a single frame for face recognition with anti-spoofing using UniFace only"""
        
        if self._detector is None or self._recognizer is None or self._spoof_detector is None:
            print("⚠️ UniFace models not initialized")
            return "NO_FACE", None, False, 0.0
        
        try:
            # Use a slightly larger image for better detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using RetinaFace
            faces = self._detector.detect(rgb_small_frame)
            
            if not faces:
                return "NO_FACE", None, False, 0.0
            
            # Process first detected face
            face = faces[0]
            
            # Check liveness (anti-spoofing)
            try:
                spoof_result = self._spoof_detector.predict(rgb_small_frame, face.bbox)
                is_live = spoof_result.is_real
                liveness_conf = spoof_result.confidence
            except Exception as e:
                print(f"⚠️ Liveness check error: {e}")
                is_live = False
                liveness_conf = 0.0
            
            # If it's a spoof, reject immediately
            if not is_live:
                return "SPOOF", None, False, liveness_conf
            
            # Get face embedding for recognition
            if len(self.known_faces) == 0:
                return "UNKNOWN", None, is_live, liveness_conf
            
            embedding = self._recognizer.get_embedding(rgb_small_frame, face.landmarks)
            embedding = embedding.flatten()
            
            # Compare with known faces (using cosine similarity)
            best_match_idx = -1
            best_similarity = -1
            
            for i, known_emb in enumerate(self.known_faces):
                # Ensure known_emb is 1D
                if len(known_emb.shape) > 1:
                    known_emb = known_emb.flatten()
                
                # Cosine similarity
                similarity = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            # Threshold for match (0.5 is reasonable for ArcFace)
            if best_similarity >= 0.5:
                return "FOUND", self.known_names[best_match_idx], is_live, liveness_conf
            else:
                return "UNKNOWN", None, is_live, liveness_conf
                
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
            return "NO_FACE", None, False, 0.0
    
    # ========== CLASS LIFECYCLE ==========
    
    def check_class_start(self):
        """Check if any class should start based on current time"""
        if self.current_schedule:
            return None
        
        now = datetime.now()
        today_date = now.date()
        
        for schedule in self.local_schedules:
            try:
                # Skip if ended
                if schedule.get('status') == 'ended':
                    continue
                
                sched_date = schedule.get('date')
                if sched_date:
                    if isinstance(sched_date, str):
                        sched_date = datetime.strptime(sched_date, "%Y-%m-%d").date()
                else:
                    sched_date = today_date
                
                if sched_date != today_date:
                    continue
                
                schedule_time = datetime.strptime(schedule['start_time'], "%H:%M:%S").time()
                class_start = datetime.combine(now.date(), schedule_time)
                
                seconds_until = (class_start - now).total_seconds()
                
                # Catch classes whose start time has arrived (seconds_until <= 0)
                # or is within 2 seconds in the future (handles precise timer firing slightly early).
                # Also catch up to 5 minutes late so a missed timer still works.
                if -300 <= seconds_until <= 2:
                    print(f"🕒 CLOCK: Starting class {schedule['subject']} - {schedule['section']} "
                          f"(delta: {seconds_until:.1f}s)")
                    
                    self.attendance_log.clear()
                    self.known_faces.clear()
                    self.known_names.clear()
                    self.known_student_ids.clear()
                    
                    self.current_schedule = schedule
                    self.current_section = schedule['section']
                    
                    self.stop_background_sync()
                    
                    face_count = self.load_section_students(self.current_section, allow_build=False)
                    if face_count == 0:
                        print(f"⚠️ No students loaded for section {self.current_section}")
                    
                    return schedule
                        
            except Exception as e:
                print(f"⚠️ Error processing schedule: {e}")
                continue
        
        return None

    def check_class_end(self):
        """Check if current class should end based on duration OR status"""
        if not self.current_schedule:
            return False
        
        try:
            now = datetime.now()
            
            # Check if status was changed to 'ended' externally
            if self.current_schedule.get('status') == 'ended':
                print(f"🕒 Class ended by status change")
                self.current_schedule = None
                self.current_section = None
                self.attendance_log.clear()
                return True
            
            # Also check time-based end
            schedule_time = datetime.strptime(self.current_schedule['start_time'], "%H:%M:%S").time()
            class_start = datetime.combine(now.date(), schedule_time)
            class_end = class_start + timedelta(minutes=self.current_schedule['duration'])
            
            if now >= class_end:
                return True
                
        except Exception as e:
            print(f"⚠️ Class end check error: {e}")
        
        return False
    
    def get_class_time_remaining(self):
        """Get remaining time for current class"""
        if not self.current_schedule:
            return None
        
        try:
            now = datetime.now()
            schedule_time = datetime.strptime(self.current_schedule['start_time'], "%H:%M:%S").time()
            class_start = datetime.combine(now.date(), schedule_time)
            class_end = class_start + timedelta(minutes=self.current_schedule['duration'])
            
            time_remaining = class_end - now
            
            if time_remaining.total_seconds() > 0:
                hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                if hours > 0:
                    return f"{hours}h {minutes}m"
                else:
                    return f"{minutes}m {seconds}s"
            else:
                return "Class Ended"
                
        except Exception as e:
            print(f"⚠️ Countdown calculation error: {e}")
            return "Error"
    
    def end_class(self):
        """End the current class and update status in database"""
        if not self.current_schedule:
            return False, "No active class"
        
        self.stop_background_sync()
        
        try:
            schedule_id = self.current_schedule['id']
            subject = self.current_schedule['subject']
            section = self.current_schedule['section']
            
            # Count students marked for THIS SPECIFIC SCHEDULE
            students_marked = 0
            for (sched_id, student_id) in self.attendance_log.keys():
                if sched_id == schedule_id:
                    students_marked += 1
            
            print(f"\n{'='*50}")
            print(f"🔄 ENDING CLASS: {subject} - {section}")
            print(f"📋 Schedule ID: {schedule_id}")
            print(f"📊 Students marked in this class: {students_marked}")
            print(f"{'='*50}")
            
            # DEBUG: Show local schedules before removal
            print(f"\n📋 BEFORE REMOVAL - Local schedules: {len(self.local_schedules)}")
            for i, s in enumerate(self.local_schedules):
                print(f"  {i+1}. {s['subject']} (ID: {s['id']}) {'⬅ CURRENT' if s['id'] == schedule_id else ''}")
            
            # ✅ Update database status to 'ended'
            try:
                print(f"\n📡 Updating database status to 'ended'...")
                result = self.supabase.table("schedule")\
                    .update({"status": "ended"})\
                    .eq("id", schedule_id)\
                    .execute()
                
                if result.data:
                    print(f"✅ Database updated: {result.data[0].get('status')}")
                else:
                    print(f"⚠️ No data returned from update")
                    
            except Exception as db_error:
                print(f"❌ Database error: {db_error}")
                # Continue with local cleanup even if DB update fails
            
            # ✅ Find and remove from local list
            found = False
            for i, schedule in enumerate(self.local_schedules[:]):  # Use slice copy
                if schedule['id'] == schedule_id:
                    print(f"✅ Found at index {i}: {schedule['subject']}")
                    self.local_schedules.remove(schedule)
                    print(f"✅ Removed from local list")
                    found = True
                    break
            
            if not found:
                print(f"❌ Could not find schedule {schedule_id} in local list!")
            
            # DEBUG: Show local schedules after removal
            print(f"\n📋 AFTER REMOVAL - Local schedules: {len(self.local_schedules)}")
            for i, s in enumerate(self.local_schedules):
                print(f"  {i+1}. {s['subject']} (ID: {s['id']})")
            
            # ✅ Clear ONLY attendance for this specific schedule
            print(f"\n🧹 Clearing attendance for schedule {schedule_id}...")
            keys_to_remove = []
            for key in self.attendance_log.keys():
                if key[0] == schedule_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.attendance_log[key]
            
            print(f"✅ Removed {len(keys_to_remove)} attendance records for this schedule")
            print(f"📊 Remaining attendance records: {len(self.attendance_log)}")
            
            # ✅ Clear current schedule
            self.current_schedule = None
            self.current_section = None
            
            # ✅ Clear face encodings for this section
            self.known_faces.clear()
            self.known_names.clear()
            self.known_student_ids.clear()
            print(f"✅ Cleared face encodings")
            
            # ✅ Refresh schedules
            self.sync_schedules(sync_students=False)
            
            print(f"\n✅ Class ended successfully")
            print(f"{'='*50}\n")
            
            return True, {
                'subject': subject,
                'section': section,
                'total_students': students_marked,
                'schedule_id': schedule_id
            }
            
        except Exception as e:
            print(f"❌ Error ending class: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
