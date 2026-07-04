"""
Document Viewer Logic - Separate from UI
Handles document fetching, caching, and management
"""

import os
import tempfile
import requests
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class DocumentViewerLogic(QObject):
    """Business logic for document viewing"""
    
    # Signals for UI updates
    terms_loaded = pyqtSignal(list)  # Emits list of terms
    document_ready = pyqtSignal(str)  # Emits path to temp PDF file
    document_error = pyqtSignal(str)   # Emits error message
    progress_update = pyqtSignal(int, str)  # progress %, message
    
    def __init__(self, supabase_client, config):
        super().__init__()
        self.supabase = supabase_client
        self.config = config
        self.current_pdf_path = None
        self.available_terms = []
        self.current_student = None
    
    def load_student_terms(self, student):
        """Load all academic terms for a student from database"""
        self.current_student = student
        student_id = student.get('id')
        
        self.progress_update.emit(0, "Loading academic terms...")
        print(f"🔍 Loading terms for student: {student_id}")
        
        try:
            # Get all enrollments with subject details
            response = self.supabase.table("enrollment_sub")\
                .select("subjects!inner(academic_term)")\
                .eq("stud_id", student_id)\
                .execute()
            
            print(f"📡 Enrollment response: {response}")
            
            terms = set()
            if response.data:
                print(f"📊 Found {len(response.data)} enrollments")
                for enrollment in response.data:
                    if enrollment.get('subjects') and enrollment['subjects'].get('academic_term'):
                        term = enrollment['subjects']['academic_term']
                        terms.add(term)
                        print(f"   - Term found: {term}")
            else:
                print("⚠️ No enrollment data found")
                # Try alternative query without inner join
                print("🔄 Trying alternative query...")
                alt_response = self.supabase.table("enrollment_sub")\
                    .select("*, subjects(academic_term)")\
                    .eq("stud_id", student_id)\
                    .execute()
                
                if alt_response.data:
                    print(f"📊 Alternative found {len(alt_response.data)} enrollments")
                    for enrollment in alt_response.data:
                        if enrollment.get('subjects') and enrollment['subjects'].get('academic_term'):
                            term = enrollment['subjects']['academic_term']
                            terms.add(term)
                            print(f"   - Term found: {term}")
            
            if not terms:
                # If still no terms, try to get from student's year level
                print("⚠️ No terms found in enrollments, using default")
                year_section = student.get('year-section', '')
                if year_section and len(year_section) > 0:
                    year = year_section[0]
                    # Add default terms for this year
                    terms.add(f"{year}-1")
                    terms.add(f"{year}-2")
                    if year == '1' or year == '2':
                        terms.add(f"{year}-S")
                    print(f"📅 Added default terms for year {year}")
            
            # Sort terms chronologically
            def term_sort_key(term):
                parts = term.split('-')
                if len(parts) == 2:
                    try:
                        year = int(parts[0])
                        sem = parts[1]
                        sem_value = 0
                        if sem == '1':
                            sem_value = 1
                        elif sem == '2':
                            sem_value = 2
                        elif sem == 'S' or sem == 's':
                            sem_value = 3
                        return (year, sem_value)
                    except:
                        return (0, 0)
                return (0, 0)
            
            self.available_terms = sorted(list(terms), key=term_sort_key, reverse=True)
            print(f"✅ Final terms: {self.available_terms}")
            
            # Format terms for display
            formatted_terms = []
            for term in self.available_terms:
                formatted_terms.append({
                    'raw': term,
                    'display': self.format_term_display(term)
                })
            
            self.terms_loaded.emit(formatted_terms)
            self.progress_update.emit(100, f"Loaded {len(formatted_terms)} terms")
            
        except Exception as e:
            print(f"❌ Error loading terms: {e}")
            import traceback
            traceback.print_exc()
            self.document_error.emit(f"Failed to load terms: {str(e)}")
    
    def format_term_display(self, term):
        """Convert raw term (e.g., '1-1') to display format (e.g., '1st Year, 1st Semester')"""
        if not term:
            return "Not specified"
        
        parts = term.split('-')
        if len(parts) == 2:
            year = parts[0]
            semester = parts[1]
            
            # Year ordinal
            if year == '1':
                year_display = "1st Year"
            elif year == '2':
                year_display = "2nd Year"
            elif year == '3':
                year_display = "3rd Year"
            elif year == '4':
                year_display = "4th Year"
            else:
                year_display = f"{year}th Year"
            
            # Semester display
            if semester == '1':
                sem_display = "1st Semester"
            elif semester == '2':
                sem_display = "2nd Semester"
            elif semester == 'S' or semester == 's':
                sem_display = "Summer"
            else:
                sem_display = semester
            
            return f"{year_display}, {sem_display}"
        
        return term
    
    def fetch_document(self, doc_type, term_raw):
        """Fetch document from Supabase storage"""
        if not self.current_student:
            self.document_error.emit("No student selected")
            return
        
        student_id = self.current_student.get('id')
        student_name = f"{self.current_student.get('fname', '')} {self.current_student.get('lname', '')}".strip()
        
        # Clean student ID for URL (replace hyphens with underscores)
        clean_student = student_id.replace('-', '_')
        
        # Construct URL
        url = f"{self.config.SUPABASE_URL}/storage/v1/object/public/documents/{doc_type}/{clean_student}/{term_raw}.pdf"
        
        print(f"🌐 Fetching document from: {url}")
        self.progress_update.emit(10, f"Fetching {doc_type} document...")
        
        try:
            # Download PDF
            response = requests.get(url, timeout=30)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"✅ Download successful! Size: {len(response.content)} bytes")
                self.progress_update.emit(50, "Download complete, saving temporarily...")
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(response.content)
                    self.current_pdf_path = tmp.name
                    print(f"💾 Saved to temp file: {self.current_pdf_path}")
                
                self.progress_update.emit(100, f"Document ready for {student_name}")
                self.document_ready.emit(self.current_pdf_path)
                
            elif response.status_code == 404:
                error_msg = f"Document not found at:\n{url}\n\nIt may not have been generated yet."
                print(f"❌ 404: {error_msg}")
                self.document_error.emit(error_msg)
            else:
                error_msg = f"Error {response.status_code}: Failed to fetch document\n{url}"
                print(f"❌ {error_msg}")
                self.document_error.emit(error_msg)
                
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error - Please check your internet"
            print(f"❌ {error_msg}")
            self.document_error.emit(error_msg)
        except requests.exceptions.Timeout:
            error_msg = "Timeout - Server took too long to respond"
            print(f"❌ {error_msg}")
            self.document_error.emit(error_msg)
        except Exception as e:
            error_msg = f"Error: {str(e)[:100]}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            self.document_error.emit(error_msg)
    
    def cleanup_current_document(self):
        """Delete the current temporary PDF file"""
        if self.current_pdf_path and os.path.exists(self.current_pdf_path):
            try:
                os.unlink(self.current_pdf_path)
                print(f"🧹 Deleted temp file: {self.current_pdf_path}")
                self.current_pdf_path = None
                return True
            except Exception as e:
                print(f"⚠️ Failed to delete temp file: {e}")
                return False
        return True
    
    def get_student_info(self):
        """Get current student info for display"""
        if self.current_student:
            return {
                'id': self.current_student.get('id', ''),
                'name': f"{self.current_student.get('fname', '')} {self.current_student.get('lname', '')}".strip(),
                'section': self.current_student.get('year-section', '')
            }
        return None