-- ============================================================
-- ADV-Attendify Complete Database Schema
-- PostgreSQL / Supabase
-- ============================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. USER TABLES
-- ============================================================

-- Students
CREATE TABLE student (
    id TEXT PRIMARY KEY,
    fname TEXT NOT NULL,
    lname TEXT NOT NULL,
    uname TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- SHA-256 hashed
    "year-section" TEXT,
    "profile-pic" TEXT,
    "e-sign" TEXT,           -- JSON signature data
    info TEXT,               -- "About Me" section
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE student IS 'Student accounts and personal information';

-- Professors
CREATE TABLE professor (
    uid TEXT PRIMARY KEY,
    fname TEXT NOT NULL,
    lname TEXT NOT NULL,
    uname TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- SHA-256 hashed
    "profile-pic" TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE professor IS 'Professor accounts and personal information';

-- Admins
CREATE TABLE admin (
    id SERIAL PRIMARY KEY,
    uname TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- SHA-256 hashed
    "prof_pic" TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE admin IS 'Administrator accounts';

-- ============================================================
-- 2. SUBJECT & ENROLLMENT TABLES
-- ============================================================

-- Subject master list (system-wide)
CREATE TABLE list_of_subs (
    id SERIAL PRIMARY KEY,
    subject TEXT NOT NULL,
    code TEXT,
    units INTEGER DEFAULT 3,
    prerequisite TEXT[],      -- Array of prerequisite subject names
    "academic-term" TEXT,     -- Format: "1-1", "2-2", "3-S", etc.
    year_level INTEGER,
    semester INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(subject, "academic-term")
);

COMMENT ON TABLE list_of_subs IS 'Master list of all available subjects with prerequisites';

-- Active subjects (assigned to professors)
CREATE TABLE subjects (
    id BIGINT PRIMARY KEY,
    sub TEXT NOT NULL,
    sub_code TEXT,
    prof_id TEXT REFERENCES professor(uid) ON DELETE SET NULL,
    professor_name TEXT,
    academic_term TEXT,
    section TEXT,             -- 'Any', 'A', 'B', 'C', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE subjects IS 'Subjects currently assigned to professors';

-- Subject enrollment (students enrolled in subjects)
CREATE TABLE enrollment_sub (
    enroll_id SERIAL PRIMARY KEY,
    sub_id BIGINT REFERENCES subjects(id) ON DELETE CASCADE,
    stud_id TEXT REFERENCES student(id) ON DELETE CASCADE,
    year_section TEXT,
    grades TEXT,              -- Can be numeric or 'IP' (In Progress)
    date_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(sub_id, stud_id)
);

COMMENT ON TABLE enrollment_sub IS 'Student enrollments in subjects with grades';

-- Archived grades (moved when subjects are deleted)
CREATE TABLE subject_mirror (
    id BIGINT PRIMARY KEY,
    student_id TEXT REFERENCES student(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    grade TEXT,
    professor_id TEXT,
    academic_term TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE subject_mirror IS 'Archived grades from deleted subjects';

-- ============================================================
-- 3. SECTION MANAGEMENT
-- ============================================================

-- Available year-sections (e.g., '1A', '2B', '3C')
CREATE TABLE list_of_year_section (
    id SERIAL PRIMARY KEY,
    "year-section" TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE list_of_year_section IS 'Available year-section combinations';

-- Student section assignments (historical tracking)
CREATE TABLE year_section (
    id SERIAL PRIMARY KEY,
    "student-id" TEXT REFERENCES student(id) ON DELETE CASCADE,
    "student-name" TEXT,
    "year-section" TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE year_section IS 'Student section assignment history';

-- ============================================================
-- 4. SCHEDULING & ATTENDANCE
-- ============================================================

-- Class schedules
CREATE TABLE schedule (
    id SERIAL PRIMARY KEY,
    sub TEXT NOT NULL,
    date DATE NOT NULL,
    "time-in" TIME NOT NULL,
    "time-out" TIME NOT NULL,
    late TIME DEFAULT '00:15:00',  -- Grace period
    "year-section" TEXT NOT NULL,
    status TEXT DEFAULT 'scheduled',  -- 'scheduled', 'active', 'ended'
    uid TEXT REFERENCES professor(uid) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE schedule IS 'Class schedules with status tracking';

-- Attendance records
CREATE TABLE schedule_list (
    id SERIAL PRIMARY KEY,
    sched_id INTEGER REFERENCES schedule(id) ON DELETE CASCADE,
    student_id TEXT REFERENCES student(id) ON DELETE CASCADE,
    time_in TIME,
    status TEXT DEFAULT 'absent',  -- 'present', 'late', 'absent', 'pending'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(sched_id, student_id)
);

COMMENT ON TABLE schedule_list IS 'Attendance records for each class session';

-- ============================================================
-- 5. REQUEST & APPROVAL TABLES
-- ============================================================

-- Professor registration requests
CREATE TABLE requests_professor (
    uid TEXT PRIMARY KEY,
    fname TEXT NOT NULL,
    lname TEXT NOT NULL,
    uname TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    type TEXT DEFAULT 'professor_registration',
    date_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE requests_professor IS 'Pending professor registration requests';

-- Subject creation requests
CREATE TABLE requests_subject (
    id SERIAL PRIMARY KEY,
    sub TEXT NOT NULL,
    sub_code TEXT,
    uid TEXT REFERENCES professor(uid) ON DELETE CASCADE,
    prof_name TEXT,
    type TEXT DEFAULT 'subject_request',
    acad_term TEXT,
    section TEXT,
    date_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE requests_subject IS 'Pending subject creation requests';

-- Advising form requests
CREATE TABLE request_advising (
    id SERIAL PRIMARY KEY,
    student_id TEXT REFERENCES student(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE request_advising IS 'Student advising form requests';

-- ============================================================
-- 6. ADVISING & DOCUMENTS
-- ============================================================

-- Approved advising forms
CREATE TABLE advising (
    id SERIAL PRIMARY KEY,
    student_id TEXT REFERENCES student(id) ON DELETE CASCADE,
    academic_term TEXT,
    subjects JSONB,            -- Array of {code, title, units}
    approved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE advising IS 'Approved advising forms with subject lists';

-- ============================================================
-- 7. FACE IMAGES
-- ============================================================

-- Student face photos (stored in Supabase Storage)
CREATE TABLE face_images (
    id SERIAL PRIMARY KEY,
    "student-ID" TEXT REFERENCES student(id) ON DELETE CASCADE,
    "image-url" TEXT NOT NULL,
    "file-name" TEXT NOT NULL,
    "uploaded-at" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE face_images IS 'Student face photos for recognition';

-- ============================================================
-- 8. INDEXES FOR PERFORMANCE
-- ============================================================

-- Student indexes
CREATE INDEX idx_student_uname ON student(uname);
CREATE INDEX idx_student_email ON student(email);
CREATE INDEX idx_student_year_section ON student("year-section");

-- Professor indexes
CREATE INDEX idx_professor_uname ON professor(uname);
CREATE INDEX idx_professor_email ON professor(email);

-- Subject indexes
CREATE INDEX idx_subjects_prof_id ON subjects(prof_id);
CREATE INDEX idx_subjects_academic_term ON subjects(academic_term);
CREATE INDEX idx_subjects_section ON subjects(section);

-- Enrollment indexes
CREATE INDEX idx_enrollment_sub_stud_id ON enrollment_sub(stud_id);
CREATE INDEX idx_enrollment_sub_sub_id ON enrollment_sub(sub_id);
CREATE INDEX idx_enrollment_sub_grades ON enrollment_sub(grades);

-- Schedule indexes
CREATE INDEX idx_schedule_date ON schedule(date);
CREATE INDEX idx_schedule_status ON schedule(status);
CREATE INDEX idx_schedule_uid ON schedule(uid);
CREATE INDEX idx_schedule_year_section ON schedule("year-section");

-- Schedule list indexes
CREATE INDEX idx_schedule_list_sched_id ON schedule_list(sched_id);
CREATE INDEX idx_schedule_list_student_id ON schedule_list(student_id);
CREATE INDEX idx_schedule_list_status ON schedule_list(status);

-- Request indexes
CREATE INDEX idx_requests_professor_email ON requests_professor(email);
CREATE INDEX idx_requests_professor_uname ON requests_professor(uname);
CREATE INDEX idx_requests_subject_uid ON requests_subject(uid);
CREATE INDEX idx_requests_subject_acad_term ON requests_subject(acad_term);
CREATE INDEX idx_request_advising_student_id ON request_advising(student_id);

-- Face images indexes
CREATE INDEX idx_face_images_student_id ON face_images("student-ID");

-- ============================================================
-- 9. ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE student ENABLE ROW LEVEL SECURITY;
ALTER TABLE professor ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin ENABLE ROW LEVEL SECURITY;
ALTER TABLE subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE enrollment_sub ENABLE ROW LEVEL SECURITY;
ALTER TABLE schedule ENABLE ROW LEVEL SECURITY;
ALTER TABLE schedule_list ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE request_advising ENABLE ROW LEVEL SECURITY;
ALTER TABLE advising ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- 9a. STUDENT POLICIES
-- ============================================================

-- Students can view their own data
CREATE POLICY "Students can view own data"
ON student FOR SELECT
USING (auth.uid()::text = id);

-- Students can update their own data
CREATE POLICY "Students can update own data"
ON student FOR UPDATE
USING (auth.uid()::text = id);

-- ============================================================
-- 9b. PROFESSOR POLICIES
-- ============================================================

-- Professors can view their own data
CREATE POLICY "Professors can view own data"
ON professor FOR SELECT
USING (auth.uid()::text = uid);

-- Professors can update their own data
CREATE POLICY "Professors can update own data"
ON professor FOR UPDATE
USING (auth.uid()::text = uid);

-- ============================================================
-- 9c. FACE IMAGES POLICIES
-- ============================================================

-- Students can insert their own photos
CREATE POLICY "Students can insert own photos"
ON face_images FOR INSERT
WITH CHECK (auth.uid()::text = "student-ID");

-- Students can view their own photos
CREATE POLICY "Students can view own photos"
ON face_images FOR SELECT
USING (auth.uid()::text = "student-ID");

-- Students can delete their own photos
CREATE POLICY "Students can delete own photos"
ON face_images FOR DELETE
USING (auth.uid()::text = "student-ID");

-- ============================================================
-- 9d. STORAGE POLICIES (Supabase Storage)
-- ============================================================

-- Note: These are applied to the storage.objects table
-- They need to be created in the Supabase dashboard

/*
-- Allow students to upload their own files
CREATE POLICY "Students can upload own files"
ON storage.objects FOR INSERT
WITH CHECK (auth.uid()::text = (storage.foldername(name))[1]);

-- Allow students to view their own files
CREATE POLICY "Students can view own files"
ON storage.objects FOR SELECT
USING (auth.uid()::text = (storage.foldername(name))[1]);

-- Allow students to delete their own files
CREATE POLICY "Students can delete own files"
ON storage.objects FOR DELETE
USING (auth.uid()::text = (storage.foldername(name))[1]);
*/

-- ============================================================
-- 9e. ENROLLMENT POLICIES
-- ============================================================

-- Students can view their own enrollments
CREATE POLICY "Students can view own enrollments"
ON enrollment_sub FOR SELECT
USING (auth.uid()::text = stud_id);

-- ============================================================
-- 9f. SCHEDULE POLICIES
-- ============================================================

-- Professors can view their own schedules
CREATE POLICY "Professors can view own schedules"
ON schedule FOR SELECT
USING (auth.uid()::text = uid);

-- Professors can insert their own schedules
CREATE POLICY "Professors can insert own schedules"
ON schedule FOR INSERT
WITH CHECK (auth.uid()::text = uid);

-- Professors can update their own schedules
CREATE POLICY "Professors can update own schedules"
ON schedule FOR UPDATE
USING (auth.uid()::text = uid);

-- Professors can delete their own schedules
CREATE POLICY "Professors can delete own schedules"
ON schedule FOR DELETE
USING (auth.uid()::text = uid);

-- ============================================================
-- 9g. SCHEDULE LIST POLICIES
-- ============================================================

-- Professors can view all attendance records
CREATE POLICY "Professors can view attendance"
ON schedule_list FOR SELECT
USING (EXISTS (
    SELECT 1 FROM schedule s
    WHERE s.id = schedule_list.sched_id
    AND s.uid = auth.uid()::text
));

-- Students can view their own attendance
CREATE POLICY "Students can view own attendance"
ON schedule_list FOR SELECT
USING (auth.uid()::text = student_id);

-- ============================================================
-- 9h. REQUEST ADVISING POLICIES
-- ============================================================

-- Students can view their own requests
CREATE POLICY "Students can view own advising requests"
ON request_advising FOR SELECT
USING (auth.uid()::text = student_id);

-- Students can insert their own requests
CREATE POLICY "Students can insert own advising requests"
ON request_advising FOR INSERT
WITH CHECK (auth.uid()::text = student_id);

-- ============================================================
-- 9i. ADVISING POLICIES
-- ============================================================

-- Students can view their own advising records
CREATE POLICY "Students can view own advising"
ON advising FOR SELECT
USING (auth.uid()::text = student_id);

-- ============================================================
-- 10. FUNCTIONS & TRIGGERS
-- ============================================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
CREATE TRIGGER update_student_updated_at
BEFORE UPDATE ON student
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_professor_updated_at
BEFORE UPDATE ON professor
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_admin_updated_at
BEFORE UPDATE ON admin
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subjects_updated_at
BEFORE UPDATE ON subjects
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_enrollment_sub_updated_at
BEFORE UPDATE ON enrollment_sub
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_schedule_updated_at
BEFORE UPDATE ON schedule
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_schedule_list_updated_at
BEFORE UPDATE ON schedule_list
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_request_advising_updated_at
BEFORE UPDATE ON request_advising
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 11. HELPER FUNCTIONS
-- ============================================================

-- Function to get server timestamp (for PDF generation)
CREATE OR REPLACE FUNCTION get_server_timestamp()
RETURNS TIMESTAMP WITH TIME ZONE AS $$
BEGIN
    RETURN NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to get student's current academic term
CREATE OR REPLACE FUNCTION get_student_current_term(p_student_id TEXT)
RETURNS TEXT AS $$
DECLARE
    v_term TEXT;
BEGIN
    SELECT academic_term
    FROM enrollment_sub es
    JOIN subjects s ON es.sub_id = s.id
    WHERE es.stud_id = p_student_id
    GROUP BY s.academic_term
    ORDER BY MAX(es.created_at) DESC
    LIMIT 1
    INTO v_term;
    
    RETURN v_term;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 12. SAMPLE DATA (Optional - for testing)
-- ============================================================

-- Sample year-sections
INSERT INTO list_of_year_section ("year-section") VALUES
('1A'), ('1B'), ('1C'),
('2A'), ('2B'), ('2C'),
('3A'), ('3B'), ('3C'),
('4A'), ('4B'), ('4C')
ON CONFLICT ("year-section") DO NOTHING;

-- ============================================================
-- 13. DOCUMENTATION
-- ============================================================

COMMENT ON DATABASE adv_attendify IS 'ADV-Attendify Attendance System Database';

-- Table documentation
COMMENT ON COLUMN student."e-sign" IS 'JSON-encoded signature data from signature pad';
COMMENT ON COLUMN subjects.section IS 'Section identifier: Any, A, B, C, etc.';
COMMENT ON COLUMN enrollment_sub.grades IS 'Grade value: numeric (e.g., 85) or IP (In Progress)';
COMMENT ON COLUMN schedule.status IS 'Schedule status: scheduled, active, ended';
COMMENT ON COLUMN schedule_list.status IS 'Attendance status: present, late, absent, pending';

-- ============================================================
-- END OF SCHEMA
-- ============================================================