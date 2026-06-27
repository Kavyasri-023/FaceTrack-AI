# 🎯 FaceTrack AI — Smart Attendance System with Face Recognition

FaceTrack AI is an intelligent, end-to-end attendance management system that uses real-time facial recognition to automatically identify students and mark their attendance through a webcam. It eliminates manual roll calls, prevents proxy attendance, and maintains clean, digital, exportable attendance records — built for hackathon-grade demos and real classroom use alike.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [MySQL Setup](#-mysql-setup)
- [Environment Variables](#-environment-variables)
- [Running the Application](#-running-the-application)
- [Usage Workflow](#-usage-workflow)
- [Screenshots](#-screenshots)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🚀 Overview

Traditional attendance systems are slow, manual, and easy to manipulate. **FaceTrack AI** solves this by combining computer vision and a lightweight web interface to:

- Register students with a few clicks and a webcam.
- Train a facial recognition model from captured images.
- Recognize faces live and mark attendance automatically — once per day, per student.
- Provide a clean dashboard for reviewing, searching, and exporting attendance data.

---

## ✨ Features

### Student Registration
- Capture Student ID, Full Name, Department, and Email.
- Input validation (including email format) and duplicate-ID/email prevention.
- Captures 20–30 face images per student via webcam, saved under `dataset/<student_id>/`.

### Face Encoding & Training
- Generates facial encodings using the `face_recognition` library.
- Stores encodings in `models/face_encodings.pkl`.
- One-click **Retrain Model** option after adding new students.

### Real-Time Face Recognition
- Live webcam face detection using OpenCV.
- Green bounding box with `Name` and `ID` for recognized students.
- `"Unknown User"` label for unregistered faces.
- Automatic attendance marking with duplicate-entry prevention (once per day).
- On-screen `"Attendance Marked Successfully!"` confirmation.

### Attendance Dashboard
- Total registered students, today's attendance count, and attendance percentage.
- Search by Student ID or name.
- Filter attendance history by date.
- Export attendance records to CSV.

### Modern UI
- Dark blue themed Streamlit interface.
- Sidebar navigation, dashboard cards, icons, and status indicators.
- Responsive layout suitable for live presentations.

---

## 🛠️ Technology Stack

| Layer                | Technology                          |
|-----------------------|--------------------------------------|
| Frontend              | Streamlit                            |
| Backend               | Python                               |
| Face Detection        | OpenCV                               |
| Face Recognition      | `face_recognition` (built on dlib)   |
| Database              | MySQL                                |
| Data Processing       | Pandas                               |
| Model Storage         | Pickle (`.pkl`)                      |
| Configuration         | python-dotenv                        |
| Version Control       | Git & GitHub                         |

---

## 📂 Folder Structure