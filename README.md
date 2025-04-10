```bash
.
├── README.md
├── app.py                                      # main program
├── attendance.json                             # Attendance record
├── face_pairings.json                          # Face-to-student mappings
├── recognitionDemo
│   ├── FER
│   │   ├── model.h5                            # Emotion recognition model
│   ├── Yolo
│   │   ├── yolov8n-face.pt                     # YOLO face detection model
│   │   └── yolov8..
│   ├── classroom_Moinitor_System_debug.py
│   ├── classroom_Monitor_System.py             # Logic of Face and Emotion recognition
│   ├── recognition_emotion.py
│   └── recognition_face.py
├── requirements.txt
├── static
│   ├── css
│   │   └── style.css
│   ├── default_avatar.png                      # Default avatar image
│   ├── face_database                           # Directory to store captured face images
│   │   ├── 2XXXXXXXX
│   └── js
├── students.json                               # Student data
└── templates
    ├── attendance.html                         # HTML check attendance page
    └── index.html                              # HTML index page
```