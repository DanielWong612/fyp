```bash
project_root/
│
├── app.py
├── recognitionDemo/
│   ├── classroom_Monitor_System.py
│   ├── face_database/           # Directory to store captured face images
│   ├── FER/
│   │   └── model.h5             # Emotion recognition model
│   └── Yolo/
│       └── yolov8n-face.pt      # YOLO face detection model
├── static/
│   ├── thumbnails/              # Directory for face thumbnails
│   └── default_avatar.png       # Default avatar image
├── templates/
│   └── index.html               # HTML template
├── students.json                # Student data
└── face_pairings.json           # Face-to-student mappings
```