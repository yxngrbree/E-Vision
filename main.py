import cv2
import threading
import customtkinter as ctk
from PIL import Image
from deepface import DeepFace
import collections

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SentientVision(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Emotion Vision")
        self.geometry("1280x720")
        self.configure(fg_color="#1a1a1a")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#111111")
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        self.brand = ctk.CTkLabel(self.sidebar, text="Emotion Vision", font=ctk.CTkFont(size=24, weight="bold", family="Helvetica"))
        self.brand.pack(pady=(40, 10))
        
        self.subbrand = ctk.CTkLabel(self.sidebar, text="Check your emotion on camera motion", font=ctk.CTkFont(size=12), text_color="#666666")
        self.subbrand.pack(pady=(0, 30))

        self.stats_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.stats_container.pack(fill="x", padx=30)
        
        self.emotion_config = {
            "happy": {"emoji": "😊", "color": "#FFD700"},
            "sad": {"emoji": "😢", "color": "#1E90FF"},
            "angry": {"emoji": "😠", "color": "#FF4500"},
            "neutral": {"emoji": "😐", "color": "#808080"},
            "surprise": {"emoji": "😲", "color": "#00FF7F"},
            "fear": {"emoji": "😨", "color": "#9370DB"},
            "disgust": {"emoji": "🤢", "color": "#8B4513"}
        }
        
        self.stats_labels = {}
        self.counts = collections.Counter()
        
        for emo, cfg in self.emotion_config.items():
            f = ctk.CTkFrame(self.stats_container, fg_color="transparent")
            f.pack(fill="x", pady=5)
            l = ctk.CTkLabel(f, text=f"{cfg['emoji']} {emo.upper()}", font=ctk.CTkFont(size=13, weight="bold"))
            l.pack(side="left")
            v = ctk.CTkLabel(f, text="0", font=ctk.CTkFont(family="Consolas", size=14), text_color=cfg['color'])
            v.pack(side="right")
            self.stats_labels[emo] = v

        self.display_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#000000")
        self.display_frame.grid(row=0, column=1, padx=30, pady=30, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.display_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

        self.control_bar = ctk.CTkFrame(self, height=120, corner_radius=20, fg_color="#222222")
        self.control_bar.grid(row=1, column=1, padx=30, pady=(0, 30), sticky="ew")

        self.indicator = ctk.CTkLabel(self.control_bar, text="●", font=ctk.CTkFont(size=30), text_color="#333333")
        self.indicator.pack(side="left", padx=(30, 10))

        self.status_text = ctk.CTkLabel(self.control_bar, text="WAITING FOR INPUT...", font=ctk.CTkFont(size=26, weight="bold"))
        self.status_text.pack(side="left")

        self.conf_bar = ctk.CTkProgressBar(self.control_bar, width=350, height=12, progress_color="#3b8ed0")
        self.conf_bar.pack(side="right", padx=40)
        self.conf_bar.set(0)

        self.cap = cv2.VideoCapture(0)
        self.active = True
        
        self.logic_thread = threading.Thread(target=self.stream, daemon=True)
        self.logic_thread.start()

    def stream(self):
        while self.active:
            success, frame = self.cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            try:
                data = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if data:
                    target = data[0]
                    emo = target['dominant_emotion']
                    val = target['emotion'][emo] / 100
                    box = target['region']
                    
                    color_bgr = (200, 200, 200) # Default
                    cv2.rectangle(display_frame, (box['x'], box['y']), (box['x']+box['w'], box['y']+box['h']), (255, 255, 255), 2)
                    
                    self.after(0, self.sync_ui, emo, val)
            except:
                pass

            rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Use fixed size to ensure it fits the UI container
            img_ctk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(800, 500))
            
            self.after(0, self.update_image, img_ctk)

    def update_image(self, img_ctk):
        self.video_label.configure(image=img_ctk)
        self.video_label._image = img_ctk # This anchor prevents the blank screen

    def sync_ui(self, emo, val):
        cfg = self.emotion_config.get(emo, {"emoji": "●", "color": "#3b8ed0"})
        self.status_text.configure(text=f"{cfg['emoji']} {emo.upper()}")
        self.indicator.configure(text_color=cfg['color'])
        self.conf_bar.set(val)
        self.conf_bar.configure(progress_color=cfg['color'])
        
        self.counts[emo] += 1
        self.stats_labels[emo].configure(text=str(self.counts[emo] // 5)) # Scaled down for readability

    def close_app(self):
        self.active = False
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SentientVision()
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()