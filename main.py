import customtkinter as ctk
from google import genai
import whisper
import edge_tts
import sounddevice as sd
import soundfile as sf
import numpy as np
import asyncio
import threading
import re
import os

# --- КОНФИГУРАЦИЯ ---
API_KEY = "AIzaSyDZVhaCZgkzVBkuY7NO3GWsHyKn5b_Mg2Q"
client = genai.Client(api_key=API_KEY)

class VoiceBot(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Настройка окна
        self.title("Нейронный термина v3.0")
        self.geometry("900x600")
        ctk.set_appearance_mode("dark")
        
        # Переменные настроек
        self.voice_var = ctk.StringVar(value="ru-RU-DmitryNeural")
        self.speed_var = ctk.IntVar(value=15)
        self.model_var = ctk.StringVar(value="small")
        
        self.stt_model = None
        self.is_recording = False
        self.audio_data = []

        self.setup_ui()

    def setup_ui(self):
        # Сетка: левая панель (250px) и правая (основная)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- ЛЕВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#0f0f12", border_width=1, border_color="#1f1f2e")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.side_label = ctk.CTkLabel(self.sidebar, text="CORE SETTINGS", font=("Share Tech Mono", 18, "bold"), text_color="#00d4ff")
        self.side_label.pack(pady=30, padx=20)

        # Выбор голоса
        ctk.CTkLabel(self.sidebar, text="VOICE UNIT:", font=("Share Tech Mono", 12)).pack(anchor="w", padx=20)
        self.voice_menu = ctk.CTkOptionMenu(self.sidebar, values=["ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"], 
                                           variable=self.voice_var, fg_color="#1a1a2e", button_color="#00d4ff")
        self.voice_menu.pack(pady=(5, 20), padx=20, fill="x")

        # Скорость речи
        ctk.CTkLabel(self.sidebar, text="SPEECH RATE:", font=("Share Tech Mono", 12)).pack(anchor="w", padx=20)
        self.speed_slider = ctk.CTkSlider(self.sidebar, from_=-20, to=50, variable=self.speed_var, progress_color="#00d4ff")
        self.speed_slider.pack(pady=(5, 20), padx=20, fill="x")

        # Модель Whisper
        ctk.CTkLabel(self.sidebar, text="NEURAL MODEL:", font=("Share Tech Mono", 12)).pack(anchor="w", padx=20)
        self.model_menu = ctk.CTkOptionMenu(self.sidebar, values=["tiny", "base", "small", "medium", "large", "turbo"], 
                                            variable=self.model_var, command=self.reload_model,
                                            fg_color="#1a1a2e", button_color="#00d4ff")
        self.model_menu.pack(pady=(5, 20), padx=20, fill="x")

        self.info_label = ctk.CTkLabel(self.sidebar, text="v3.0 OMNI-LINK\nSTATUS: ONLINE", font=("Share Tech Mono", 10), text_color="#444")
        self.info_label.pack(side="bottom", pady=20)

        # --- ПРАВАЯ ПАНЕЛЬ (ЧАТ) ---
        self.main_container = ctk.CTkFrame(self, fg_color="#0a0a0c", corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=1, pady=1)

        self.chat_box = ctk.CTkTextbox(self.main_container, font=("Consolas", 14), fg_color="#0d0d11", border_color="#00d4ff", border_width=1)
        self.chat_box.pack(pady=30, padx=30, fill="both", expand=True)
        self.chat_box.configure(state="disabled")

        self.status_bar = ctk.CTkLabel(self.main_container, text="READY FOR COMMAND", font=("Share Tech Mono", 12), text_color="#00d4ff")
        self.status_bar.pack()

        self.btn_mic = ctk.CTkButton(self.main_container, text="[ INITIATE NEURAL LINK ]", 
                                     font=("Share Tech Mono", 16, "bold"),
                                     height=60, fg_color="transparent", border_width=2, border_color="#00d4ff",
                                     hover_color="#002b36")
        self.btn_mic.pack(pady=30, padx=100, fill="x")
        
        self.btn_mic.bind("<ButtonPress-1>", self.start_rec)
        self.btn_mic.bind("<ButtonRelease-1>", self.stop_rec)

    # --- ЛОГИКА ---

    def reload_model(self, choice):
        self.stt_model = None
        self.log(f"System: Switching to {choice} model...")

    def log(self, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"> {text}\n\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def start_rec(self, event):
        if not self.stt_model:
            self.status_bar.configure(text="BOOTING NEURAL ENGINE...", text_color="yellow")
            self.update()
            self.stt_model = whisper.load_model(self.model_var.get())

        self.is_recording = True
        self.audio_data = []
        self.status_bar.configure(text="RECEIVING AUDIO...", text_color="#ff0055")
        self.btn_mic.configure(border_color="#ff0055", text_color="#ff0055")
        threading.Thread(target=self._record_loop).start()

    def _record_loop(self):
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
            while self.is_recording:
                chunk, _ = stream.read(1024)
                self.audio_data.append(chunk.copy())

    def stop_rec(self, event):
        self.is_recording = False
        self.status_bar.configure(text="DECODING...", text_color="#00d4ff")
        self.btn_mic.configure(border_color="#00d4ff", text_color="#fff")
        threading.Thread(target=self.process).start()

    def process(self):
        if not self.audio_data: return
        audio_np = np.concatenate(self.audio_data, axis=0).flatten()
        
        res = self.stt_model.transcribe(audio_np, language="ru", fp16=False)
        user_text = res["text"].strip()
        
        if not user_text: return
        self.after(0, lambda: self.log(f"USER: {user_text}"))

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                config={"system_instruction": "Отвечай естественно. Никакой разметки. Не используй разметку, жирный текст или списки. Только чистый текст. Отвечай не более десяти предложений."},
                contents=user_text
            )
            ai_text = resp.text
            self.after(0, lambda: self.log(f"AI: {ai_text}"))
            
            # Очистка и запуск TTS
            clean = re.sub(r'[\*\#\_\-\>]', '', ai_text)
            asyncio.run(self.speak(clean))
            
        except Exception as e:
            self.after(0, lambda: self.log(f"CORE ERROR: {str(e)}"))

    async def speak(self, text):
        self.after(0, lambda: self.status_bar.configure(text="SYNTHESIZING..."))
        
        speed = f"{'+' if self.speed_var.get() >=0 else ''}{self.speed_var.get()}%"
        communicate = edge_tts.Communicate(text, self.voice_var.get(), rate=speed)
        await communicate.save("out.mp3")
        
        try:
            data, fs = sf.read("out.mp3")
            self.after(0, lambda: self.status_bar.configure(text="TRANSMITTING VOICE", text_color="#00ff41"))
            sd.play(data, fs)
            sd.wait()
        except: pass
        
        self.after(0, lambda: self.status_bar.configure(text="READY FOR COMMAND", text_color="#00d4ff"))

if __name__ == "__main__":
    app = VoiceBot()
    app.mainloop()