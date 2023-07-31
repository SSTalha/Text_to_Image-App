import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from auth_token import AuthToken

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

app = tk.Tk()
app.geometry("530x530")
app.title("Stable App")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white", font=("Arial", 20))
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=60) 

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid , revision= "fp16", torch_dtypes = torch.float16, use_auth_token = AuthToken)
pipe.to(device)


def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save("generatedimg.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(app, height=40, width=120, text_color="black", fg_color="grey", font=("Arial", 20),command=generate)
trigger.configure(text="Generate")
trigger.place(x=200, y=70) 

app.mainloop()
