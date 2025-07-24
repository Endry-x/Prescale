import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Campionamento Rosso 3×3", layout="wide")
st.title("Campionamento equispaziato del rosso (media 3 × 3 px)")

# ---------- Upload --------------------------------------------------
up = st.file_uploader("Carica un’immagine", ["png", "jpg", "jpeg"])
if up is None:
    st.info("Carica un’immagine per iniziare.")
    st.stop()

img       = Image.open(up).convert("RGB")
arr       = np.asarray(img)
h, w, _   = arr.shape

# ---------- Parametri griglia --------------------------------------
n_tot = st.number_input("Numero totale di punti da rilevare", 1, 50_000, 400)
n_x   = int(round(math.sqrt(n_tot * w / h)))
n_y   = int(round(n_tot / n_x))
n_tot = n_x * n_y
st.write(f"Punti effettivi: **{n_tot}** → griglia **{n_x} × {n_y}**")

xs      = np.linspace(0, w - 1, n_x, dtype=int)
ys_img  = np.linspace(0, h - 1, n_y, dtype=int)
ys_pix  = ys_img[::-1]                       # origine in basso
X, Ypix = np.meshgrid(xs, ys_pix)

# ---------- Media 3×3 ------------------------------------------------
r_med = np.empty_like(X, dtype=np.uint8)
for iy, y_pix in enumerate(ys_pix):
    for ix, x in enumerate(xs):
        y_img      = h - 1 - y_pix          # coord immagine
        x0, x1     = max(0, x-1), min(w, x+2)
        y0, y1     = max(0, y_img-1), min(h, y_img+2)
        r_med[iy, ix] = arr[y0:y1, x0:x1, 0].mean()

# ---------- Dataframe & download ------------------------------------
df = pd.DataFrame({
    "x": X.flatten(),
    "y": Ypix.flatten(),          # origine in basso-sinistra
    "R_medio": r_med.flatten()    # <-- ora 1-D
})
st.dataframe(df.head(10_000))
st.download_button("📥 Scarica CSV", df.to_csv(index=False).encode(),
                   "campionamento_rosso.csv", "text/csv")

# ---------- Overlay punti verdi -------------------------------------
draw_img = img.copy()
draw     = ImageDraw.Draw(draw_img)
for x, y_pix in zip(X.flatten(), Ypix.flatten()):
    y_img = h - 1 - y_pix
    draw.ellipse((x-2, y_img-2, x+2, y_img+2), fill=(0,255,0))
st.subheader("Immagine con punti (verde)")
st.image(draw_img, use_column_width=True)

# ---------- Heat-map opzionale --------------------------------------
with st.expander("Mostra heat-map intensità rosso"):
    fig, ax = plt.subplots()
    im = ax.imshow(r_med[::-1], cmap="Reds",
                   extent=[0, w, 0, h], origin="lower", aspect='auto')
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px] (origine in basso)")
    fig.colorbar(im, ax=ax, label="R medio")
    st.pyplot(fig)
