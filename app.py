import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(page_title="RGB Line Sampler", layout="wide")
st.title("Estrazione intensità colore lungo un segmento")

# -- Sidebar -----------------------------------------------------------
st.sidebar.header("Impostazioni")
avg_kernel = st.sidebar.checkbox("Media locale 3×3 (meno rumore)", value=False)
line_width = st.sidebar.slider("Spessore linea grafico", 1, 5, 2)
multi_segment = st.sidebar.checkbox("Permetti più segmenti", value=False)

# -- Caricamento immagine ---------------------------------------------
uploaded_file = st.file_uploader("Carica un'immagine", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("➡️ Carica un’immagine per iniziare.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

# Converte in base64 → data-URL
buf = BytesIO()
image.save(buf, format="PNG")
img_b64 = base64.b64encode(buf.getvalue()).decode()
img_data_url = f"data:image/png;base64,{img_b64}"

st.subheader("Traccia una linea sull'immagine")

# -- Canvas ------------------------------------------------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#ff0000",
    background_image_url=img_data_url,   # <-- richiede streamlit-drawable-canvas ≥ 0.9.5
    update_streamlit=True,
    height=image.height,
    width=image.width,
    drawing_mode="line",
    key="canvas",
)

# -- Elaborazione ------------------------------------------------------
objs = canvas_result.json_data["objects"] if canvas_result.json_data else []
lines = [o for o in objs if o["type"] == "line"]
if not lines:
    st.warning("Disegna almeno un segmento rosso per continuare.")
    st.stop()
if not multi_segment:
    lines = lines[:1]

all_dfs = []
for idx, ln in enumerate(lines, 1):
    x0, y0, x1, y1 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
    st.markdown(f"### Segmento {idx}")
    n_pts = st.slider("Numero di punti", 5, 200, 50, key=f"pts{idx}")

    xs = np.linspace(x0, x1, n_pts).astype(int)
    ys = np.linspace(y0, y1, n_pts).astype(int)

    rgb = []
    for x, y in zip(xs, ys):
        if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
            if avg_kernel:
                sl = img_array[max(0, y-1):y+2, max(0, x-1):x+2]
                r, g, b = sl.mean(axis=(0, 1))
            else:
                r, g, b = img_array[y, x]
            rgb.append((x, y, int(r), int(g), int(b)))

    df = pd.DataFrame(rgb, columns=["x", "y", "R", "G", "B"])
    all_dfs.append(df)

    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots()
    idxs = np.arange(len(df))
    ax.plot(idxs, df["R"], "r", linewidth=line_width, label="R")
    ax.plot(idxs, df["G"], "g", linewidth=line_width, label="G")
    ax.plot(idxs, df["B"], "b", linewidth=line_width, label="B")
    ax.set_xlabel("Punto lungo il segmento")
    ax.set_ylabel("Intensità")
    ax.legend()
    st.pyplot(fig)

# -- Download ----------------------------------------------------------
if all_dfs:
    if len(all_dfs) == 1:
        csv = all_dfs[0].to_csv(index=False).encode()
        st.download_button("📥 Scarica CSV", csv, "colori_segmento.csv", "text/csv")
    else:
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i, df in enumerate(all_dfs, 1):
                zf.writestr(f"segmento_{i}.csv", df.to_csv(index=False))
        st.download_button("📥 Scarica tutti i CSV (ZIP)", buf.getvalue(),
                           "colori_segmenti.zip", "application/zip")


