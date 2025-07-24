import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.set_page_config(page_title="RGB Line Sampler", layout="wide")
st.title("Estrazione intensità colore lungo un segmento")

# -- Sidebar -----------------------------------------------------------
st.sidebar.header("Impostazioni")
avg_kernel = st.sidebar.checkbox("Media locale 3×3 (meno rumore)", value=False)
line_width = st.sidebar.slider("Spessore linea grafico", 1, 5, 2)
show_all_objects = st.sidebar.checkbox("Permetti più segmenti", value=False)

# -- Caricamento immagine ---------------------------------------------
uploaded_file = st.file_uploader("Carica un'immagine", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("➡️ Carica un’immagine per iniziare.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

st.subheader("Traccia una linea sull'immagine")

# -- Canvas ------------------------------------------------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#ff0000",
    background_image=image,
    update_streamlit=True,
    height=image.height,
    width=image.width,
    drawing_mode="line",
    key="canvas",
)

# -- Elaborazione ------------------------------------------------------
objects = canvas_result.json_data["objects"] if canvas_result.json_data else []

if not objects:
    st.warning("Disegna un segmento rosso per continuare.")
    st.stop()

# Se “più segmenti” è disattivato, considera solo il primo oggetto “line”
if not show_all_objects:
    objects = [obj for obj in objects if obj["type"] == "line"][:1]

segment_counter = 0
all_dataframes = []

for obj in objects:
    if obj["type"] != "line":
        continue

    x0, y0, x1, y1 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
    segment_counter += 1
    st.markdown(f"### Segmento {segment_counter}")

    num_points = st.slider(
        "Numero di punti da campionare",
        5, 200, 50,
        key=f"slider_{segment_counter}"
    )

    xs = np.linspace(x0, x1, num_points).astype(int)
    ys = np.linspace(y0, y1, num_points).astype(int)

    rgb_values = []
    for x, y in zip(xs, ys):
        if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
            if avg_kernel:
                # media su finestra 3×3 (gestione bordi)
                x0k, x1k = max(0, x-1), min(img_array.shape[1], x+2)
                y0k, y1k = max(0, y-1), min(img_array.shape[0], y+2)
                r, g, b = img_array[y0k:y1k, x0k:x1k].mean(axis=(0, 1))
            else:
                r, g, b = img_array[y, x]
            rgb_values.append((x, y, int(r), int(g), int(b)))

    # -- Tabella -------------------------------------------------------
    df = pd.DataFrame(rgb_values, columns=["x", "y", "R", "G", "B"])
    all_dataframes.append(df)
    st.dataframe(df, use_container_width=True)

    # -- Grafico -------------------------------------------------------
    fig, ax = plt.subplots()
    idx = range(len(df))
    ax.plot(idx, df["R"], label="R", color="red", linewidth=line_width)
    ax.plot(idx, df["G"], label="G", color="green", linewidth=line_width)
    ax.plot(idx, df["B"], label="B", color="blue", linewidth=line_width)
    ax.set_xlabel("Punto lungo il segmento")
    ax.set_ylabel("Intensità")
    ax.legend()
    st.pyplot(fig)

# -- Download unico (CSV ZIP) ------------------------------------------
if all_dataframes:
    # Se un solo segmento → CSV singolo; se più segmenti → ZIP
    if len(all_dataframes) == 1:
        csv = all_dataframes[0].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Scarica CSV",
            data=csv,
            file_name="colori_segmento.csv",
            mime="text/csv"
        )
    else:
        import io, zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for i, df in enumerate(all_dataframes, start=1):
                zf.writestr(f"segmento_{i}.csv", df.to_csv(index=False))
        st.download_button(
            "📥 Scarica tutti i CSV (ZIP)",
            data=buffer.getvalue(),
            file_name="colori_segmenti.zip",
            mime="application/zip"
        )

