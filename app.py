import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title("Estrazione intensità colore lungo un segmento")

# 1. Caricamento immagine
uploaded_file = st.file_uploader("Carica un'immagine", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Traccia una linea sull'immagine")
    
    # 2. Disegno su canvas
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

    # 3. Estrazione segmento e calcolo RGB
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][0]
        if obj["type"] == "line":
            x0, y0, x1, y1 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            num_points = st.slider("Numero di punti da campionare", 5, 200, 50)

            xs = np.linspace(x0, x1, num_points).astype(int)
            ys = np.linspace(y0, y1, num_points).astype(int)

            rgb_values = []
            for x, y in zip(xs, ys):
                if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
                    r, g, b = img_array[y, x]
                    rgb_values.append((x, y, r, g, b))

            # Tabella
            df = pd.DataFrame(rgb_values, columns=["x", "y", "R", "G", "B"])
            st.subheader("Valori RGB lungo il segmento")
            st.dataframe(df)

            # Grafico
            st.subheader("Grafico intensità colore")
            fig, ax = plt.subplots()
            ax.plot(range(num_points), df["R"], label="R", color="red")
            ax.plot(range(num_points), df["G"], label="G", color="green")
            ax.plot(range(num_points), df["B"], label="B", color="blue")
            ax.set_xlabel("Punto lungo il segmento")
            ax.set_ylabel("Intensità")
            ax.legend()
            st.pyplot(fig)

            # Esportazione CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Scarica CSV", data=csv, file_name="colori_segmento.csv", mime="text/csv")
