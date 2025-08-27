import streamlit as st
from streamlit_drawable_canvas import st_canvas

import numpy as np
import torch
from PIL import Image, ImageOps

from kmnist_cnn import CNN

class KMNISTStreamlitApp:
    """
    A class that creates the KMNIST Streamlit App.
    """
    def __init__(self):
        # Model Setup
        self.model = CNN()
        self.weights = torch.load("./kmnist_cnn_weights.pth", map_location="cpu")
        self.model.load_state_dict(self.weights)
        self.model.eval()

        # Mapping
        self.mapping = {0: "お (o)", 1: "き (ki)", 2: "す (su)", 3: "つ (tsu)", 4: "な (na)", 5: "は (ha)", 6: "ま (ma)", 7: "や (ya)", 8: "れ (re)", 9: "を (wo)"}

    def process_image(self, image):
        """
        Helper function to convert the PIL image to the format for the model to run.
        """
        image = image.convert("L") # Grayscale
        image = image.resize((28, 28), resample=Image.LANCZOS) # Using LANCZOS to downsample but with higher quality for better results
        image = ImageOps.invert(image) # Inverting the image
        image = np.array(image) / 255.0
        image = (image - 0.1918) / 0.3483 # Normalization for KMNIST (since we are doing BatchNorm)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return image

    def main(self):
        """ 
        Main Streamlit App Logic.
        """
        st.title("KMNIST Character Recognition App")
        st.write("Draw a KMNIST character below to see if we can predict it correctly!")

        # Drawing Canvas
        canvas_result = st_canvas( 
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        predict_button = st.button("Predict!")

        if predict_button and canvas_result.image_data is not None:
            pil_image = Image.fromarray(canvas_result.image_data[:, :, 0].astype("uint8")) # Convert numpy image to PIL, shape is (height, width, 4). 4 being RGBA. Use 0 (red) as default since it doesn't matter because of grayscaling later on.
            image = self.process_image(pil_image)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(image)
                pred = outputs.argmax(dim=1).item()
                mapped_pred = self.mapping[pred]

            st.success(f"Predicted KMNIST Character: {mapped_pred}")


if __name__ == "__main__":
    app = KMNISTStreamlitApp()
    app.main()