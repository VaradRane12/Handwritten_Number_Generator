import streamlit as st
import torch
import matplotlib.pyplot as plt
from train_model import CVAE, one_hot  # Make sure this matches your training script

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 45
num_classes = 10

# Load model
model = CVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("models/cvae_mnist.pth", map_location=device))
model.eval()

# Digit generation function
def generate_digit(model, digit, n=5):
    model.eval()
    with torch.no_grad():
        y = one_hot(torch.tensor([digit]*n)).to(device)
        z = torch.randn(n, latent_dim).to(device)
        samples = model.decode(z, y).cpu()
        return samples.view(-1, 28, 28)

# Streamlit UI
st.title("🧠 Generate MNIST Digits with CVAE")
st.markdown("Select a digit (0–9) to generate 5 handwritten versions using your trained model.")

digit = st.selectbox("Choose a digit to generate:", list(range(10)))

if st.button("Generate"):
    images = generate_digit(model, digit, n=5)

    st.subheader(f"Generated Digit: {digit}")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
