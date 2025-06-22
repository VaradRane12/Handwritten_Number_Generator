import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim=45):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x, y):
        h1 = torch.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = torch.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

device = torch.device("cpu")  # Streamlit Cloud doesn't support CUDA
model = CVAE().to(device)
model.load_state_dict(torch.load("models/cvae_mnist.pth", map_location=device))
model.eval()

@torch.no_grad()
def generate_digit_images(digit, n=5):
    y = one_hot(torch.tensor([digit]*n)).to(device)
    z = torch.randn(n, 20).to(device)
    samples = model.decode(z, y).cpu()
    return samples.view(-1, 28, 28)

def plot_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 2))
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
    return fig

st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("Handwritten Digit Generator")
st.markdown("Select a digit (0–9) and generate 5 samples using a trained CVAE.")

digit = st.selectbox("Choose a digit:", list(range(10)))
if st.button("Generate"):
    imgs = generate_digit_images(digit, n=5)
    fig = plot_images(imgs)
    st.pyplot(fig)
