import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.fc3 = nn.Linear(latent_dim + 10, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 784)

    def decode(self, z, y):
        x = torch.cat([z, y], dim=1)
        h3 = F.relu(self.fc3(x))
        h4 = F.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))


    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], 1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], 1)))
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
