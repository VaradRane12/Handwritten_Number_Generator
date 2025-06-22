import streamlit as st
import torch
import matplotlib.pyplot as plt
from torch.nn import nn
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

# CVAE model
class CVAE(nn.Module):
    def __init__(self, latent_dim=45, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        input_dim = 784 + num_classes  # input image + label

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mu = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim + num_classes, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x, y):
        x = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3_mu(h), self.fc3_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z = torch.cat([z, y], dim=1)
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

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
