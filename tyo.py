import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt

st.title("Kävely")

# Luetaan kiityvyysdata
df = pd.read_csv("https://raw.githubusercontent.com/t3vajo05/-Fysiikan-loppuprojekti/refs/heads/main/My%20Experiment/Linear%20Acceleration.csv")

t = df["Time (s)"].values
x = df["Linear Acceleration x (m/s^2)"].values
y = df["Linear Acceleration y (m/s^2)"].values
z = df["Linear Acceleration z (m/s^2)"].values

# Kiihtyvyysdatan suodatus ja piirto
T_tot = t.max()
n = len(t)
fs = n / T_tot
nyq = fs / 2

def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

filtered_data = butter_lowpass_filter(z, cutoff=1 / 0.7, nyq=nyq, order=3)

plt.figure(figsize=(12, 4))
plt.plot(t, z, label="Alkuperäinen data")
plt.plot(t, filtered_data, label="Suodatettu data", linewidth=3)
plt.xlabel("Aika (s)")
plt.ylabel("Kiihtyvyys z (m/s^2)")
plt.title("Askelmittaus")
plt.legend()

st.pyplot(plt)

# Askelmäärä suodatuksesta
askeleet = 0
for i in range(1, len(filtered_data)):
    if filtered_data[i - 1] < 0 and filtered_data[i] >= 0:
        askeleet += 1

st.write("Askelmäärä suodatuksen perusteella:", askeleet)

# Fourier-muunnos
dt = t[1] - t[0]
N = len(z)

X = np.fft.fft(z, N)
psd = X * np.conj(X) / N
freq = np.fft.fftfreq(N, dt)

L = np.arange(1, np.floor(N / 2), dtype=int)

plt.figure(figsize=(12, 6))
plt.plot(freq[L], psd[L].real)
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Teho")
plt.title("Tehospektri")
plt.xlim(0, 15)

st.pyplot(plt)

dominant_freq = freq[L][np.argmax(psd[L].real)]
askelmaara_fourier = int(dominant_freq * T_tot)

st.write("Askelmäärä Fourier-analyysin perusteella:", askelmaara_fourier)

# Luetaan GPS-data
gps = pd.read_csv("https://raw.githubusercontent.com/t3vajo05/-Fysiikan-loppuprojekti/refs/heads/main/My%20Experiment/Location.csv")

lat = np.deg2rad(gps["Latitude (°)"].values)
lon = np.deg2rad(gps["Longitude (°)"].values)

R = 6371000

dlat = np.diff(lat)
dlon = np.diff(lon)

a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distances = R * c
kokonaismatka = np.sum(distances)

st.write("Kokonaismatka (GPS-datasta):", round(kokonaismatka, 2), "m")

# Lasketaan askelpituus GPS-datasta
askelpituus = kokonaismatka / askeleet
st.write("Askelpituus:", round(askelpituus, 2), "m")

# Luodaan kartta
start_lat = gps["Latitude (°)"].mean()
start_lon = gps["Longitude (°)"].mean()

map = folium.Map(location=[start_lat, start_lon], zoom_start=14)
folium.PolyLine(
    gps[["Latitude (°)", "Longitude (°)"]],
    color="blue",
    weight=3
).add_to(map)

st_folium(map, width=900, height=650)
