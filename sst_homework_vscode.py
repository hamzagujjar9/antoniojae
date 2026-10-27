# %% [markdown]
# # SST local standard deviation (VS Code, Python)
# Run cell by cell with "Run Cell" (Shift+Enter) above each `# %%`, or run the whole file.
# Setup (once, in the VS Code terminal):
#     pip install numpy scipy matplotlib xarray netCDF4

# %% 0. Settings: EDIT THE FILE PATH
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import fftconvolve

FILE = Path("20251231120000-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.1.nc")  # <- put full path if needed
WINDOW = 21          # odd number of pixels
SIGMA = WINDOW / 4   # width of the Gaussian weights (pixels)
MIN_VALID = 0.5      # minimum fraction of valid weight in a window, else result = NaN

# %% 1. Open the NetCDF file and look inside
ds = xr.open_dataset(FILE)   # CF decoding: fill values become NaN automatically
print(ds)                    # dimensions, coordinates, variables, attributes
print(ds["analysed_sst"].attrs)

# %% 2. Extract analysed_sst as a 2D array (NaN = land / missing)
sst = ds["analysed_sst"]
if "time" in sst.dims:
    sst = sst.isel(time=0)

field = sst.values.astype(float)
if str(sst.attrs.get("units", "")).lower().startswith("kelvin"):
    field = field - 273.15   # convert to Celsius
print("shape:", field.shape)
print("valid fraction:", np.isfinite(field).mean())
print("min / max:", np.nanmin(field), np.nanmax(field))

lon = ds["lon"].values
lat = ds["lat"].values
extent = [lon.min(), lon.max(), lat.min(), lat.max()]

# %% 3. Plot the SST map
plt.figure(figsize=(11, 5))
plt.imshow(field, origin="lower", extent=extent, cmap="turbo", aspect="auto")
plt.colorbar(label="SST (°C)")
plt.title("analysed_sst")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.show()

# %% 4. Function: local standard deviation, ignoring NaNs
def weighted_local_std(data, kernel, min_valid=0.5):
    """
    Std in a window around every pixel using only valid (non-NaN) pixels.
    mean = sum(w*x)/sum(w);  var = sum(w*x^2)/sum(w) - mean^2
    Uniform kernel -> plain moving-window std. Gaussian kernel -> weighted std.
    """
    kernel = kernel / kernel.sum()
    valid = np.isfinite(data)
    x = np.where(valid, data - np.nanmean(data), 0.0)   # subtract mean for numerical precision
    v = valid.astype(float)

    W = fftconvolve(v, kernel, mode="same")        # valid weight inside each window
    m1 = fftconvolve(x, kernel, mode="same")
    m2 = fftconvolve(x * x, kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = m1 / W
        var = m2 / W - mean ** 2
    var[var < 0] = 0.0
    std = np.sqrt(var)
    std[W < min_valid] = np.nan    # window mostly land/missing
    std[~valid] = np.nan           # no value on land/missing centers
    return std

# %% 5. Part 2: uniform 21x21 window
uniform_kernel = np.ones((WINDOW, WINDOW))
std_uniform = weighted_local_std(field, uniform_kernel, MIN_VALID)

plt.figure(figsize=(11, 5))
plt.imshow(std_uniform, origin="lower", extent=extent, cmap="viridis",
           vmax=np.nanpercentile(std_uniform, 99), aspect="auto")
plt.colorbar(label="local std (°C)")
plt.title(f"Local standard deviation, {WINDOW}x{WINDOW} uniform window")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.show()

# %% 6. Part 3: Gaussian weights decaying with distance from the center
r = WINDOW // 2
yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
gauss_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIGMA ** 2))

std_gauss = weighted_local_std(field, gauss_kernel, MIN_VALID)

plt.figure(figsize=(11, 5))
plt.imshow(std_gauss, origin="lower", extent=extent, cmap="viridis",
           vmax=np.nanpercentile(std_uniform, 99), aspect="auto")
plt.colorbar(label="local std (°C)")
plt.title(f"Local standard deviation, {WINDOW}x{WINDOW} Gaussian-weighted (sigma={SIGMA:.1f})")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.show()

# %% 7. Quick check: compare with a brute-force calculation at one pixel
i, j = field.shape[0] // 2, field.shape[1] // 2
win = field[i - r:i + r + 1, j - r:j + r + 1]
print("brute force:", np.nanstd(win), " function:", std_uniform[i, j])


