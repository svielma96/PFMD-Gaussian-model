import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# Path to your files
file_path = "/Users/svielma/Library/CloudStorage/OneDrive-ITG/ITG/2.EXPERIMENTS/CHAPTER 4 - Male and female swarm interplay/2024 ANALYSIS_NEW/"  # Replace with your folder path
all_files = glob.glob(file_path + "*.csv")  # Matches all CSV files

# Load each file into a list of DataFrames
dataframes = [pd.read_csv(f) for f in all_files]

# Time correction 
correction_dict = {
    'group_f200_01': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f200_02': pd.Timedelta(hours=16, minutes=0, seconds=0),
    'group_f200_03': pd.Timedelta(hours=16, minutes=0, seconds=0),
    'group_f150_01': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f150_02': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f150_03': pd.Timedelta(hours=16, minutes=0, seconds=0),
    'group_f100_01': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f100_02': pd.Timedelta(hours=16, minutes=0, seconds=0),
    'group_f100_03': pd.Timedelta(hours=16, minutes=0, seconds=0),
    'group_f50_01': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f50_02': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f50_03': pd.Timedelta(hours=16, minutes=0, seconds=0),
    'group_f0_01': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f0_02': pd.Timedelta(hours=17, minutes=0, seconds=0),
    'group_f0_03': pd.Timedelta(hours=16, minutes=0, seconds=0)
}

# Add group and file identifiers (if needed)
for i, df in enumerate(dataframes):
    # Extract the replicate name from the file
    filename = all_files[i].split("/")[-1].split(".")[0]  
    
    # Apply the time correction using the dictionary
    if 'datetime' in df.columns and filename in correction_dict:
        # Convert 'time' column to datetime if not already
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Apply the corresponding time correction
        df['datetime'] = df['datetime'] + correction_dict[filename]
    
    # Add group_id and file_id for organizational purposes
    df['group_id'] = filename.split("_")[1]  
    df['file_id'] = filename  
    

import pandas as pd

# Combine all dataframes
combined_data = pd.concat(dataframes, ignore_index=True)

# Define speed range for filtering
speed_min = 0.55  # Minimum speed threshold in m/s
speed_max = 5     # Maximum speed threshold in m/s

# Apply speed filtering
combined_data = combined_data[(combined_data["speed_mps"] >= speed_min) & 
                              (combined_data["speed_mps"] <= speed_max)]

# Filter based on x and z position ranges
combined_data = combined_data[(combined_data['x_cm'] >= -100) & (combined_data['x_cm'] <= 100) & 
                              (combined_data['z_cm'] >= 225) & (combined_data['z_cm'] <= 425)]

# Extract only the time from datetime column
combined_data['time_only'] = combined_data['datetime'].dt.time

# Filter data to include times between 18:04:00 and 18:24:00
time_filter = (combined_data['time_only'] >= pd.to_datetime('18:04:00').time()) & \
              (combined_data['time_only'] <= pd.to_datetime('18:24:00').time())

combined_data_filtered = combined_data[time_filter]

# --- Additional Step: Filter tracks based on total duration ---
# Convert datetime to seconds since midnight (time_only is HH:MM:SS)
combined_data_filtered['time_seconds'] = combined_data_filtered['datetime'].dt.hour * 3600 + \
                                         combined_data_filtered['datetime'].dt.minute * 60 + \
                                         combined_data_filtered['datetime'].dt.second

# Calculate flight duration for each track
track_durations = combined_data_filtered.groupby("id")["time_seconds"].agg(lambda x: x.max() - x.min())

# Keep only tracks with duration > 14 seconds
valid_tracks = track_durations[track_durations > 25].index

# Filter the main dataset
combined_data_filtered = combined_data_filtered[combined_data_filtered["id"].isin(valid_tracks)]

# Display minimum y_cm after filtering
min_y_cm = combined_data_filtered["y_cm"].min()
min_y_cm


combined_data_filtered.y_cm = combined_data_filtered.y_cm + 95
combined_data_filtered['z_cm'] = combined_data_filtered.z_cm - 325

# Ensure your dataset is loaded into `combined_data_filtered`

# Plot density heatmaps for each group
unique_groups = combined_data_filtered['group_id'].unique()

# Sort the groups in ascending order (f0 to f200)
sorted_groups = sorted(unique_groups, key=lambda x: int(x[1:]))
from scipy.stats import chi2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import seaborn as sns

####### GMM MODEL ########3

# Function to plot confidence ellipsoid
def plot_confidence_ellipsoid(ax, mean, cov, confidence=0.90, color='tomato', alpha=0):
    """
    Plots a confidence ellipsoid based on the covariance matrix.
    """
    # Compute chi-square critical value
    chi2_val = chi2.ppf(confidence, df=3)

    # Eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Compute the radii of the ellipsoid (scaled by chi2_val)
    radii = np.sqrt(eigenvalues * chi2_val)

    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale and rotate sphere to match the covariance
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            point = radii * point
            point = np.dot(eigenvectors, point) + mean
            x[i, j], y[i, j], z[i, j] = point

    # Plot the ellipsoid
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='k', linewidth=0.5)

# Store volume and mean speed data
volume_speed_data = []

# Set up the 3D plot layout
fig, axes = plt.subplots(3, 2, figsize=(12, 18), subplot_kw={'projection': '3d'}, dpi=300)
axes = axes.flatten()  # Flatten axes array for easier iteration

for i, group in enumerate(sorted_groups):
    # Extract data for the current group
    group_data = combined_data_filtered[combined_data_filtered['group_id'] == group]

    # Get unique file IDs (replicates)
    unique_replicates = group_data['file_id'].unique()

    for replicate in unique_replicates:
        replicate_data = group_data[group_data['file_id'] == replicate]

        # Ensure enough points exist for GMM
        if len(replicate_data) < 5:
            continue

        # Extract x, y, z coordinates
        x = replicate_data['x_cm'].values
        y = replicate_data['z_cm'].values
        z = replicate_data['y_cm'].values
        points = np.vstack((x, y, z)).T

        # Fit GMM model
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm.fit(points)

        # Extract mean and covariance matrix
        mean = gmm.means_[0]
        cov = gmm.covariances_[0]

        # Compute chi-square critical value for the desired confidence level
        chi2_val = chi2.ppf(0.90, df=3)

        # Compute eigenvalues (axes of the ellipsoid)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        radii = np.sqrt(eigenvalues) / np.sqrt(chi2_val)

        # Compute volume of the ellipsoid
        volume = (4/3) * np.pi * radii[0] * radii[1] * radii[2]  # Compute volume

        # Determine which points are inside the ellipsoid
        diff = points - mean
        mahalanobis_dist = np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)
        inside_ellipsoid = mahalanobis_dist <= chi2_val

        # Calculate mean speed of points inside the ellipsoid
        mean_speed_inside = replicate_data.loc[inside_ellipsoid, 'speed_mps'].mean()

        # Store data
        volume_speed_data.append({
            'Group': group,
            'Replicate': replicate,
            'Volume_cm3': volume,
            'Mean_Speed_mps': mean_speed_inside
        })

        # Select subplot
        ax = axes[i]
        ax.scatter(x, y, z, color="blue", alpha=0.5, s=1)

        # Plot GMM confidence ellipsoid
        plot_confidence_ellipsoid(ax, mean, cov, confidence=0.90)

        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Z (cm)')
        ax.set_zlabel('Y (cm)')
        ax.view_init(elev=20, azim=300)  # Adjust viewing angle
        ax.grid(False)  # Turn off the grid
        ax.set_box_aspect([1, 1, 2])  # Set proportional aspect ratio
        ax.set_title(f'Group {group} (GMM Ellipsoid)')
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 300)
        ax.set_xlim(-100, 100)

        # Adjust step size of the ticks
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.zaxis.set_major_locator(MultipleLocator(50))

# Adjust layout and show 3D plot
plt.tight_layout()
plt.show()

# Convert to DataFrame
volume_speed_df = pd.DataFrame(volume_speed_data)

# Plot the relationship between Volume and Mean Speed
# Plot the data and smoothed fit
plt.figure(figsize=(4, 4), dpi=500)
plt.gca().set_facecolor('whitesmoke')  # Change this to any color you prefer
sns.scatterplot(data=volume_speed_df, x='Volume_cm3', y='Mean_Speed_mps', hue='Group', palette='viridis', s=100)
plt.xlabel('Volume of GMM Ellipsoid (cm³)')
plt.ylabel('Mean Speed within Ellipsoid (m/s)')
plt.title('Relationship between Ellipsoid Volume and Mean Speed')
plt.legend(title='Group')
plt.grid(True)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline

# Convert volume data to DataFrame
volume_speed_df = pd.DataFrame(volume_speed_df)

# Define different fitting functions
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def logistic_model(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Extract volume and speed data
x_data = volume_speed_df["Volume_cm3"]
y_data = volume_speed_df["Mean_Speed_mps"]

# Ensure x_data is sorted for spline fitting
sorted_indices = np.argsort(x_data)
x_data_sorted = x_data[sorted_indices]
y_data_sorted = y_data[sorted_indices]

# Fit models
popt_linear, _ = curve_fit(linear_model, x_data, y_data)
popt_quadratic, _ = curve_fit(quadratic_model, x_data, y_data)
popt_cubic, _ = curve_fit(cubic_model, x_data, y_data)
popt_logistic, _ = curve_fit(logistic_model, x_data, y_data, p0=[1, 0.01, np.median(x_data)])

# Fit spline interpolation
spline = UnivariateSpline(x_data_sorted, y_data_sorted, s=1)


# Generate predicted values
x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit_linear = linear_model(x_fit, *popt_linear)
y_fit_quadratic = quadratic_model(x_fit, *popt_quadratic)
y_fit_cubic = cubic_model(x_fit, *popt_cubic)
y_fit_logistic = logistic_model(x_fit, *popt_logistic)
y_fit_spline = spline(x_fit)

# Compute R² values
def r2_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2_linear = r2_manual(y_data, linear_model(x_data, *popt_linear))
r2_quadratic = r2_manual(y_data, quadratic_model(x_data, *popt_quadratic))
r2_cubic = r2_manual(y_data, cubic_model(x_data, *popt_cubic))
r2_logistic = r2_manual(y_data, logistic_model(x_data, *popt_logistic))
r2_spline = r2_manual(y_data_sorted, spline(x_data_sorted))

# Create a colorblind-friendly colormap
colors = ["#88CCEE", "#CC6677", "#117733"]
cmap = LinearSegmentedColormap.from_list("muted_colorblind_friendly", colors, N=256)


# Plot volume vs. speed with different fits
plt.figure(figsize=(4, 4), dpi=500)
plt.gca().set_facecolor('whitesmoke')  # Change this to any color you prefer
sns.scatterplot(data=volume_speed_df, x=volume_speed_df['Volume_cm3'], y='Mean_Speed_mps', palette='viridis', s=100)
plt.plot(x_fit, y_fit_spline, 'tomato', label=f"Spline Fit (R²={r2_spline:.3f})",linestyle='dashed')

plt.xlabel("Volume of GMM Ellipsoid (cm³)")
plt.ylabel("Mean Speed within Ellipsoid (m/s)")
plt.ylim(0.5, 1)
plt.legend()
plt.grid()
plt.show()


# Create KDE plot
plt.figure(figsize=(2, 2),dpi=500,facecolor='whitesmoke')
plt.gca().set_facecolor('whitesmoke')  # Change this to any color you prefer
sns.kdeplot(volume_speed_df['Volume_cm3'], fill=True, color="darkorange", alpha=0.5)
# Labels and title
plt.xlabel("Volume (cm3)")
plt.ylabel("Density")
plt.legend(loc='upper left', bbox_to_anchor=(0, 1),fontsize=7)  # Adjust b

# Create KDE plot
plt.figure(figsize=(2, 2),dpi=500,facecolor='whitesmoke')
plt.gca().set_facecolor('whitesmoke')  # Change this to any color you prefer
sns.kdeplot(volume_speed_df['Mean_Speed_mps'], fill=True, color="teal", alpha=0.5)
plt.xlabel("Mean Speed (m/s)")
plt.ylabel("Density")
plt.legend(loc='upper left', bbox_to_anchor=(0, 1),fontsize=7)  # Adjust b

# Residual Plot
plt.figure(figsize=(4, 4), dpi=500)
plt.gca().set_facecolor('whitesmoke')
residuals = y_data - cubic_model(x_data, *popt_cubic)
sns.scatterplot(x=x_data, y=residuals, color='purple', alpha=0.7)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("Volume of GMM Ellipsoid (cm³)")
plt.ylabel("Residuals (Observed - Predicted)")
plt.title("Residuals of Cubic Fit")
plt.grid()
plt.show()


# Compare volume distributions across groups
# Residual Plot
plt.figure(figsize=(4, 4), dpi=500)
plt.gca().set_facecolor('whitesmoke')
sns.boxplot(data=volume_speed_df, x='Group', y='Volume_cm3', color='dodgerblue')
plt.xlabel("Group")
plt.ylabel("Volume of GMM Ellipsoid (cm³)")
plt.title("Comparison of Swarm Volumes Across Groups")
plt.grid()
plt.show()

####### ELLIPSE PLOT ########3

import seaborn as sns

# Function to plot confidence ellipsoid
def plot_confidence_ellipsoid(ax, mean, cov, confidence=0.90, color='tomato', alpha=0.5):
    """
    Plots a confidence ellipsoid based on the covariance matrix.
    """
    # Compute chi-square critical value
    chi2_val = chi2.ppf(confidence, df=3)

    # Eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Compute the radii of the ellipsoid (scaled by chi2_val)
    radii = np.sqrt(eigenvalues * chi2_val)

    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale and rotate sphere to match the covariance
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            point = radii * point
            point = np.dot(eigenvectors, point) + mean
            x[i, j], y[i, j], z[i, j] = point

    # Plot the ellipsoid
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='k', linewidth=0.5)


# Set up the 3D plot layout
fig, axes = plt.subplots(3, 2, figsize=(12, 12), subplot_kw={'projection': '3d'}, dpi=300)
axes = axes.flatten()  # Flatten axes array for easier iteration

for i, group in enumerate(sorted_groups):
    # Extract data for the current group
    group_data = combined_data_filtered[combined_data_filtered['group_id'] == group]

    # Get unique track IDs
    track_ids = group_data['id'].unique()
    num_tracks = len(track_ids)

    # Assign unique colors to each track
    unique_colors = sns.color_palette("husl", num_tracks)
    track_color_dict = dict(zip(track_ids, unique_colors))

    # Ensure there are enough points for GMM
    if len(group_data) < 5:
        continue

    # Extract x, y, z coordinates for GMM
    x = group_data['x_cm'].values
    y = group_data['z_cm'].values
    z = group_data['y_cm'].values
    points = np.vstack((x, y, z)).T

    # Fit a single-component GMM (assuming a single swarm cluster)
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(points)

    # Extract mean and covariance matrix
    mean = gmm.means_[0]
    cov = gmm.covariances_[0]

    # Select subplot
    ax = axes[i]

    # Plot each track in a unique color
    for track_id in track_ids:
        track_data = group_data[group_data['id'] == track_id]
        ax.plot(track_data['x_cm'], track_data['z_cm'], track_data['y_cm'],
                color=track_color_dict[track_id], alpha=0.7, linewidth=0.8)

    # Plot GMM confidence ellipsoid
    plot_confidence_ellipsoid(ax, mean, cov, confidence=0.90)

    # Labels and formatting
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Z (cm)')
    ax.set_zlabel('Y (cm)')
    ax.view_init(elev=20, azim=300)
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
        # Adjust title position using text2D
    ax.text2D(0.5, 0.9, f'Group {group} (Tracks: {num_tracks})', transform=ax.transAxes,
              ha='center', fontsize=12)

    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 200)
    ax.set_xlim(-100, 100)

    # Adjust step size of the ticks
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.zaxis.set_major_locator(MultipleLocator(50))

# Adjust layout and show plot
plt.tight_layout()
plt.show()


from scipy.optimize import curve_fit


volume_df = pd.DataFrame(volume_speed_df)

# Aggregate volume data per group (mean and std)
volume_summary = volume_df.groupby("Group")["Volume_cm3"].agg(["mean", "std"]).reset_index()

# Convert group labels to numerical values (e.g., f0 -> 0, f50 -> 50, etc.)
volume_summary["Group_Num"] = volume_summary["Group"].apply(lambda x: int(x[1:]))

# Define different fitting functions
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Fit models
popt_linear, _ = curve_fit(linear_model, volume_summary["Group_Num"], volume_summary["mean"])
popt_quadratic, _ = curve_fit(quadratic_model, volume_summary["Group_Num"], volume_summary["mean"])
popt_exponential, _ = curve_fit(exponential_model, volume_summary["Group_Num"], volume_summary["mean"], p0=[1, 0.01])

# Generate predicted values
x_fit = np.linspace(volume_summary["Group_Num"].min(), volume_summary["Group_Num"].max(), 100)
y_fit_linear = linear_model(x_fit, *popt_linear)
y_fit_quadratic = quadratic_model(x_fit, *popt_quadratic)
y_fit_exponential = exponential_model(x_fit, *popt_exponential)

from sklearn.metrics import r2_score

# Compute R² scores for each fit
r2_linear = r2_score(volume_summary["mean"], linear_model(volume_summary["Group_Num"], *popt_linear))
r2_quadratic = r2_score(volume_summary["mean"], quadratic_model(volume_summary["Group_Num"], *popt_quadratic))
r2_exponential = r2_score(volume_summary["mean"], exponential_model(volume_summary["Group_Num"], *popt_exponential))

# Plot volume trends with different fits and R² scores
# Plot the data and smoothed fit
plt.figure(figsize=(4, 4), dpi=500)
plt.gca().set_facecolor('whitesmoke')  # Change this to any color you prefer
plt.errorbar(volume_summary["Group_Num"], volume_summary["mean"], yerr=volume_summary["std"],
             fmt='o', label="Mean Volume ± Std Dev", capsize=5,c='dodgerblue')
plt.plot(x_fit, y_fit_quadratic, 'tomato', label=f"Quadratic Fit (R²={r2_quadratic:.3f})",linestyle='dashed')

plt.xlabel("Group (f-value)")
plt.ylabel("Volume of GMM Ellipsoid (m³)")
plt.legend()
plt.grid(False)
plt.show()

#### data export ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline

# Convert volume data to DataFrame
volume_speed_df
# Define different fitting functions
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def logistic_model(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Extract volume and speed data
x_data = np.array(volume_speed_df["Volume_cm3"])
y_data = np.array(volume_speed_df["Mean_Speed_mps"])

# Ensure x_data is sorted for spline fitting
sorted_indices = np.argsort(x_data)
x_data_sorted = x_data[sorted_indices]
y_data_sorted = y_data[sorted_indices]

# Fit models
popt_linear, _ = curve_fit(linear_model, x_data, y_data)
popt_quadratic, _ = curve_fit(quadratic_model, x_data, y_data)
popt_cubic, _ = curve_fit(cubic_model, x_data, y_data)
popt_logistic, _ = curve_fit(logistic_model, x_data, y_data, p0=[1, 0.01, np.median(x_data)])

# Fit spline interpolation
spline = UnivariateSpline(x_data_sorted, y_data_sorted, s=1)

# Generate predicted values
x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit_linear = linear_model(x_fit, *popt_linear)
y_fit_quadratic = quadratic_model(x_fit, *popt_quadratic)
y_fit_cubic = cubic_model(x_fit, *popt_cubic)
y_fit_logistic = logistic_model(x_fit, *popt_logistic)
y_fit_spline = spline(x_fit)

# Compute R² values
def r2_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2_linear = r2_manual(y_data, linear_model(x_data, *popt_linear))
r2_quadratic = r2_manual(y_data, quadratic_model(x_data, *popt_quadratic))
r2_cubic = r2_manual(y_data, cubic_model(x_data, *popt_cubic))
r2_logistic = r2_manual(y_data, logistic_model(x_data, *popt_logistic))
r2_spline = r2_manual(y_data_sorted, spline(x_data_sorted))

# Extract data points inside ellipsoids
def points_inside_ellipsoid(group_data, mean, cov, chi2_val):
    points = group_data[['x_cm', 'z_cm', 'y_cm']].values
    diff = points - mean
    mahalanobis_dist = np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)
    return group_data[mahalanobis_dist <= chi2_val]

inside_ellipsoid_data = []
for group in sorted_groups:
    group_data = combined_data_filtered[combined_data_filtered['group_id'] == group]
    unique_replicates = group_data['file_id'].unique()
    for replicate in unique_replicates:
        replicate_data = group_data[group_data['file_id'] == replicate]
        if len(replicate_data) < 5:
            continue
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm.fit(replicate_data[['x_cm', 'z_cm', 'y_cm']])
        mean = gmm.means_[0]
        cov = gmm.covariances_[0]
        chi2_val = chi2.ppf(0.90, df=3)
        inside_points = points_inside_ellipsoid(replicate_data, mean, cov, chi2_val)
        inside_ellipsoid_data.append(inside_points)

inside_ellipsoid_df = pd.concat(inside_ellipsoid_data, ignore_index=True)

# Save the inside ellipsoid data
inside_ellipsoid_df.to_csv("inside_ellipsoid_data.csv", index=False)
