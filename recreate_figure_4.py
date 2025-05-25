#%%
from pathlib import Path
import tqdm

import numpy as np
import pandas as pd
import pypulseq as pp
import matplotlib.pyplot as plt
import colorcet as cc
from visualize import plot_mtr_asym, calculate_mtr_asymmetry_df

#%%
# load the scanner measurement data
file_path = "./pulseq_sequences/fig4_in_vitro/fully_sampled_comparison_multi_concentrations_new.xlsx"
df = pd.read_excel(file_path)
color_dict_comparisons = {"conventional 5mM": "#8c3bff", "conventional 20mM": "#018700", "conventional 50mM": "#d60000",
                          "mixed-ss 5mM": "#8c3bff", "mixed-ss 20mM": "#018700", "mixed-ss 50mM": "#d60000",
                          "pure-ss 5mM": "#8c3bff", "pure-ss 20mM": "#018700", "pure-ss 50mM": "#d60000"}
#%%
df_conv = df[["x data", "conventional 5mM", "conventional 20mM", "conventional 50mM"]]
df_mixed = df[["x data", "mixed-ss 5mM", "mixed-ss 20mM", "mixed-ss 50mM"]]
df_pure = df[["x data", "pure-ss 5mM", "pure-ss 20mM", "pure-ss 50mM"]]
#%%
fig, ax = plot_mtr_asym(df_conv, color_dict_comparisons)
# %%
fig, ax = plot_mtr_asym(df_mixed, color_dict_comparisons)
# %%
fig, ax = plot_mtr_asym(df_pure, color_dict_comparisons)
# %%
# aplly MTR asymmetry to the data
df_mtr_asym = pd.DataFrame()
for i in range(len(df.columns)-1):
    label = df.columns[i+1]
    offsets, mtr_asym = calculate_mtr_asymmetry_df(df["x data"], df.iloc[:,i+1])
    df_mtr_asym["x data"] = offsets    
    df_mtr_asym[label] = mtr_asym
    
    #df_mtr_asym.iloc[:,i+1] = mtr_asym
#%%
# Filter rows where the first column (offsets) is 6 or 5
filtered_df = df_mtr_asym[df_mtr_asym["x data"].isin([6, 5])]

# Calculate the mean for each of the 18 measurement columns
means = filtered_df.iloc[:, 1:].mean()

# Print the results
print(means)
# %%
# Create a new DataFrame with the desired structure
new_df = pd.DataFrame({
    "concentration": ["1mM", "2mM", "5mM","10mM", "20mM", "50mM"],
    "conventional": [
        means["conventional 1mM"],
        means["conventional 2mM"],
        means["conventional 5mM"],
        means["conventional 10mM"],
        means["conventional 20mM"],
        means["conventional 50mM"]
    ],
    "mixed-ss": [
        means["mixed-ss 1mM"],
        means["mixed-ss 2mM"],
        means["mixed-ss 5mM"],
        means["mixed-ss 10mM"],
        means["mixed-ss 20mM"],
        means["mixed-ss 50mM"]
    ],
    "pure-ss": [
        means["pure-ss 1mM"],
        means["pure-ss 2mM"],
        means["pure-ss 5mM"],
        means["pure-ss 10mM"],
        means["pure-ss 20mM"],
        means["pure-ss 50mM"]
    ]
})

# Print the new DataFrame
print(new_df)
# %%
## fit linear model to the data
import numpy as np

# Fit a line to each group
conventional_fit = np.polyfit(new_df["concentration"], new_df["conventional"], 1)
mixed_ss_fit = np.polyfit(new_df["concentration"], new_df["mixed-ss"], 1)
pure_ss_fit = np.polyfit(new_df["concentration"], new_df["pure-ss"], 1)

# Generate the fitted lines
conventional_line = np.polyval(conventional_fit, new_df["concentration"])
mixed_ss_line = np.polyval(mixed_ss_fit, new_df["concentration"])
pure_ss_line = np.polyval(pure_ss_fit, new_df["concentration"])

# Plot the scatter plot with fitted lines
fig, ax = plt.subplots()

# Scatter plot for each column
ax.scatter(new_df["concentration"], new_df["conventional"], label="Conventional", color="blue")
ax.scatter(new_df["concentration"], new_df["mixed-ss"], label="Mixed-SS", color="green")
ax.scatter(new_df["concentration"], new_df["pure-ss"], label="Pure-SS", color="red")

# Add fitted lines
ax.plot(new_df["concentration"], conventional_line, label="Conventional Fit", color="blue", linestyle="--")
ax.plot(new_df["concentration"], mixed_ss_line, label="Mixed-SS Fit", color="green", linestyle="--")
ax.plot(new_df["concentration"], pure_ss_line, label="Pure-SS Fit", color="red", linestyle="--")

# Add labels and legend
ax.set_xlabel("Concentration (mM)")
ax.set_ylabel("Mean Values")
ax.set_title("Scatter Plot with Fitted Lines")
ax.legend()

# Show the plot
plt.show()
# %%
# Extract and print the parameters
print("Conventional Fit: Slope =", conventional_fit[0], ", Intercept =", conventional_fit[1])
print("Mixed-SS Fit: Slope =", mixed_ss_fit[0], ", Intercept =", mixed_ss_fit[1])
print("Pure-SS Fit: Slope =", pure_ss_fit[0], ", Intercept =", pure_ss_fit[1])
# %%
## run ANCOVA test
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Combine the data into a single DataFrame
ancova_df = pd.DataFrame({
    "concentration": pd.concat([new_df["concentration"]] * 3, ignore_index=True),
    "mean_values": pd.concat([
        new_df["conventional"],
        new_df["mixed-ss"],
        new_df["pure-ss"]
    ], ignore_index=True),
    "group": ["conventional"] * len(new_df) + ["mixed-ss"] * len(new_df) + ["pure-ss"] * len(new_df)
})

# Fit the ANCOVA model
model = ols("mean_values ~ concentration * group", data=ancova_df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
# Extract p-value for interaction effect
interaction_p = anova_results.loc["concentration:group", "PR(>F)"]
interaction_text = f"ANCOVA Interaction: p = {interaction_p:.3f}" if interaction_p > 0.001 else "ANCOVA Interaction: p < 0.001"
# Print the summary of the ANCOVA test
print(model.summary())
# %%
# Generate the fitted lines
conventional_line = np.polyval(conventional_fit, new_df["concentration"])
mixed_ss_line = np.polyval(mixed_ss_fit, new_df["concentration"])
pure_ss_line = np.polyval(pure_ss_fit, new_df["concentration"])

# Plot the scatter plot with fitted lines
fig, ax = plt.subplots()

# Scatter plot for each column with different symbols and colors
ax.scatter(new_df["concentration"], new_df["conventional"], label="Conventional", color="green", marker="o")
ax.scatter(new_df["concentration"], new_df["mixed-ss"], label="Mixed-SS", color="purple", marker="s")
ax.scatter(new_df["concentration"], new_df["pure-ss"], label="Pure-SS", color="darkred", marker="^")

# Add fitted lines
ax.plot(new_df["concentration"], conventional_line, label="Conventional Fit", color="green", linestyle="--")
ax.plot(new_df["concentration"], mixed_ss_line, label="Mixed-SS Fit", color="purple", linestyle="--")
ax.plot(new_df["concentration"], pure_ss_line, label="Pure-SS Fit", color="darkred", linestyle="--")

# Add labels and legend
ax.set_xlabel("Concentration (mM)")
ax.set_ylabel("Mean Values")
ax.set_title("Scatter Plot with Fitted Lines and ANCOVA Results")
ax.legend()

# Overlay ANCOVA summary statistics
interaction_p = anova_results.loc["concentration:group", "PR(>F)"]
interaction_text = f"ANCOVA Interaction: p = {interaction_p:.3f}" if interaction_p > 0.001 else "ANCOVA Interaction: p < 0.001"
ax.text(0.05, 0.95, interaction_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

# Show the plot
plt.show()
# %%
