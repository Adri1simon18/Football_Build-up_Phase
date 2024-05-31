import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.widgets import EllipseSelector

import seaborn as sns
from scipy.stats import chi2_contingency


""" Fonction calculant les coordonnées standards et principales de l'analyse de correspondance d'un tableau de contingence donné """
def calcul_coord(contingency_table):
    grand_total = contingency_table.sum().sum()

    row_masses = contingency_table.sum(axis=1) / grand_total
    col_masses = contingency_table.sum(axis=0) / grand_total

    
    corrspnd_mat = contingency_table / grand_total
    
    expc_freq = np.kron(row_masses.reshape(-1, 1),
                        col_masses.reshape(1, -1))
    centr_corrspnd_mat = corrspnd_mat - expc_freq

    
    chi_squared = \
        grand_total * ((centr_corrspnd_mat ** 2) / expc_freq).sum().sum()

    Dr_sqrt_inv = np.diag(1 / np.sqrt(row_masses))
    Dc_sqrt_inv = np.diag(1 / np.sqrt(col_masses))

    pearson_resd = Dr_sqrt_inv @ centr_corrspnd_mat @ Dc_sqrt_inv

    U, D_lamb, V_T = np.linalg.svd(pearson_resd, full_matrices=False)

    principal_inertias = D_lamb ** 2

    percent_explnd_var = (principal_inertias / principal_inertias.sum()) * 100

    D_lamb_mat = np.diag(D_lamb)

    princpl_coords_row = Dr_sqrt_inv @ U @  D_lamb_mat
    princpl_coords_col = Dc_sqrt_inv @ V_T.T @ D_lamb_mat.T

    std_coords_row = Dr_sqrt_inv @ U
    std_coords_col = Dc_sqrt_inv @ V_T.T

    pri_coord_col = [[sublist[0], sublist[1]] for sublist in princpl_coords_col]
    pri_coord_row = [[sublist[0], sublist[1]] for sublist in princpl_coords_row]

    std_coord_col = [[sublist[0], sublist[1]] for sublist in std_coords_col]
    std_coord_row = [[sublist[0], sublist[1]] for sublist in std_coords_row]

    return percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col



###              ###
### Type of pass ###
###              ###

contingency_table = np.array([[1702, 150, 645], 
                              [217, 4, 11], 
                              [528, 16, 29], 
                              [694, 34, 44], 
                              [560, 31, 95], 
                              [261, 16, 44]])


# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour type de passes :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Type of pass': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Type of pass': {
        'Ground': std_coord_col[0],
        'Low': std_coord_col[1],
        'High': std_coord_col[2],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Type of pass')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Type of pass': {
        'Ground': pri_coord_col[0],
        'Low': pri_coord_col[1],
        'High': pri_coord_col[2],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Type of pass': {
        'Ground': std_coord_col[0],
        'Low': std_coord_col[1],
        'High': std_coord_col[2],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Type of pass': {
        'Ground': pri_coord_col[0],
        'Low': pri_coord_col[1],
        'High': pri_coord_col[2],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "High" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            else :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###  Weak foot   ###
###              ###
contingency_table = np.array([[404, 1184, 459], 
                              [26, 143, 86], 
                              [49, 229, 113], 
                              [70, 257, 187], 
                              [77, 269, 158], 
                              [44, 138, 102]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour weak foot :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Weak foot': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Weak foot': {
        '<3/5': std_coord_col[0],
        '3/5': std_coord_col[1],
        '>3/5': std_coord_col[2],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Weak foot')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Weak foot': {
        '<3/5': std_coord_col[0],
        '3/5': std_coord_col[1],
        '>3/5': std_coord_col[2],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Weak foot': {
        '<3/5': std_coord_col[0],
        '3/5': std_coord_col[1],
        '>3/5': std_coord_col[2],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Weak foot': {
        '<3/5': std_coord_col[0],
        '3/5': std_coord_col[1],
        '>3/5': std_coord_col[2],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Exceptional" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "2/5" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "4/5" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###    Skills    ###
###              ###
contingency_table = np.array([[693, 813, 469, 68], 
                              [41, 133, 70, 11], 
                              [79, 160, 128, 18], 
                              [82, 259, 151, 15], 
                              [124, 228, 137, 10], 
                              [80, 109, 87, 8]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour skills :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Skills': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Skills': {
        '1/5': std_coord_col[0],
        '2/5': std_coord_col[1],
        '3/5': std_coord_col[2],
        '4/5': std_coord_col[3],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Skills')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Skills': {
        '1/5': pri_coord_col[0],
        '2/5': pri_coord_col[1],
        '3/5': pri_coord_col[2],
        '4/5': pri_coord_col[3],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Skills': {
        '1/5': std_coord_col[0],
        '2/5': std_coord_col[1],
        '3/5': std_coord_col[2],
        '4/5': std_coord_col[3],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Skills': {
        '1/5': pri_coord_col[0],
        '2/5': pri_coord_col[1],
        '3/5': pri_coord_col[2],
        '4/5': pri_coord_col[3],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "3/5" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "2/5" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###     BMI      ###
###              ###          
contingency_table = np.array([[233, 450, 893, 259, 164], 
                              [48, 70, 70, 29, 38], 
                              [72, 86, 169, 43, 16], 
                              [62, 129, 201, 65, 28], 
                              [58, 143, 142, 72, 68],
                              [68, 54, 76, 33, 47]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour BMI :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'BMI': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'BMI': {
        '21-22': std_coord_col[0],
        '22-23': std_coord_col[1],
        '23-24': std_coord_col[2],
        '24-25': std_coord_col[3],
        '25-26': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='BMI')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'BMI': {
        '21-22': pri_coord_col[0],
        '22-23': pri_coord_col[1],
        '23-24': pri_coord_col[2],
        '24-25': pri_coord_col[3],
        '25-26': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],   
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'BMI': {
        '21-22': std_coord_col[0],
        '22-23': std_coord_col[1],
        '23-24': std_coord_col[2],
        '24-25': std_coord_col[3],
        '25-26': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'BMI': {
        '21-22': pri_coord_col[0],
        '22-23': pri_coord_col[1],
        '23-24': pri_coord_col[2],
        '24-25': pri_coord_col[3],
        '25-26': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()




###              ###
###    Passing   ###
###              ###
contingency_table = np.array([[314, 564, 385, 54, 141], 
                              [26, 97, 62, 29, 2], 
                              [33, 121, 124, 32, 11], 
                              [76, 126, 184, 29, 7], 
                              [47, 157, 140, 20, 26], 
                              [15, 87, 81, 21, 22]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour Passing :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Passing': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Passing': {
        '50-60': std_coord_col[0],
        '60-70': std_coord_col[1],
        '70-80': std_coord_col[2],
        '80-90': std_coord_col[3],
        '90-100': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Passing')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Passing': {
        '50-60': pri_coord_col[0],
        '60-70': pri_coord_col[1],
        '70-80': pri_coord_col[2],
        '80-90': pri_coord_col[3],
        '90-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],   
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Passing': {
        '50-60': std_coord_col[0],
        '60-70': std_coord_col[1],
        '70-80': std_coord_col[2],
        '80-90': std_coord_col[3],
        '90-100': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Passing': {
        '50-60': pri_coord_col[0],
        '60-70': pri_coord_col[1],
        '70-80': pri_coord_col[2],
        '80-90': pri_coord_col[3],
        '90-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
### Positionning ###
###              ###
contingency_table = np.array([[1149, 436, 237, 181, 44], 
                              [130, 34, 50, 33, 8], 
                              [160, 82, 53, 78, 18], 
                              [233, 112, 80, 78, 11], 
                              [215, 104, 108, 65, 12], 
                              [132, 64, 41, 39, 8]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour positionning :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Positionning': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Positionning': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Positionning')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Positionning': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Positionning': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Positionning': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Very successful" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "Exceptional" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "60-70" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "<50" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###    Vision    ###
###              ###
contingency_table = np.array([[435, 620, 733, 148, 111], 
                              [31, 87, 79, 10, 48], 
                              [51, 115, 104, 64, 57], 
                              [85, 118, 159, 73, 79], 
                              [61, 126, 201, 58, 58], 
                              [24, 65, 138, 21, 36]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour vision :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Vision': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Vision': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Vision')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Vision': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Vision': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Vision': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "60-70" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "80-100" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "<50" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
### Long Passing ###
###              ###
contingency_table = np.array([[636, 256, 440, 533, 180], 
                              [36, 23, 52, 91, 53], 
                              [70, 31, 99, 125, 64], 
                              [66, 56, 86, 218, 88], 
                              [91, 70, 114, 149, 75], 
                              [58, 33, 55, 84, 54]])

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Long passing': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Long passing': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Long passing')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Long passing': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Long passing': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Long passing': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()




###              ###
###     Pace     ###
###              ###
contingency_table = np.array([[101, 184, 274, 411, 301, 776], 
                              [7, 51, 50, 56, 31, 60], 
                              [5, 67, 75, 67, 62, 115], 
                              [31, 84, 145, 95, 54, 105], 
                              [32, 60, 94, 100, 80, 138], 
                              [28, 16, 59, 67, 30, 84]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour pace :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Pace': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Pace': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Pace')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Pace': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],   
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Pace': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Pace': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Exceptional" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "Failed" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "60-70" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "<50" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "80-90" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "90-100" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='center', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
### Ball control ###
###              ###
contingency_table = np.array([[693, 134, 379, 620, 221], 
                              [41, 18, 34, 100, 62], 
                              [79, 12, 75, 135, 90], 
                              [82, 43, 89, 168, 132], 
                              [124, 12, 77, 182, 109], 
                              [80, 7, 31, 99, 67]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour ball control :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Ball control': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Ball control': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Ball control')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Ball control': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],   
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Ball control': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower right')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Ball control': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "60-70" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "50-60" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "80-100" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

"""


"""
###              ###
###Short passing ###
###              ###
contingency_table = np.array([[628, 103, 333, 723, 232, 28], 
                              [39, 2, 20, 110, 75, 9], 
                              [77, 5, 45, 155, 89, 20], 
                              [67, 26, 66, 200, 147, 8], 
                              [92, 23, 61, 207, 116, 5], 
                              [61, 13, 25, 104, 75, 6]])

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Short passing': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Short passing': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Short passing')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Short passing': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Short passing': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Short passing': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(10, 5))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.show()





###              ###
###  Work rate   ###
###              ###
## J'ai supp le work rate Medium / Low et Low / Low 
contingency_table = np.array([[217, 263, 8, 251, 1167, 91, 48], 
                              [32, 27, 0, 62, 117, 17, 0], 
                              [33, 66, 5, 85, 178, 17, 5], 
                              [59, 61, 3, 143, 206, 36, 6], 
                              [53, 71, 9, 67, 265, 27, 7], 
                              [44, 66, 0, 37, 133, 3, 1]])
# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour left back :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Work rate': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Work rate': {
        'H/H': std_coord_col[0],
        'H/M': std_coord_col[1],
        'H/L': std_coord_col[2],
        'M/H': std_coord_col[3],
        'M/M': std_coord_col[4],
        'L/H': std_coord_col[5],
        'L/M': std_coord_col[6],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Work rate')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Work rate': {
        'H/H': pri_coord_col[0],
        'H/M': pri_coord_col[1],
        'H/L': pri_coord_col[2],
        'M/H': pri_coord_col[3],
        'M/M': pri_coord_col[4],
        'L/H': pri_coord_col[5],
        'L/M': pri_coord_col[6],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Work rate': {
        'H/H': std_coord_col[0],
        'H/M': std_coord_col[1],
        'H/L': std_coord_col[2],
        'M/H': std_coord_col[3],
        'M/M': std_coord_col[4],
        'L/H': std_coord_col[5],
        'L/M': std_coord_col[6],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Work rate': {
        'H/H': pri_coord_col[0],
        'H/M': pri_coord_col[1],
        'H/L': pri_coord_col[2],
        'M/H': pri_coord_col[3],
        'M/M': pri_coord_col[4],
        'L/H': pri_coord_col[5],
        'L/M': pri_coord_col[6],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Exceptional" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "Failed" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "L/H" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "L/M" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "H/M" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "H/L" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "M/M" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###    Physic    ###
###              ###
contingency_table = np.array([[87, 762, 501, 695], 
                              [6, 129, 79, 41], 
                              [22, 167, 117, 81], 
                              [16, 246, 163, 82], 
                              [20, 227, 128, 129], 
                              [6, 133, 65, 80]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour physic :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Physic': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Physic': {
        '60-70': std_coord_col[0],
        '70-80': std_coord_col[1],
        '80-90': std_coord_col[2],
        '90-100': std_coord_col[3],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Physic')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Physic': {
        '60-70': pri_coord_col[0],
        '70-80': pri_coord_col[1],
        '80-90': pri_coord_col[2],
        '90-100': pri_coord_col[3],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Physic': {
        '60-70': std_coord_col[0],
        '70-80': std_coord_col[1],
        '80-90': std_coord_col[2],
        '90-100': std_coord_col[3],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Physic': {
        '60-70': pri_coord_col[0],
        '70-80': pri_coord_col[1],
        '80-90': pri_coord_col[2],
        '90-100': pri_coord_col[3],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Very successful" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "60-70" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "80-90" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()
"""


"""

###              ###
### Acceleration ###
###              ###
contingency_table = np.array([[397, 563, 381, 456, 199, 51], 
                              [41, 81, 45, 65, 17, 6], 
                              [46, 134, 69, 80, 31, 31], 
                              [64, 183, 128, 79, 49, 11], 
                              [91, 133, 126, 86, 59, 9], 
                              [69, 68, 65, 70, 12, 0]])

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Acceleration': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Acceleration': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Acceleration')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Acceleration': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],   
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Acceleration': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Acceleration': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###   Strength   ###
###              ###
contingency_table = np.array([[85, 128, 374, 894, 518, 48], 
                              [2, 11, 49, 108, 80, 5], 
                              [11, 31, 73, 156, 112, 8], 
                              [5, 14, 94, 179, 214, 8], 
                              [2, 23, 101, 228, 134, 16], 
                              [0, 3, 71, 131, 78, 1]])

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Strength': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Strength': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Strength')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Strength': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Strength': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-90': std_coord_col[4],
        '90-100': std_coord_col[5],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Strength': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-90': pri_coord_col[4],
        '90-100': pri_coord_col[5],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###   Agility    ###
###              ###
contingency_table = np.array([[441, 523, 679, 342, 62], 
                              [24, 44, 134, 47, 6], 
                              [44, 76, 167, 85, 19], 
                              [101, 111, 210, 67, 25], 
                              [94, 109, 216, 63, 22], 
                              [63, 65, 96, 52, 8]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour agility :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Agility': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Agility': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Agility')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Agility': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Agility': {
        '<50': std_coord_col[0],
        '50-60': std_coord_col[1],
        '60-70': std_coord_col[2],
        '70-80': std_coord_col[3],
        '80-100': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Agility': {
        '<50': pri_coord_col[0],
        '50-60': pri_coord_col[1],
        '60-70': pri_coord_col[2],
        '70-80': pri_coord_col[3],
        '80-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "60-70" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "80-100" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "50-60" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower right')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
### Composition  ###
###              ###
contingency_table = np.array([[385, 303, 251, 122, 137, 122, 171], 
                              [13, 19, 3, 3, 6, 8, 7], 
                              [30, 22, 23, 10, 17, 19, 16], 
                              [44, 32, 24, 11, 25, 20, 22], 
                              [59, 41, 33, 15, 10, 8, 19], 
                              [24, 19, 11, 9, 3, 3, 9]] ) 

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour compo :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Tactical system': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Tactical system': {
        '4231': std_coord_col[0],
        '433': std_coord_col[1],
        '442': std_coord_col[2],
        '3421': std_coord_col[3],
        '352': std_coord_col[4],
        '4141': std_coord_col[5],
        '343': std_coord_col[6],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Tactical system')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Tactical system': {
        '4231': pri_coord_col[0],
        '433': pri_coord_col[1],
        '442': pri_coord_col[2],
        '3421': pri_coord_col[3],
        '352': pri_coord_col[4],
        '4141': pri_coord_col[5],
        '343': pri_coord_col[6],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],   
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Tactical system': {
        '4231': std_coord_col[0],
        '433': std_coord_col[1],
        '442': std_coord_col[2],
        '3421': std_coord_col[3],
        '352': std_coord_col[4],
        '4141': std_coord_col[5],
        '343': std_coord_col[6],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Tactical system': {
        '4231': pri_coord_col[0],
        '433': pri_coord_col[1],
        '442': pri_coord_col[2],
        '3421': pri_coord_col[3],
        '352': pri_coord_col[4],
        '4141': pri_coord_col[5],
        '343': pri_coord_col[6],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Failed" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "352" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "4231" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "3421" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "442" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###    League    ###
###              ###
contingency_table = np.array([[141, 242, 137, 142, 859], 
                              [5, 24, 29, 22, 106], 
                              [9, 50, 21, 39, 172], 
                              [16, 90, 15, 24, 305], 
                              [20, 59, 47, 25, 233], 
                              [12, 13, 34, 5, 167]])
# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour aggress :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'League': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'League': {
        'Ligue 1': std_coord_col[0],
        'Liga': std_coord_col[1],
        'Bundesliga': std_coord_col[2],
        'Serie A': std_coord_col[3],
        'Premier League': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='League')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'League': {
        'Ligue 1': pri_coord_col[0],
        'Liga': pri_coord_col[1],
        'Bundesliga': pri_coord_col[2],
        'Serie A': pri_coord_col[3],
        'Premier League': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'League': {
        'Ligue 1': std_coord_col[0],
        'Liga': std_coord_col[1],
        'Bundesliga': std_coord_col[2],
        'Serie A': std_coord_col[3],
        'Premier League': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'League': {
        'Ligue 1': pri_coord_col[0],
        'Liga': pri_coord_col[1],
        'Bundesliga': pri_coord_col[2],
        'Serie A': pri_coord_col[3],
        'Premier League': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Successful" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "Very successful" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "Bundesliga" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "Premier League" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###Aggressiveness###
###              ###
contingency_table = np.array([[738, 121, 637, 459, 92], 
                              [43, 14, 111, 56, 31], 
                              [87, 27, 149, 100, 28], 
                              [89, 39, 182, 147, 57], 
                              [137, 29, 166, 146, 26], 
                              [82, 11, 115, 59, 17]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour aggress :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Aggressiveness': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Aggressiveness': {
        '<60': std_coord_col[0],
        '60-70': std_coord_col[1],
        '70-80': std_coord_col[2],
        '80-90': std_coord_col[3],
        '90-100': std_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Aggressiveness')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Aggressiveness': {
        '<60': pri_coord_col[0],
        '60-70': pri_coord_col[1],
        '70-80': pri_coord_col[2],
        '80-90': pri_coord_col[3],
        '90-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Aggressiveness': {
        '<60': std_coord_col[0],
        '60-70': std_coord_col[1],
        '70-80': std_coord_col[2],
        '80-90': std_coord_col[3],
        '90-100': std_coord_col[4],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Aggressiveness': {
        '<60': pri_coord_col[0],
        '60-70': pri_coord_col[1],
        '70-80': pri_coord_col[2],
        '80-90': pri_coord_col[3],
        '90-100': pri_coord_col[4],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Exceptional" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            elif f"{color}" == "Successful" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "<60" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='top', color=colors[category])
            elif f"{color}" == "90-100" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "70-80" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='center', va='top', color=colors[category])
            elif f"{color}" == "80-90" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###Preferred foot###
###              ###
contingency_table = np.array([[519, 1528], 
                              [71, 184], 
                              [110, 281], 
                              [110, 404], 
                              [104, 400], 
                              [71, 213]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour weak foot :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Preferred foot': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Preferred foot': {
        'Left': std_coord_col[0],
        'Right': std_coord_col[1],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Preferred foot')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Preferred foot': {
        'Left': pri_coord_col[0],
        'Right': pri_coord_col[1],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Preferred foot': {
        'Left': std_coord_col[0],
        'Right': std_coord_col[1],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Preferred foot': {
        'Left': pri_coord_col[0],
        'Right': pri_coord_col[1],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "Right" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()






###              ###
###Center back left ###
###              ###
contingency_table = np.array([[170, 214], 
                              [38, 27], 
                              [49, 40], 
                              [40, 80], 
                              [27, 78], 
                              [24, 51]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour left center back :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Left center back': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Left center back': {
        'left-handed': std_coord_col[0],
        'right-handed': std_coord_col[1],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Left center back')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Left center back': {
        'left-handed': pri_coord_col[0],
        'right-handed': pri_coord_col[1],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Left center back': {
        'left-handed': std_coord_col[0],
        'right-handed': std_coord_col[1],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Left center back': {
        'Left-footed': pri_coord_col[0],
        'Right-footed': pri_coord_col[1],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Success" :
            if f"{color}" == "Perfect" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            elif f"{color}" == "Neutral" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
            else :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
        else : 
            if f"{color}" == "Right-footed" :
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='bottom', color=colors[category])
            else : 
                plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
                plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category]) 
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()





###              ###
###  Left back   ###
###              ###
contingency_table = np.array([[115, 31], 
                              [19, 0], 
                              [33, 6], 
                              [24, 0], 
                              [30, 1], 
                              [21, 1]])

# Test du chi-carré
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Affichage des résultats du test
print("Statistique du test du chi-carré pour left back :", chi2)
print("P-valeur :", p)

percent_explnd_var, pri_coord_row, pri_coord_col, std_coord_row, std_coord_col = calcul_coord(contingency_table)

colors = {'Left back': 'orange', 'Success': 'blue'}

### SCREE PLOT ###
explained_variance_ratio = percent_explnd_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
components = range(1, len(explained_variance_ratio) + 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(components, explained_variance_ratio, color='skyblue', label='% of explained variance')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('% of explained variance', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(components, cumulative_variance_ratio, color='orange', label='% of cumulative explained variance')
ax2.plot(components, cumulative_variance_ratio, color='orange', linestyle='--')
ax2.set_ylabel('% of cumulative explained variance', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

for i, v in enumerate(cumulative_variance_ratio):
    ax2.text(i + 1, v + 0.01, f'{v:.1f}', ha='center', va='bottom')

fig.legend(loc='upper right')
plt.title('Scree Plot')
plt.xticks(components)
plt.tight_layout()
plt.show()

### SYMMETRIC PLOT IN STANDARD COORDINATES ###
rows_coordinates = {
    'Left back': {
        'left-footed': std_coord_col[0],
        'right-footed': std_coord_col[1],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5], 
    }
}

orange_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Left back')
blue_circle_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Success')

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in standard coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### SYMMETRIC PLOT IN PRINCIPAL COORDINATES ###
rows_coordinates = {
    'Left back': {
        'left-footed': pri_coord_col[0],
        'right-footed': pri_coord_col[1],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],  
        'Perfect': pri_coord_row[5],    
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Symmetric plot in principal coordinates')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC ROW PLOT ###
rows_coordinates = {
    'Left back': {
        'left-footed': std_coord_col[0],
        'right-footed': std_coord_col[1],
    },
    'Success': {
        'Failed': pri_coord_row[0],
        'Neutral': pri_coord_row[1],
        'Successful': pri_coord_row[2],
        'Very successful': pri_coord_row[3],
        'Exceptional': pri_coord_row[4],
        'Perfect': pri_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
        plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric row plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.show()

### ASYMMETRIC COLUMN PLOT ###
rows_coordinates = {
    'Left back': {
        'Left-footed': pri_coord_col[0],
        'Right-footed': pri_coord_col[1],
    },
    'Success': {
        'Failed': std_coord_row[0],
        'Neutral': std_coord_row[1],
        'Successful': std_coord_row[2],
        'Very successful': std_coord_row[3],
        'Exceptional': std_coord_row[4],
        'Perfect': std_coord_row[5],  
    }
}

plt.figure(figsize=(6, 6))
for i, (category, coordinates) in enumerate(rows_coordinates.items()):
    for j, (color, coord) in enumerate(coordinates.items()):
        if category == "Left back" :
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='left', va='top', color=colors[category])
        else : 
            plt.scatter(coord[0], coord[1], label=f"{category}: {color}", color=colors[category])
            plt.text(coord[0], coord[1], f"{color}", fontsize=11, ha='right', va='bottom', color=colors[category])
plt.xlabel(f'Dimension 1 ({percent_explnd_var[0]:.1f}%)', fontsize=12)
plt.ylabel(f'Dimension 2 ({percent_explnd_var[1]:.1f}%)', fontsize=12)
plt.title('Asymmetric column plot')
plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)  
plt.axvline(0, color='grey', linestyle='--', linewidth=1.5) 
plt.legend(handles=[orange_circle_handle, blue_circle_handle], loc='lower left')
plt.grid(True)
plt.axis('equal')
plt.show()