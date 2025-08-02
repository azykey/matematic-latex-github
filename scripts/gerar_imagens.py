#!/usr/bin/env python3

"""
Script para gerar imagens de fórmulas LaTeX

Este script lê fórmulas de um arquivo de configuração e gera imagens PNG
usando a biblioteca matplotlib para renderizar LaTeX.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Configurar matplotlib para usar LaTeX
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{physics}'
})

# Criar diretório de imagens se não existir
Path("imagens").mkdir(exist_ok=True)

# Dicionário de fórmulas a serem renderizadas
formulas = {
    "faraday": r"\nabla \times \mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}",
    "schrodinger": r"i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r},t)\right]\Psi(\mathbf{r},t)",
    "maxwell1": r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
    "maxwell2": r"\nabla \cdot \mathbf{B} = 0",
    "maxwell3": r"\nabla \times \mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}",
    "maxwell4": r"\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial\mathbf{E}}{\partial t}",
    "divergencia": r"\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}",
    "rotacional": r"\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_x & F_y & F_z \end{vmatrix}"
}

# Gerar imagens para cada fórmula
print("Gerando imagens de fórmulas...")
for nome, formula in formulas.items():
    fig = plt.figure(figsize=(10, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, f"${formula}$", size=20, ha='center', va='center')
    
    # Salvar a imagem
    output_path = os.path.join("imagens", f"{nome}.png")
    plt.savefig(output_path, bbox_inches='tight', transparent=True, dpi=150)
    plt.close()
    
    print(f"Imagem gerada: {output_path}")

print("Todas as imagens foram geradas com sucesso!")