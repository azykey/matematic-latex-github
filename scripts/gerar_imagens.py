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
    # Física
    "faraday": r"\nabla \times \mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}",
    "schrodinger": r"i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r},t)\right]\Psi(\mathbf{r},t)",
    "maxwell1": r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
    "maxwell2": r"\nabla \cdot \mathbf{B} = 0",
    "maxwell3": r"\nabla \times \mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}",
    "maxwell4": r"\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial\mathbf{E}}{\partial t}",
    "divergencia": r"\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}",
    "rotacional": r"\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_x & F_y & F_z \end{vmatrix}",
    
    # AGI - Percepção
    "convolution2d": r"F(i,j) = \sum_m \sum_n K(m,n)I(i-m,j-n)",
    "batch_norm": r"y = \gamma \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta",
    "attention": r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
    
    # AGI - Memória
    "lstm": r"\begin{align*} c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\ h_t &= o_t \odot \tanh(c_t) \end{align*}",
    "hopfield": r"s_i(t+1) = \text{sign}\left(\sum_j w_{ij} s_j(t)\right)",
    
    # AGI - Raciocínio
    "bayes": r"P(H|E) = \frac{P(E|H)P(H)}{P(E)}",
    "causal": r"P(Y|\text{do}(X=x)) = \sum_Z P(Y|X=x,Z)P(Z)",
    
    # AGI - Aprendizado
    "gradient_descent": r"w_t = w_{t-1} - \eta \nabla L(w_{t-1})",
    "adam": r"w_t = w_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
    "qlearning": r"Q(s,a) = Q(s,a) + \alpha \left[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]",
    
    # AGI - Auto-Melhoria
    "maml": r"\theta = \theta - \beta \nabla_{\theta} \sum_{\tau} L_{\tau}(f_{\theta'})",
    "nas": r"A^* = \arg\max_A \mathbb{E}_{(x,y) \sim D} [L(w^*(A), x, y)]"
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