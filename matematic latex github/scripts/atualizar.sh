#!/bin/bash

# Script para atualizar automaticamente os arquivos derivados do repositório

echo "Iniciando atualização dos arquivos..."

# Criar diretório de saída se não existir
mkdir -p output

# Converter notebooks para HTML
echo "Convertendo notebooks para HTML..."
jupyter nbconvert --to html fisica/**/*.ipynb --output-dir=./output/html/fisica/
jupyter nbconvert --to html matematica/**/*.ipynb --output-dir=./output/html/matematica/

# Compilar arquivos LaTeX para PDF
echo "Compilando arquivos LaTeX para PDF..."
find . -name "*.tex" -exec pdflatex -output-directory=./output/pdf {} \;

# Gerar imagens de fórmulas
echo "Gerando imagens de fórmulas..."
python scripts/gerar_imagens.py

echo "Atualização concluída!"