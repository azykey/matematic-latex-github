# Imagens de Fórmulas e Diagramas 🖼️

Esta pasta contém imagens de fórmulas matemáticas, diagramas e visualizações utilizadas no repositório.

## Conteúdo

### Fórmulas Físicas
- `faraday.png`: Lei de Faraday da indução eletromagnética
- `maxwell1.png`: Primeira equação de Maxwell (Lei de Gauss para o campo elétrico)
- `maxwell2.png`: Segunda equação de Maxwell (Lei de Gauss para o campo magnético)
- `maxwell3.png`: Terceira equação de Maxwell (Lei de Faraday)
- `maxwell4.png`: Quarta equação de Maxwell (Lei de Ampère-Maxwell)

### Fórmulas Matemáticas
- `divergencia.png`: Fórmula da divergência de um campo vetorial
- `rotacional.png`: Fórmula do rotacional de um campo vetorial
- `schrodinger.png`: Equação de Schrödinger

## Geração de Imagens

As imagens nesta pasta são geradas automaticamente pelo script `scripts/gerar_imagens.py`, que utiliza matplotlib para renderizar fórmulas LaTeX em formato PNG.

Para gerar ou atualizar as imagens, execute:

```bash
python scripts/gerar_imagens.py
```

## Uso em Markdown

Para incluir estas imagens em arquivos Markdown, use a sintaxe:

```markdown
![Descrição da Imagem](./imagens/nome_da_imagem.png)
```

Exemplo:

```markdown
![Equação de Schrödinger](./imagens/schrodinger.png)
```