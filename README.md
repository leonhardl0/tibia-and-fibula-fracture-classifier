# Classificador de Fraturas em Imagens de Raio-X (Tíbia e Fíbula)

Este projeto utiliza uma Rede Neural Convolucional (CNN) desenvolvida com TensorFlow/Keras para classificar imagens de raio-X da tíbia e fíbula em duas categorias: **normal** ou **com fratura**.

## 🔍 Objetivo

O modelo analisa imagens médicas simples (formato `.png`) em tons de cinza e identifica se há fraturas. O foco está em imagens da tíbia e fíbula com resolução padronizada de 64x64 pixels.

## 📥 Download do Dataset

O dataset utilizado pode ser baixado diretamente do Kaggle:

🔗 [Bone Fracture Dataset - Kaggle](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset)

### Usando o Kaggle CLI

Se você tiver o Kaggle CLI instalado, pode baixar diretamente com:

```bash
kaggle datasets download -d orvile/bone-fracture-dataset
unzip bone-fracture-dataset.zip
```

Após extrair, certifique-se de renomear a pasta para `tibia_and_fibula_fracture` e posicioná-la no mesmo diretório do script `tibia_and_fibula.py`.

## 📁 Estrutura Esperada do Dataset

```
tibia_and_fibula_fracture/
├── fracture/
│   ├── 0.png
│   ├── ...
├── normal/
│   ├── 0.png
│   ├── ...
```

## 🚀 Funcionalidades

- Carregamento e divisão automática do dataset (treinamento/teste)
- Pré-processamento e codificação dos dados
- Treinamento de uma CNN com Keras
- Avaliação da acurácia e perda
- Predição de imagens externas para teste

## ⚙️ Como usar

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

2. Instale os requisitos:

```bash
pip install -r requirements.txt
```

3. Baixe e posicione o dataset conforme indicado acima.

4. Execute o script:

```bash
python tibia_and_fibula.py
```

## 🧪 Requisitos

Consulte o arquivo `requirements.txt` para instalar todas as dependências necessárias.

## 📌 Observação

Este projeto tem finalidade educacional e não deve ser utilizado como ferramenta diagnóstica clínica.
