
# Classificador de Fraturas em Imagens de Raio-X (Tíbia e Fíbula)

Este projeto implementa uma Rede Neural Convolucional (CNN) para classificar imagens de raio-x da região da tíbia e fíbula como **normais** ou **com fratura**.  
Além disso, agora o modelo inclui uma **verificação automática de validade das imagens**, rejeitando aquelas que não correspondem a exames médicos (ex: fotos comuns, imagens irrelevantes).

## Objetivo

Identificar fraturas ósseas em imagens médicas da tíbia e fíbula com alta acurácia, filtrando automaticamente entradas inválidas.  
O projeto é de caráter educacional e **não deve ser utilizado para diagnóstico clínico real**.

---

## Funcionalidades

- Carregamento e pré-processamento automatizado de dataset
- Treinamento de modelo CNN com Keras/TensorFlow
- Predição de imagens externas
- Avaliação de desempenho no conjunto de teste
- **Filtro de imagens inválidas:** rejeita arquivos que não se assemelham a raio-x médico

---

## Estrutura Esperada do Dataset

A pasta do dataset deve ter a seguinte estrutura:

```
tibia_and_fibula_fracture/
├── fracture/
│   ├── 0.png
│   ├── ...
├── normal/
│   ├── 0.png
│   ├── ...
```

> As imagens devem estar no formato `.png` e com conteúdo em tons de cinza (escala de cinza).

---

## Download do Dataset

Utilize o dataset disponível no Kaggle:

[Bone Fracture Dataset - Kaggle](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset)

### Via terminal:

```bash
kaggle datasets download -d orvile/bone-fracture-dataset
unzip bone-fracture-dataset.zip
mv bone-fracture-dataset tibia_and_fibula_fracture
```

---

## Como Executar

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Posicione o dataset no diretório esperado (`tibia_and_fibula_fracture/`).

4. Execute o script:

```bash
python modelo_tibia_atualizado.py
```

---

## 📷 Como Fazer Predições

Após o treinamento, o modelo será salvo como `modelo_tibia.keras`.  
Para fazer previsões com imagens externas, adicione um trecho como este ao seu código:

```python
from tensorflow.keras.models import load_model
from tibia_and_fibula import prever_imagem, label_encoder_global

modelo = load_model("modelo_tibia.keras")
classe, conf = prever_imagem("minha_imagem.png", modelo, label_encoder_global)

print("Classe:", classe)
```

> Se a imagem não for reconhecida como raio-x válido, a predição será ignorada e uma mensagem de erro será exibida.

---

## Requisitos

Veja o arquivo `requirements.txt`:

```
tensorflow>=2.0
numpy
pillow
scikit-learn
kaggle
```

---

## Aviso

Este projeto é apenas para fins de estudo e não substitui diagnósticos realizados por profissionais de saúde. Use com responsabilidade.

