# Classificador de Fraturas na Tíbia e Fíbula

Este projeto implementa um classificador de imagens para detectar fraturas na tíbia e fíbula usando redes neurais com Keras/TensorFlow.

## Visão Geral

O objetivo deste projeto é treinar e utilizar um modelo de aprendizado profundo para identificar automaticamente fraturas em imagens médicas da tíbia e fíbula. Ele é baseado no dataset de raio-X disponível no Kaggle.

## Dataset

Para executar este projeto, é necessário baixar o dataset de imagens:

- **Kaggle Dataset**: [Bone Fracture Dataset](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset)

Após o download, extraia os arquivos e coloque-os em um diretório chamado `data/` na raiz do projeto (ou ajuste o caminho no script conforme necessário).

Você pode baixar via terminal (com a [Kaggle API](https://github.com/Kaggle/kaggle-api)):

```bash
kaggle datasets download -d orvile/bone-fracture-dataset
unzip bone-fracture-dataset.zip -d data/
```

> ⚠️ **Importante:** No código-fonte (por exemplo, no arquivo `main.py` ou `tibia_and_fibula.py`), há uma variável chamada `pasta_base_dataset`. Você deve alterar seu valor para o caminho correto onde o dataset foi salvo no seu computador. Exemplo:
>
> ```python
> pasta_base_dataset = "C:/seu/caminho/para/o/dataset"
> ```
>
> Caso contrário, o código não conseguirá localizar as imagens e irá falhar.

## Modelo Pré-Treinado

Para utilizar diretamente o classificador (sem re-treinamento), você precisa do arquivo:

- `modelo_tibia.keras` (incluso neste repositório)

Esse é o modelo treinado com os dados acima e pronto para fazer previsões.

```python
from tensorflow.keras.models import load_model

modelo = load_model('modelo_tibia.keras')
```

## Requisitos

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (se estiver usando pré-processamento de imagem)
- Jupyter Notebook (opcional, mas recomendado)

Instale os pacotes com:

```bash
pip install -r requirements.txt
```

## Execução

1. Certifique-se de que os dados estão em `./data` ou no caminho ajustado corretamente
2. Verifique se `modelo_tibia.keras` está na raiz do projeto
3. Execute o script ou notebook principal:

```bash
python main.py
```

Ou, se for um notebook:

```bash
jupyter notebook
```

## Estrutura Esperada

```
tibia_and_fibula_fracture/
├── data/
│   └── Fractured/ e Not Fractured/
├── modelo_tibia.keras
├── main.py
├── requirements.txt
├── README.md
```

## Observações

- O modelo já treinado é útil para inferência imediata.
- Se quiser treinar do zero, ajuste os scripts para carregar os dados e treinar novamente.
- O dataset original possui imagens divididas por fraturas/not-fraturas em subpastas.

## Licença

Verifique os termos de uso do dataset no [Kaggle](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset) e adapte sua licença de projeto conforme necessário.


