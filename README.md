# Classificador de Fraturas na Tíbia e Fíbula

Este projeto utiliza uma Rede Neural Convolucional (CNN) para classificar imagens de raios-X, identificando a presença de fraturas na tíbia e fíbula. O modelo é capaz de distinguir entre três categorias: imagens com fratura, sem fratura e imagens que não são raios-X relevantes.

O projeto inclui scripts para treinar o modelo do zero (`tibia_and_fibula.py`) e para classificar uma nova imagem utilizando um modelo pré-treinado (`classificar_imagem.py`).

## Dataset de Treinamento

O modelo `modelo_tibia.keras` incluso neste repositório foi treinado com uma combinação de datasets para alcançar um desempenho robusto e preciso.

* **Classes `Fractured` e `Not Fractured`**:
    * **Fonte**: [Bone Fracture Dataset no Kaggle](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset)
    * **Imagens Utilizadas**: 2.127 imagens, divididas em:
        * 2.000 imagens para a classe `Fractured`.
        * 127 imagens para a classe `Not Fractured`.

* **Classe `Others`**:
    * **Fonte**: [Imagenette e Imagewoof](https://github.com/fastai/imagenette)
    * **Imagens Utilizadas**: Cerca de 2.127 imagens (em resolução de 320px) foram usadas para ensinar o modelo a identificar e rejeitar imagens que não correspondem ao domínio de interesse (raios-X de ossos).

## Funcionalidades

### Treinamento do Modelo (`tibia_and_fibula.py`)

* **Carregamento de Dados**: Carrega imagens nos formatos `.png`, `.jpeg` e `.jpg` a partir de um diretório base.
* **Pré-processamento**: As imagens são convertidas para escala de cinza e redimensionadas para 64x64 pixels. Os valores dos pixels são normalizados para o intervalo [0, 1].
* **Arquitetura do Modelo**: Uma Rede Neural Convolucional sequencial é construída com camadas de `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` e `Dropout` para regularização.
* **Salvamento**: Após o treinamento, o modelo final é salvo como `modelo_tibia.keras` no diretório do projeto.

### Classificação de Imagem (`classificar_imagem.py`)

* **Carregamento do Modelo**: Carrega o modelo `modelo_tibia.keras` para realizar predições.
* **Predição**: Classifica uma imagem fornecida em uma das três categorias: `Fractured`, `Not Fractured`, ou `Others`.
* **Validação**: Se a imagem for classificada como `Others`, o script informa que a imagem foi rejeitada por não ser um raio-X.

## Requisitos

Para executar este projeto, instale as dependências listadas no arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```
As dependências incluem:
* tensorflow==2.15.0
* numpy==1.24.3
* pillow==10.2.0
* scikit-learn==1.3.0
* kaggle==1.6.12

## Como Utilizar

⚠️ **Importante**: Para que os scripts funcionem corretamente, você deve ajustar os caminhos dos arquivos para que correspondam à localização no seu computador.

### 1. Para Classificar uma Imagem (usando o modelo pré-treinado)

1.  Certifique-se de que o modelo `modelo_tibia.keras` está na mesma pasta que os scripts.
2.  Abra o arquivo `classificar_imagem.py` e **altere o valor da variável `caminho_sua_imagem`** para o caminho completo da imagem que você deseja analisar.
    ```python
    # altere essa linha para o caminho da imagem que você quer classificar
    caminho_sua_imagem = "caminho/da/sua/imagem" # <-- mude aqui
    ```
3.  Execute o script no terminal:
    ```bash
    python classificar_imagem.py
    ```

### 2. Para Treinar o Modelo do Zero

1.  Baixe os datasets e organize as imagens em um diretório base, com subpastas para cada classe (ex: `Fractured/`, `Not Fractured/`, `Others/`).
2.  Abra o arquivo `tibia_and_fibula.py` e **altere o valor da variável `pasta_base_dataset`** para o caminho do diretório onde você salvou o dataset.
    ```python
    # altere essa linha para o caminho do seu dataset
    pasta_base_dataset = "caminho/do/seu/dataset" # <-- mude aqui
    ```
3.  Execute o script de treinamento no terminal:
    ```bash
    python tibia_and_fibula.py
    ```

## Aviso Legal

Este projeto foi desenvolvido para fins educacionais. O resultado gerado pelo modelo é uma previsão e não substitui um diagnóstico médico profissional.
