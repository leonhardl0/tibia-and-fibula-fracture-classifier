## Classificador de Fraturas na Tíbia e Fíbula

Este projeto utiliza um modelo de aprendizado de máquina para classificar fraturas na tíbia e fíbula a partir de imagens médicas.

## Requisitos

- Python 3.8+
- TensorFlow / Keras
- NumPy
- OpenCV (se estiver usando imagens diretamente)
- Outros pacotes especificados em `requirements.txt`

## Arquivos Necessários

- `modelo_tibia.keras`: arquivo do modelo treinado. **Este arquivo é essencial para a execução do código.**

  Coloque esse arquivo na raiz do projeto ou ajuste o caminho no código que o carrega:

  ```python
  from tensorflow.keras.models import load_model
  modelo = load_model('modelo_tibia.keras')
  ```

## Execução

1. Instale os requisitos:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o script principal:
   ```bash
   python main.py
   ```

## Estrutura do Projeto

```
tibia_and_fibula_fracture/
├── main.py
├── modelo_tibia.keras
├── README.md
├── requirements.txt
└── (outros arquivos)
```

## Observações

- Certifique-se de que o modelo esteja no local correto antes de rodar o script.
- O modelo foi treinado previamente e não está incluído no repositório público por questões de tamanho/licença (se aplicável).
