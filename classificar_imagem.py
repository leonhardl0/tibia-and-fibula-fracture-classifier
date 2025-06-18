import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

def prever_imagem(caminho_imagem, modelo, tamanho_img=(64, 64)):
    """
    Carrega, pré-processa e classifica uma única imagem usando um modelo de 3 classes.
    """
    try:
        
        img = Image.open(caminho_imagem).convert('L')
        img = img.resize(tamanho_img)
        img_array = np.array(img).astype('float32')
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=[0, -1])

        predicoes = modelo.predict(img_array)
        indice_classe_predita = np.argmax(predicoes, axis=1)[0]

        classes_mapeadas = ['Fractured', 'Not Fractured', 'Others']
        classe_predita = classes_mapeadas[indice_classe_predita]
        return classe_predita, predicoes[0]

    except Exception as e:
        print(f"Erro inesperado ao processar a imagem: {e}")
        return None, None

if __name__ == "__main__":
    # altere essa linha para o caminho da imagem que você quer classificar
    caminho_sua_imagem = "caminho/da/sua/imagem" # <-- mude aqui
    # exemplo: "C:/Users/seu_nome/Downloads/imagem_fratura.png"
    # nome do arquivo do modelo treinado (deve estar na mesma pasta)
    caminho_modelo = "modelo_tibia.keras"

    if not os.path.exists(caminho_modelo):
        print(f"\nERRO CRÍTICO: O modelo treinado '{caminho_modelo}' não foi encontrado.")
        exit()
    if not os.path.exists(caminho_sua_imagem):
         print(f"\nAVISO: O caminho da imagem '{caminho_sua_imagem}' é inválido.")
         exit()

    
    print(f"Carregando modelo de 3 classes: {caminho_modelo}...")
    modelo = load_model(caminho_modelo, compile=False)

    print(f"Analisando a imagem: {caminho_sua_imagem}...")
    classe, confianca = prever_imagem(caminho_sua_imagem, modelo)

    if classe:
        print("\n--------------------------------------")
        print("    Resultado da Análise")
        print("--------------------------------------")
        if classe == 'Others':
            print("  Status: Imagem Inválida")
            print("  Motivo: A imagem foi rejeitada por não ser um raio-X.")
        else:
            print(f"  Diagnóstico Previsto: {classe}")
            print("\nEste é um resultado gerado por um modelo de IA e não substitui um diagnóstico médico profissional.")
        print("--------------------------------------\n")
