import os
import glob
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def carregar_e_dividir_dataset(pasta_base, tamanho_img=(64, 64), proporcao_teste=0.2, semente_aleatoria=42):
    imagens = []
    labels = []

    classes = sorted([d for d in os.listdir(pasta_base) if os.path.isdir(os.path.join(pasta_base, d))])
    print(f"Classes encontradas: {classes}")

    for nome_classe in classes:
        caminho_classe = os.path.join(pasta_base, nome_classe)
        caminhos_imagens = glob.glob(os.path.join(caminho_classe, '*.png'))

        print(f"Carregando imagens de: {nome_classe} ({len(caminhos_imagens)} imagens)")

        for caminho_img in caminhos_imagens:
            try:
                img = Image.open(caminho_img).convert('L')
                img = img.resize(tamanho_img)
                imagens.append(np.array(img))
                labels.append(nome_classe)
            except Exception as e:
                print(f"Erro ao carregar imagem {caminho_img}: {e}")

    X = np.array(imagens)
    y = np.array(labels)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=proporcao_teste,
        random_state=semente_aleatoria,
        stratify=y
    )

    return X_treino, X_teste, y_treino, y_teste

def pre_processar_dados(X_treino, X_teste, y_treino, y_teste, dimensoes_img):
    X_treino_proc = X_treino.astype('float32') / 255.0
    X_teste_proc = X_teste.astype('float32') / 255.0

    if len(X_treino_proc.shape) == 3:
        X_treino_proc = np.expand_dims(X_treino_proc, axis=-1)
        X_teste_proc = np.expand_dims(X_teste_proc, axis=-1)

    encoder_labels = LabelEncoder()
    y_treino_codificado = encoder_labels.fit_transform(y_treino)
    y_teste_codificado = encoder_labels.transform(y_teste)

    num_classes = len(encoder_labels.classes_)
    y_treino_one_hot = to_categorical(y_treino_codificado, num_classes=num_classes)
    y_teste_one_hot = to_categorical(y_teste_codificado, num_classes=num_classes)

    print(f"Classes detectadas e mapeadas: {encoder_labels.classes_}")

    input_shape = (dimensoes_img[1], dimensoes_img[0], 1)

    return (
        X_treino_proc,
        X_teste_proc,
        y_treino_one_hot,
        y_teste_one_hot,
        encoder_labels,
        num_classes,
        input_shape
    )

def construir_modelo_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def treinar_modelo(modelo, X_treino, y_treino_one_hot, X_teste, y_teste_one_hot, epocas=20, tamanho_lote=32):
    historico = modelo.fit(
        X_treino, y_treino_one_hot,
        epochs=epocas,
        batch_size=tamanho_lote,
        validation_data=(X_teste, y_teste_one_hot),
        verbose=1
    )
    return historico

def avaliar(modelo, X_teste, y_teste_one_hot):
    print("\nAvaliando o modelo no conjunto de teste...")
    perda, acuracia = modelo.evaluate(X_teste, y_teste_one_hot, verbose=0)
    print(f"Perda final no conjunto de teste: {perda:.4f}")
    print(f"Acurácia final no conjunto de teste: {acuracia:.4f}")

# validador de imagem: verifica se a imagem parece ser um raio-X da tíbia ou fíbula
def imagem_parece_raiox_medico(img_array):
    media = np.mean(img_array)
    desvio = np.std(img_array)
    return 30 < media < 200 and desvio > 20

def prever_imagem(caminho_imagem, modelo, label_encoder, tamanho_img=(64, 64)):
    try:
        img = Image.open(caminho_imagem).convert('L')
        img = img.resize(tamanho_img)
        img_array = np.array(img).astype('float32')

        if not imagem_parece_raiox_medico(img_array):
            print("Imagem rejeitada: não parece ser um raio-X da tíbia ou fíbula.")
            return "Imagem não reconhecida como raio-X da tíbia/fíbula", None

        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        predicoes = modelo.predict(img_array)
        indice_classe_predita = np.argmax(predicoes, axis=1)[0]
        classe_predita = label_encoder.inverse_transform([indice_classe_predita])[0]

        return classe_predita, predicoes[0]

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None, None

label_encoder_global = None

def main():
    global label_encoder_global

    pasta_base_dataset = "CAMINHO/DO/SEU/DATASET"
    # exemplo: "C:/Users/seu_nome/Downloads/tibia-and-fibula-fracture-classifier-main"
    dimensoes_imagens = (64, 64)
    proporcao_teste = 0.2
    semente_aleatoria = 42
    epocas_treino = 5
    tamanho_lote_treino = 32

    print("\n--- ETAPA 1: Carregando e Dividindo o Dataset ---")
    X_treino, X_teste, y_treino, y_teste = carregar_e_dividir_dataset(
        pasta_base_dataset,
        dimensoes_imagens,
        proporcao_teste,
        semente_aleatoria
    )

    if len(X_treino) == 0 or len(X_teste) == 0:
        print("Não há dados suficientes para prosseguir com o treinamento. Encerrando.")
        return

    print("\n--- ETAPA 2: Pré-processando os Dados ---")
    X_treino_proc, X_teste_proc, y_treino_one_hot, y_teste_one_hot, label_encoder_global, num_classes, input_shape = pre_processar_dados(
        X_treino, X_teste, y_treino, y_teste, dimensoes_imagens
    )

    print("\n--- ETAPA 3: Construindo o Modelo CNN ---")
    modelo_cnn = construir_modelo_cnn(input_shape, num_classes)

    print("\n--- ETAPA 4: Treinando o Modelo ---")
    treinar_modelo(
        modelo_cnn,
        X_treino_proc, y_treino_one_hot,
        X_teste_proc, y_teste_one_hot,
        epocas_treino, tamanho_lote_treino
    )

    print("\n--- ETAPA 5: Avaliando e Salvando o Modelo ---")
    avaliar(modelo_cnn, X_teste_proc, y_teste_one_hot)

    modelo_cnn.save("modelo_tibia.keras")
    print("Modelo salvo como 'modelo_tibia.keras'")

if __name__ == "__main__":
    main()
