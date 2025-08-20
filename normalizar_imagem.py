import staintools
import cv2
import os

# --- CONFIGURAÇÕES ---
# Caminho para a sua imagem de referência (a "coloração ideal")
PATH_REFERENCIA = 'imagem_referencia.png'

# Caminho para a imagem que você quer normalizar
PATH_FONTE = 'imagem_para_normalizar.jpg' 

# Caminho onde a imagem normalizada será salva
PATH_SAIDA = 'imagem_normalizada.png'

print("--- Iniciando Script de Normalização de Cor ---")

# --- 1. Validação dos arquivos de entrada ---
if not os.path.exists(PATH_REFERENCIA):
    print(f"ERRO: Imagem de referência não encontrada em '{PATH_REFERENCIA}'")
    exit()
if not os.path.exists(PATH_FONTE):
    print(f"ERRO: Imagem de origem não encontrada em '{PATH_FONTE}'")
    exit()

# --- 2. Carregamento das Imagens ---
# Carrega as imagens. cv2 lê em formato BGR, então convertemos para RGB.
print("Carregando imagens...")
target = cv2.imread(PATH_REFERENCIA)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

source = cv2.imread(PATH_FONTE)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)


# --- 3. Inicialização e Aplicação do Normalizador ---
# Inicializa o normalizador com o método 'vahadane'
# Outra opção popular seria 'macenko'
normalizer = staintools.StainNormalizer(method='vahadane')

# Ajusta ("fita") o normalizador à imagem de referência para aprender os corantes ideais.
print(f"Ajustando o normalizador à imagem de referência: '{PATH_REFERENCIA}'...")
normalizer.fit(target)

# Aplica a transformação à imagem de origem
print(f"Normalizando a imagem de origem: '{PATH_FONTE}'...")
normalized_image = normalizer.transform(source)


# --- 4. Salvamento do Resultado ---
# Converte a imagem de volta para BGR para salvar com cv2
print(f"Salvando a imagem normalizada em: '{PATH_SAIDA}'...")
normalized_image_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(PATH_SAIDA, normalized_image_bgr)

print("\n--- Normalização Concluída com Sucesso! ---")