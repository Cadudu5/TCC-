import pandas as pd
import os
import glob

# --- CONFIGURAÇÕES ---
# Pasta onde estão os seus arquivos CSV individuais
PASTA_DE_ENTRADA = 'datasets_individuais'

# Nome do arquivo de saída que será criado
ARQUIVO_DE_SAIDA = 'dataset_completo.csv'

print("--- Iniciando Script de União de Datasets CSV ---")

# --- 1. Validação da pasta de entrada ---
if not os.path.isdir(PASTA_DE_ENTRADA):
    print(f"ERRO: A pasta '{PASTA_DE_ENTRADA}' não foi encontrada.")
    print("Por favor, crie a pasta e coloque seus arquivos CSV dentro dela.")
    exit()

# --- 2. Encontrar todos os arquivos CSV na pasta ---
# O padrão '*.csv' encontra qualquer arquivo que termine com .csv
caminho_dos_arquivos = os.path.join(PASTA_DE_ENTRADA, '*.csv')
lista_de_arquivos = glob.glob(caminho_dos_arquivos)

if not lista_de_arquivos:
    print(f"ERRO: Nenhum arquivo .csv foi encontrado na pasta '{PASTA_DE_ENTRADA}'.")
    exit()

print(f"Encontrados {len(lista_de_arquivos)} arquivos para unir:")
for f in lista_de_arquivos:
    print(f" - {os.path.basename(f)}")

# --- 3. Ler, processar e unir os arquivos ---
# Lista para armazenar cada DataFrame individual antes de uni-los
lista_de_dataframes = []

for arquivo in lista_de_arquivos:
    # Lê o arquivo CSV para um DataFrame do pandas
    df = pd.read_csv(arquivo)
    
    # Adiciona uma nova coluna para rastrear a origem de cada linha (superpixel)
    nome_base_arquivo = os.path.basename(arquivo)
    df['image_origin'] = nome_base_arquivo
    
    # Adiciona o DataFrame processado à nossa lista
    lista_de_dataframes.append(df)

print("\nUnindo todos os dados...")
# Concatena todos os DataFrames da lista em um único DataFrame
dataset_mestre = pd.concat(lista_de_dataframes, ignore_index=True)

# --- 4. Salvar o dataset mestre ---
dataset_mestre.to_csv(ARQUIVO_DE_SAIDA, index=False)

print("\n--- Processo Concluído com Sucesso! ---")
print(f"Dataset mestre salvo em: '{ARQUIVO_DE_SAIDA}'")
print(f"O dataset final contém {dataset_mestre.shape[0]} linhas e {dataset_mestre.shape[1]} colunas.")