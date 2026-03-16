import pandas as pd
import os

# Configurações
FUNDO_DIR = 'fundo'
MAIN_DATASET = 'imagens_enriquecidas_completo.csv'
OUTPUT_DATASET = 'dataset_completo_sem_fundo_manual.csv'
OUTPUT_FUNDO_CONSOLIDADO = 'fundo/todos_fundos_identificados.csv'

# Mapeamento manual de arquivos de fundo para image_origin no dataset principal
# Chave: Nome do arquivo na pasta fundo
# Valor: Nome correspondente na coluna image_origin do dataset principal
FILE_MAPPING = {
    'rotulos_fundo_40xb.csv': 'rotulos_40xb_enriquecido.csv',
    'rotulos_fundo_7 dias 40x.csv': 'rotulos_7 dias 40x_enriquecido.csv',
    'rotulos_fundo_neutr wt 40xb2.csv': 'rotulos_neutr wt 40xb2_enriquecido.csv',
    'rotulos_fundo_Neutrofilo IL4 42dias 40X.csv': 'rotulos_Neutrofilo IL4 42dias 40X_enriquecido.csv',
    'rotulos_fundo_Neutrofilo IL4 42dias b40X.csv': 'rotulos_Neutrofilo IL4 42dias b40X (1)_enriquecido.csv',
    'rotulos_fundo_WT 42 dias neutrófilos 6 40X.csv': 'rotulos_WT 42 dias neutrófilos 6 40X_enriquecido.csv'
}

def main():
    print("--- Removendo Fundo Manualmente ---")
    
    # 1. Carregar e consolidar arquivos de fundo
    print("Lendo arquivos de fundo...")
    fundo_dfs = []
    
    for fundo_file, target_origin in FILE_MAPPING.items():
        path = os.path.join(FUNDO_DIR, fundo_file)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Filtrar apenas o que foi marcado como fundo (label == 1)
                # Assumindo que 1 = Fundo e 0 = Não Fundo, conforme rotulador_gui_fundo.py
                df_fundo = df[df['label'] == 1].copy()
                
                if not df_fundo.empty:
                    df_fundo['image_origin'] = target_origin
                    fundo_dfs.append(df_fundo[['superpixel_id', 'image_origin']])
                    print(f"  - {fundo_file}: {len(df_fundo)} superpixels de fundo identificados.")
                else:
                    print(f"  - {fundo_file}: Nenhum superpixel marcado como fundo (label=1).")
            except Exception as e:
                print(f"  ERRO ao ler {fundo_file}: {e}")
        else:
            print(f"  AVISO: Arquivo {path} não encontrado.")
            
    if not fundo_dfs:
        print("Nenhum arquivo de fundo com marcações encontrado. Encerrando.")
        return

    all_fundo = pd.concat(fundo_dfs, ignore_index=True)
    print(f"Total de superpixels de fundo para remover: {len(all_fundo)}")
    
    # Salvar consolidado para conferência
    all_fundo.to_csv(OUTPUT_FUNDO_CONSOLIDADO, index=False)
    print(f"Arquivo de fundos consolidado salvo em: {OUTPUT_FUNDO_CONSOLIDADO}")

    # 2. Carregar dataset principal
    print(f"\nCarregando dataset principal: {MAIN_DATASET}")
    if not os.path.exists(MAIN_DATASET):
        print("Dataset principal não encontrado.")
        return
        
    df_main = pd.read_csv(MAIN_DATASET)
    initial_len = len(df_main)
    print(f"Total de registros no dataset principal: {initial_len}")

    # 3. Remover superpixels de fundo
    print("Removendo superpixels...")
    # Left join com indicador. Se _merge == 'left_only', mantemos. Se 'both', removemos.
    merged = df_main.merge(all_fundo, on=['superpixel_id', 'image_origin'], how='left', indicator=True)
    
    # Filtrar apenas os que estão apenas no dataset principal (left_only)
    df_clean = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    final_len = len(df_clean)
    removed_count = initial_len - final_len
    
    print(f"Registros removidos: {removed_count}")
    print(f"Total final de registros: {final_len}")
    
    # 4. Salvar resultado
    df_clean.to_csv(OUTPUT_DATASET, index=False)
    print(f"\nDataset limpo salvo em: {OUTPUT_DATASET}")

if __name__ == "__main__":
    main()
