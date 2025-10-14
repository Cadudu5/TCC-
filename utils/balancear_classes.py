import argparse
import os
import sys
import pandas as pd


def balance_dataset(df: pd.DataFrame, label_col: str, random_state: int) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Coluna de rótulo '{label_col}' não encontrada no CSV")

    value_counts = df[label_col].value_counts().sort_index()
    if len(value_counts) != 2:
        raise ValueError("O script espera um problema binário (labels 0 e 1)")

    # Identifica classe rara e abundante
    rare_label = value_counts.idxmin()
    abundant_label = value_counts.idxmax()
    target_n = int(value_counts.min())

    df_rare = df[df[label_col] == rare_label]
    df_abundant = df[df[label_col] == abundant_label]

    if len(df_abundant) == target_n:
        # Já balanceado
        balanced = pd.concat([df_rare, df_abundant], axis=0)
    else:
        df_abundant_down = df_abundant.sample(n=target_n, random_state=random_state)
        balanced = pd.concat([df_rare, df_abundant_down], axis=0)

    # Embaralha para evitar blocos por classe
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanceia classes 0/1 por subamostragem aleatória da classe abundante",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Caminho do CSV de entrada")
    parser.add_argument("--output", "-o", required=True, help="Caminho do CSV de saída balanceado")
    parser.add_argument("--label-col", default="label", help="Nome da coluna de rótulo (0/1)")
    parser.add_argument("--random-state", type=int, default=42, help="Semente para reprodutibilidade")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERRO: arquivo de entrada não encontrado: {args.input}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        sys.exit(1)

    before_counts = df[args.label_col].value_counts().sort_index()
    print("Distribuição antes do balanceamento:")
    print(before_counts.to_string())

    try:
        balanced = balance_dataset(df, label_col=args.label_col, random_state=args.random_state)
    except Exception as e:
        print(f"ERRO no balanceamento: {e}")
        sys.exit(1)

    after_counts = balanced[args.label_col].value_counts().sort_index()
    print("\nDistribuição após o balanceamento:")
    print(after_counts.to_string())

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        balanced.to_csv(args.output, index=False)
    except Exception as e:
        print(f"ERRO ao salvar CSV: {e}")
        sys.exit(1)

    print(f"\nCSV balanceado salvo em: {args.output}")


if __name__ == "__main__":
    main()


