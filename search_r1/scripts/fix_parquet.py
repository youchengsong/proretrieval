"""Fix RL parquet: convert JSON string columns to proper dicts."""
import json
import sys
import pandas as pd

def fix_parquet(input_path, output_path):
    df = pd.read_parquet(input_path)

    for col in ['extra_info', 'reward_model']:
        if col in df.columns and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(json.loads)
            print(f"  Converted {col} from str to dict")

    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    inp = sys.argv[1]
    out = sys.argv[2]
    print(f"Fixing {inp}")
    fix_parquet(inp, out)
