# utils/data_preprocessor.py
import pandas as pd

def extract_text_feature(df: pd.DataFrame) -> pd.Series:
    """
    Automatically detect or select the best column for prediction input.
    Returns a Series of textual features (gene names or sequences).
    """

    possible_fields = [
        "CARD Short Name", "Gene Name", "Resistance Gene",
        "ARO Name", "Sequence", "Protein Name", "Locus Tag"
    ]

    for col in possible_fields:
        if col in df.columns:
            print(f"✅ Using '{col}' as feature input.")
            return df[col].astype(str)

    # fallback: if no known field found
    print("⚠️ No standard AMR text field found. Using first text-like column...")
    text_cols = df.select_dtypes(include=["object"]).columns
    if len(text_cols) == 0:
        raise ValueError("No valid text column found for prediction.")
    
    return df[text_cols[0]].astype(str)
