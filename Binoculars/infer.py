from binoculars import Binoculars
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    bino = Binoculars()
    
    # synthetic test
    syn_df = pd.read_csv("/workspace/dsc261/eurlex_synthetic_data/synthetic_10k_samples.csv")
    results = []
    for idx in tqdm(range(len(syn_df))):
        text = syn_df.loc[idx, 'synthetic_text']
        id = syn_df.loc[idx, 'celex_id']  # 0 for human, 1 for LLM
        prob = bino.compute_score(text)
        results.append(prob)
        
    # add results to dataframe
    syn_df['detection_prob'] = results
    syn_df.to_csv("./outputs/eurlex_synthetic_data/synthetic_10k_samples.csv", index=False)
    
    # human data
    human_df = pd.read_csv("/workspace/dsc261/eurlex_synthetic_data/original_10k_samples.csv")
    results = []
    for idx in tqdm(range(len(human_df))):
        text = human_df.loc[idx, 'text']
        prob = bino.compute_score(text)
        results.append(prob)
    human_df['detection_prob'] = results
    human_df.to_csv("./outputs/eurlex_synthetic_data/original_10k_samples.csv", index=False)