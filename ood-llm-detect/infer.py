import torch
from transformers import AutoTokenizer
from lightning import Fabric
import argparse
import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm

# These parameters are estimated by the test set
distrib_params = {'deepfake': {'mu0': 2.8207, 'sigma0': 1.188, 'mu1': 0.2149, 'sigma1': 2.3777},
                'M4': {'mu0': 2.8210, 'sigma0': 1.3977, 'mu1': 0.08976, 'sigma1': 2.79554},
                'raid': {'mu0': 3.3258, 'sigma0': 1.19811, 'mu1': 0.2563, 'sigma1': 2.39623}}
# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
# Copied from FastDetectGPT
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

@torch.no_grad()
def predict_single(text, model, tokenizer, device="cuda", dataset_name="deepfake"):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # 模型前向
    model.eval()
    loss, out, _, _ = model(encoded, 0, 0, torch.tensor([0]).to(device))
    # out 是概率分数（或 logits）
    prob = compute_prob_norm(out.cpu().numpy(), 
                    distrib_params[dataset_name]['mu0'], distrib_params[dataset_name]['sigma0'],
                    distrib_params[dataset_name]['mu1'], distrib_params[dataset_name]['sigma1'])
    return prob.item()

def load_model(opt):
    # 根据 ood_type 选择模型定义
    if opt.ood_type == "deepsvdd":
        from src.deep_SVDD import SimCLR_Classifier_SCL
    elif opt.ood_type == "energy":
        from src.energy import SimCLR_Classifier_SCL
    elif opt.ood_type == "hrn":
        from src.hrn import SimCLR_Classifier_SCL
    else:
        raise ValueError("Only support deepsvdd, hrn and energy")

    fabric = Fabric(accelerator="cuda", devices=1)
    fabric.launch()
    if opt.ood_type == "hrn":
        model = SimCLR_Classifier_SCL(opt, opt.num_models, fabric)
    else:
        model = SimCLR_Classifier_SCL(opt, fabric)
    state_dict = torch.load(opt.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.cuda()
    tokenizer = model.model.tokenizer
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Just for placeholder
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--d', type=float, default=1,help="classifier loss weight")
    parser.add_argument("--nu", type=float, default=0.1, help="DeepSVDD HP nu")
    parser.add_argument("--objective", type=str, default="one-class", help="one-class,soft-boundary")
    parser.add_argument("--out_dim", type=int, default=128, help="output dim and dim of c")
    parser.add_argument("--only_classifier", action='store_true',help="only use classifier, no contrastive loss")
    # Key param
    parser.add_argument('--mode', type=str, default='deepfake', help="deepfake,raid,M4")
    parser.add_argument('--ood_type', type=str, default='deepsvdd', help="deepsvdd, energy")
    parser.add_argument("--model_path", type=str, default="xxx/model_best_gpt35.pth", help="Path to the  model checkpoint")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")

    opt = parser.parse_args()

    model, tokenizer = load_model(opt)
    # text = input("请输入要检测的文本：\n> ")
    # prob = predict_single(text, model, tokenizer, dataset_name=opt.mode)
    # print(f"\n💡 该文本是 LLM 生成的概率为: {prob:.4f}")
    
    # synthetic test
    syn_df = pd.read_csv("/workspace/dsc261/eurlex_synthetic_data/synthetic_10k_samples.csv")
    results = []
    for idx in tqdm(range(len(syn_df))):
        text = syn_df.loc[idx, 'synthetic_text']
        id = syn_df.loc[idx, 'celex_id']  # 0 for human, 1 for LLM
        prob = predict_single(text, model, tokenizer, dataset_name=opt.mode)
        results.append(prob)
        
    # add results to dataframe
    syn_df['detection_prob'] = results
    syn_df.to_csv("./outputs/eurlex_synthetic_data/synthetic_10k_samples.csv", index=False)
    
    # human data
    human_df = pd.read_csv("/workspace/dsc261/eurlex_synthetic_data/original_10k_samples.csv")
    results = []
    for idx in tqdm(range(len(human_df))):
        text = human_df.loc[idx, 'text']
        prob = predict_single(text, model, tokenizer, dataset_name=opt.mode)
        results.append(prob)
    human_df['detection_prob'] = results
    human_df.to_csv("./outputs/eurlex_synthetic_data/original_10k_samples.csv", index=False)
    
    # syn_token_lens = []
    # for idx in tqdm(range(len(syn_df))):
    #     text = syn_df.loc[idx, 'synthetic_text']
    #     encoded = tokenizer(
    #         text,
    #         return_tensors="pt",
    #     )
    #     token_len = (encoded['attention_mask'] == 1).sum().item()
    #     syn_token_lens.append(token_len)
    # # print stats of token lens
    # print("Synthetic Data Token Lengths:")
    # print(f"Mean: {np.mean(syn_token_lens):.2f}, Std: {np.std(syn_token_lens):.2f}, Max: {np.max(syn_token_lens)}, Min: {np.min(syn_token_lens)}")
    # print("Quantiles:")
    # for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    #     print(f"{q*100}%: {np.quantile(syn_token_lens, q)}")
    # print("--------------------------")
    # human_token_lens = []
    # for idx in tqdm(range(len(human_df))):
    #     text = human_df.loc[idx, 'text']
    #     encoded = tokenizer(
    #         text,
    #         return_tensors="pt",
    #     )
    #     token_len = (encoded['attention_mask'] == 1).sum().item()
    #     human_token_lens.append(token_len)
    # print("Human Data Token Lengths:")
    # print(f"Mean: {np.mean(human_token_lens):.2f}, Std: {np.std(human_token_lens):.2f}, Max: {np.max(human_token_lens)}, Min: {np.min(human_token_lens)}")
    # print("Quantiles:")
    # for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    #     print(f"{q*100}%: {np.quantile(human_token_lens, q)}")  