from datasets import load_dataset
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

dataset = load_dataset("HanxiGuo/BiScope_Data")

print(f"Dataset loaded")

for split_name in dataset.keys():
    print(f"\n{split_name}: {len(dataset[split_name])} samples")
    print(f"Columns: {dataset[split_name].column_names}")

train_data = dataset['train']
sample = train_data[0]

print(f"\nFirst sample:")
print(f"task: {sample['task']}")
print(f"source: {sample['source']}")
print(f"paraphrased: {sample['paraphrased']}")
print(f"text: {sample['text'][:200]}")

print(f"\nUnique tasks: {set(train_data['task'])}")
print(f"Unique sources: {set(train_data['source'])}")
print(f"Paraphrased values: {set(train_data['paraphrased'])}")

df = pd.DataFrame({
    'task': train_data['task'],
    'source': train_data['source'],
    'paraphrased': train_data['paraphrased'],
    'text': train_data['text']
})

df['label'] = (df['source'] != 'human').astype(int)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\nDataset by Task:")
print(df.groupby('task')['source'].value_counts())

print("\nDataset by Source:")
print(df['source'].value_counts())

print("\nParaphrased Distribution:")
print(df['paraphrased'].value_counts())

print("\nSummary Statistics:")
summary = df.groupby(['task', 'source']).agg({
    'text': 'count',
    'word_count': ['mean', 'min', 'max']
}).round(1)
print(summary)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

df['task'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Samples per Task', fontsize=14)
axes[0,0].set_xlabel('Task')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=45)

df['source'].value_counts().plot(kind='bar', ax=axes[0,1], color=['blue', 'red', 'green', 'orange', 'purple'])
axes[0,1].set_title('Samples per Source', fontsize=14)
axes[0,1].set_xlabel('Source')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=45)

df.boxplot(column='word_count', by='task', ax=axes[1,0])
axes[1,0].set_title('Word Count Distribution by Task', fontsize=14)
axes[1,0].set_xlabel('Task')
axes[1,0].set_ylabel('Word Count')
plt.sca(axes[1,0])
plt.xticks(rotation=45)

human_wc = df[df['source'] == 'human']['word_count']
ai_wc = df[df['source'] != 'human']['word_count']
axes[1,1].hist([human_wc, ai_wc], bins=30, label=['Human', 'AI'], alpha=0.7, color=['blue', 'red'])
axes[1,1].set_title('Word Count: Human vs AI', fontsize=14)
axes[1,1].set_xlabel('Word Count')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('biscope_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to biscope_analysis.png")

output_dir = Path("Dataset/BiScope_Original")
output_dir.mkdir(parents=True, exist_ok=True)

tasks = df['task'].unique()

for task in tasks:
    task_df = df[df['task'] == task]

    print(f"\n{task}: {len(task_df)} samples")
    print(f"Human: {len(task_df[task_df['source'] == 'human'])}")
    print(f"AI: {len(task_df[task_df['source'] != 'human'])}")

    output_file = output_dir / f'{task.lower()}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in task_df.iterrows():
            sample = {
                'text': row['text'],
                'label': row['label'],
                'source': row['source'],
                'paraphrased': row['paraphrased'],
                'task': row['task']
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved to {output_file}")

for task in tasks:
    print(f"\n{task.upper()}")

    task_df = df[df['task'] == task]

    human_samples = task_df[task_df['source'] == 'human']
    if len(human_samples) > 0:
        print(f"\nHUMAN Sample:")
        print(f"{human_samples.iloc[0]['text'][:300]}")

    ai_samples = task_df[task_df['source'] != 'human']
    if len(ai_samples) > 0:
        print(f"\nAI Sample ({ai_samples.iloc[0]['source']}):")
        print(f"{ai_samples.iloc[0]['text'][:300]}")

train_df = df[df['paraphrased'] == False].copy()
test_df = df[df['paraphrased'] == True].copy()

print(f"\nNon-paraphrased samples: {len(train_df)}")
print(f"Paraphrased samples: {len(test_df)}")

train_file = output_dir / 'train_processed.jsonl'
test_file = output_dir / 'test_paraphrased.jsonl'

with open(train_file, 'w', encoding='utf-8') as f:
    for _, row in train_df.iterrows():
        sample = {
            'text': row['text'],
            'label': row['label'],
            'source': row['source'],
            'task': row['task']
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open(test_file, 'w', encoding='utf-8') as f:
    for _, row in test_df.iterrows():
        sample = {
            'text': row['text'],
            'label': row['label'],
            'source': row['source'],
            'task': row['task']
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\nSaved training data to {train_file}")
print(f"Saved test data to {test_file}")

print(f"\nTotal samples: {len(df)}")
print(f"Training (non-paraphrased): {len(train_df)}")
print(f"Testing (paraphrased): {len(test_df)}")

print(f"\nTasks available:")
for task in tasks:
    task_count = len(df[df['task'] == task])
    print(f"{task}: {task_count} samples")

print(f"\nAI Models in dataset:")
ai_sources = df[df['source'] != 'human']['source'].unique()
for source in ai_sources:
    count = len(df[df['source'] == source])
    print(f"{source}: {count} samples")
