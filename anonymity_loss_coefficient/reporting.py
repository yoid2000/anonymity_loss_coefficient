import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_alc_prec_best(df: pd.DataFrame,
                  strong_thresh: float, risk_thresh: float,
                  attack_name: str,
                  file_path: str) -> None:
    if len(df) < 10:
        return
    idx = df.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_best = df.loc[idx].reset_index(drop=True)
    _plot_alc_prec(df_best, strong_thresh, risk_thresh, attack_name, file_path)

def plot_alc_prec(df: pd.DataFrame,
                  strong_thresh: float, risk_thresh: float,
                  attack_name: str,
                  file_path: str) -> None:
    if len(df) < 10:
        return
    _plot_alc_prec(df, strong_thresh, risk_thresh, attack_name, file_path)

def _plot_alc_prec(df: pd.DataFrame,
                  strong_thresh: float, risk_thresh: float,
                  attack_name: str,
                  file_path: str) -> None:
    if len(df) < 10:
        return
    df = df.copy()
    # set any 'alc' values less than -3.0 to -3.0
    df['alc'] = df['alc'].apply(lambda x: max(x, -3.0))
    plt.figure(figsize=(6, 4))
    scatter = sns.scatterplot(data=df, x='attack_prec', y='alc', hue='attack_recall', palette='viridis', legend=False)
    low_alc = df['alc'].min()
    lower_ylim = min(-0.05, low_alc)
    plt.ylim(lower_ylim, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.axhline(y=0.0, color='black', linestyle='dotted')
    plt.axhline(y=strong_thresh, color='green', linestyle='dotted')
    plt.axhline(y=risk_thresh, color='red', linestyle='dotted')
    # Add labels for the horizontal lines
    #plt.text(x=0, y=strong_thresh, s=f'strong_thresh={strong_thresh}', color='red', va='bottom')
    #plt.text(x=0, y=risk_thresh, s=f'risk_thresh={risk_thresh}', color='blue', va='bottom')
    plt.ylabel('ALC')
    plt.xlabel('Attack Precision')
    plt.title(attack_name)
    plt.tight_layout()

    # Add colorbar
    norm = plt.Normalize(df['attack_recall'].min(), df['attack_recall'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=scatter, orientation='vertical', label='Attack Recall')

    plt.savefig(file_path)
    plt.close()

def plot_alc_best(df: pd.DataFrame,
                  strong_thresh: float, risk_thresh: float,
                  attack_name: str,
                  file_path: str) -> None:
    if len(df) < 10:
        return
    idx = df.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df_best = df.loc[idx].reset_index(drop=True)
    _plot_alc(df_best, strong_thresh, risk_thresh, attack_name, file_path)

def plot_alc(df: pd.DataFrame,
             strong_thresh: float, risk_thresh: float,
             attack_name: str,
             file_path: str) -> None:
    if len(df) < 10:
        return
    _plot_alc(df, strong_thresh, risk_thresh, attack_name, file_path)

def _plot_alc_line(df: pd.DataFrame,
             strong_thresh: float, risk_thresh: float,
             attack_name: str,
             file_path: str) -> None:
    df = df.copy()
    df['alc'] = df['alc'].apply(lambda x: max(x, -3.0))
    df_sorted = df.sort_values(by='alc', ascending=True).reset_index(drop=True)
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_sorted, x=df_sorted.index, y='alc')
    low_alc = df_sorted['alc'].min()
    lower_ylim = min(-0.05, low_alc)
    plt.ylim(lower_ylim, 1.05)
    plt.xlabel('')
    plt.title(attack_name)
    plt.xticks([])
    plt.axhline(y=0.0, color='black', linestyle='dotted')
    plt.axhline(y=strong_thresh, color='green', linestyle='dotted')
    plt.axhline(y=risk_thresh, color='red', linestyle='dotted')
    # Add labels for the horizontal lines
    #plt.text(x=0, y=risk_thresh, s=f'risk={risk_thresh}', color='red', va='bottom')
    #plt.text(x=0, y=strong_thresh, s=f'poor={strong_thresh}', color='blue', va='bottom')
    plt.ylabel('ALC')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def _plot_alc(df: pd.DataFrame,
             strong_thresh: float, risk_thresh: float,
             attack_name,
             file_path: str) -> None:
    df = df.copy()
    df['alc'] = df['alc'].apply(lambda x: max(x, -3.0))
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='alc', y='secret_column', orient='h')
    
    low_alc = df['alc'].min()
    lower_ylim = min(-0.05, low_alc)
    plt.xlim(lower_ylim, 1.05)
    
    plt.axvline(x=0.0, color='black', linestyle='dotted')
    plt.axvline(x=strong_thresh, color='green', linestyle='dotted')
    plt.axvline(x=risk_thresh, color='red', linestyle='dotted')
    
    # Add labels for the vertical lines
    #plt.text(x=risk_thresh, y=0, s=f'risk={risk_thresh}', color='red', va='bottom')
    #plt.text(x=strong_thresh, y=0, s=f'poor={strong_thresh}', color='blue', va='bottom')
    
    plt.xlabel('ALC')
    plt.title(attack_name)
    plt.ylabel('Secret Column')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def print_example_attack(df: pd.DataFrame) -> None:
    string = ''
    for _, row in df.iterrows():
        string += f"ALC: {round(row['alc'],2)}, base (prec: {round(row['base_prec'],2)}, recall: {round(row['base_recall'],2)}), attack (prec: {round(row['attack_prec'],2)}, recall: {round(row['attack_recall'],2)})\n"
        string +=f"    Secret: {row['secret_column']}, Known: {row['known_columns']}\n"
    return string

def make_text_summary(df_secret_known: pd.DataFrame,
                      strong_thresh: float,
                      risk_thresh: float,
                      all_secret_columns,
                      all_known_columns,
                      attack_name) -> str:
    df = df_secret_known.sort_values(by='alc', ascending=False)
    idx = df.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
    df = df.loc[idx].reset_index(drop=True)
    # Here, df has the per-secret/known combination with the highest ALC
    total_analyzed_combinations = len(df)
    total_no_anonymity_loss = len(df[df['alc'] <= 0.0])
    total_strong_anonymity = len(df[(df['alc'] > 0.0) & (df['alc'] <= strong_thresh)])
    total_at_risk = len(df[(df['alc'] > strong_thresh) & (df['alc'] <= risk_thresh)])
    total_poor_anonymity = len(df[df['alc'] > risk_thresh])
    total_no_anonymity = len(df[df['alc'] > 0.99])
    if total_strong_anonymity + total_at_risk + total_poor_anonymity == 0:
        anonymity_level = "VERY STRONG"
        note  = "Consider reducing the strength of the anonymization so as to improve data quality."
    elif total_poor_anonymity + total_at_risk == 0:
        anonymity_level = "STRONG"
        if total_strong_anonymity/(total_strong_anonymity+total_no_anonymity_loss) < 0.2:
            note  = "May consider reducing the strength of the anonymization so as to improve data quality."
        else:
            note  = "If data quality is poor, may consider reducing the strength of the anonymization so as to improve data quality."
    elif total_poor_anonymity == 0:
        if total_at_risk/(total_no_anonymity_loss+total_strong_anonymity+total_at_risk) < 0.05 or total_at_risk < 5:
            anonymity_level = "MINOR AT RISK"
            note = f"{total_at_risk} attacks ({round(100*(total_at_risk/total_analyzed_combinations),1)}%) may be at risk. Examine attacks to assess risk."
        else:
            anonymity_level = "MAJOR AT RISK"
            note = f"{total_at_risk} attacks ({round(100*(total_at_risk/total_analyzed_combinations),1)}%) may be at risk. Consider strengthening anonymity."
    else:
        if total_poor_anonymity/total_analyzed_combinations < 0.05 or total_poor_anonymity < 5:
            anonymity_level = "POOR"
            note = f"{total_poor_anonymity} attacks ({round(100*(total_poor_anonymity/total_analyzed_combinations),1)}%) have poor or no anonymity. Probably anonymity needs to be strengthened."
        else:
            anonymity_level = "VERY POOR"
            note = f"{total_poor_anonymity} attacks ({round(100*(total_poor_anonymity/total_analyzed_combinations),1)}%) have poor or no anonymity. Strengthen anonymity."
    summary = ""
    summary += "Anonymity Loss Coefficient Summary\n"
    if len(attack_name) > 0:
        summary += f"    {attack_name}\n"
    summary += f"Anonymity Level: {anonymity_level}\n"
    summary += f"    {note}\n\n"
    summary += f"{len(all_secret_columns)} columns used as secret columns:\n"
    for column in all_secret_columns:
        summary += f"  {column}\n"
    summary += "\n"
    summary += f"{len(all_known_columns)} columns used as known columns:\n"
    for column in all_known_columns:
        summary += f"  {column}\n"
    summary += "\n"
    width = len("Perfect anonymity")
    summary += f"Analyzed known column / secret column combinations: {total_analyzed_combinations}\n"
    string = "Perfect anonymity"
    summary += f"{string:>{width}}: {total_no_anonymity_loss:>5} ({round(100*(total_no_anonymity_loss/total_analyzed_combinations),1)}%)\n"
    string = "Strong anonymity"
    summary += f"{string:>{width}}: {total_strong_anonymity:>5} ({round(100*(total_strong_anonymity/total_analyzed_combinations),1)}%)\n"
    string = "At risk"
    summary += f"{string:>{width}}: {total_at_risk:>5} ({round(100*(total_at_risk/total_analyzed_combinations),1)}%)\n"
    string = "Poor anonymity"
    summary += f"{string:>{width}}: {total_poor_anonymity:>5} ({round(100*(total_poor_anonymity/total_analyzed_combinations),1)}%)\n"
    string = "No anonymity"
    summary += f"{string:>{width}}: {total_no_anonymity:>5} ({round(100*(total_no_anonymity/total_analyzed_combinations),1)}%)\n"
    summary += "\n"
    if total_no_anonymity > 0:
        summary += "Examples of complete anonymity loss:\n"
        filtered_df = df[df['alc'] > 0.99]
        summary += print_example_attack(filtered_df.head(5))
    elif total_poor_anonymity > 0:
        summary += "Examples of poor anonymity loss:\n"
        filtered_df = df[df['alc'] > risk_thresh]
        summary += print_example_attack(filtered_df.head(5))
    elif total_at_risk > 0:
        summary += "Examples of at risk anonymity loss:\n"
        filtered_df = df[df['alc'] > strong_thresh]
        summary += print_example_attack(filtered_df.head(5))
    return summary
