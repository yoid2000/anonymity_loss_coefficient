import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_alc_prec(df: pd.DataFrame,
                  strong_thresh: float, risk_thresh: float,
                  file_path: str) -> None:
    if len(df) < 10:
        return
    df = df.copy()
    # set any 'alc' values less that -3.0 to -3.0
    df['alc'] = df['alc'].apply(lambda x: max(x, -3.0))
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='attack_prec', y='alc')
    low_alc = df['alc'].min()
    lower_ylim = min(-0.05, low_alc)
    plt.ylim(lower_ylim, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.axhline(y=0.0, color='black', linestyle='dotted')
    plt.axhline(y=strong_thresh, color='green', linestyle='dotted')
    plt.axhline(y=risk_thresh, color='red', linestyle='dotted')
    #plt.text(x=0, y=strong_thresh, s=f'strong_thresh={strong_thresh}', color='red', va='bottom')
    #plt.text(x=0, y=risk_thresh, s=f'risk_thresh={risk_thresh}', color='blue', va='bottom')
    plt.ylabel('ALC')
    plt.xlabel('Attack Precision')
    plt.tight_layout()
    plt.savefig(file_path)

def plot_alc(df: pd.DataFrame,
             strong_thresh: float, risk_thresh: float,
             file_path: str) -> None:
    if len(df) < 10:
        return
    df = df.copy()
    df['alc'] = df['alc'].apply(lambda x: max(x, -3.0))
    df_sorted = df.sort_values(by='alc', ascending=True).reset_index(drop=True)
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_sorted, x=df_sorted.index, y='alc')
    low_alc = df_sorted['alc'].min()
    lower_ylim = min(-0.05, low_alc)
    plt.ylim(lower_ylim, 1.05)
    plt.xlabel('')
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

def print_example_attack(df: pd.DataFrame) -> None:
    string = ''
    for _, row in df.iterrows():
        string += f"ALC: {round(row['alc'],2)}, base (prec: {round(row['base_prec'],2)}, recall: {round(row['base_recall'],2)}), attack (prec: {round(row['attack_prec'],2)}, recall: {round(row['attack_recall'],2)})\n"
        string +=f"    Secret: {row['target_column']}, Known: {row['known_columns']}\n"
    return string

def make_text_summary(df: pd.DataFrame,
                      strong_thresh: float,
                      risk_thresh: float,
                      all_target_columns,
                      all_known_columns) -> str:
    # sort by alc descending
    df = df.sort_values(by='alc', ascending=False)
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
    summary += "Anonymity Loss Coefficient Summary\n\n"
    summary += f"Anonymity Level: {anonymity_level}\n"
    summary += f"    {note}\n\n"
    summary += f"{len(all_target_columns)} columns used as targeted columns:\n"
    for column in all_target_columns:
        summary += f"  {column}\n"
    summary += "\n"
    summary += f"{len(all_known_columns)} columns used as known columns:\n"
    for column in all_known_columns:
        summary += f"  {column}\n"
    summary += "\n"
    width = len("Perfect anonymity")
    summary += f"Analyzed known column / target column combinations: {total_analyzed_combinations}\n"
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
