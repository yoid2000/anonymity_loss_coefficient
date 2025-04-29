import os
import pandas as pd
import seaborn as sns
import matplotlib
import logging
import json
from typing import List, Optional, Dict
matplotlib.use('Agg')  # This needed because of tkinter issues
import matplotlib.pyplot as plt
from .defaults import defaults

class Reporter():
    def __init__(self,
                 results_path: str,
                 attack_name: str,
                 logger: logging.Logger,
                 flush: bool = defaults['flush'],
                 ) -> None:
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)
        self.attack_name = attack_name
        self.logger = logger
        self.all_used_known_columns = []
        self.all_used_secret_columns = []
        self.list_results = []
        self.list_results_done = []
        self.list_secret_known_results_done = []
        self.df_secret_results = None
        summary_raw_path = os.path.join(self.results_path, 'summary_raw.parquet')
        summary_secret_known_path = os.path.join(self.results_path, 'summary_secret_known.csv')
        self.df_already_attacked = None
        if flush:
            self._remove_file(summary_raw_path)
            self._remove_file(summary_secret_known_path)
        else:
            # read the summary_raw_path if it exists and convert to a list of dicts
            if os.path.exists(summary_raw_path):
                if not os.path.exists(summary_secret_known_path):
                    # throw an excption
                    raise Exception(f"Either both {summary_raw_path} and {summary_secret_known_path} must exist or neither must exist.")
                df = pd.read_parquet(summary_raw_path)
                self.logger.info(f"Reading {summary_raw_path}. Found {len(df)} rows.")
                self.list_results_done = df.to_dict(orient='records')
            if os.path.exists(summary_secret_known_path):
                if not os.path.exists(summary_raw_path):
                    # throw an excption
                    raise Exception(f"Either both {summary_raw_path} and {summary_secret_known_path} must exist or neither must exist.")
                df = pd.read_csv(summary_secret_known_path)
                self.list_secret_known_results_done = df.to_dict(orient='records')
                self.logger.info(f"Reading {summary_secret_known_path}. Found {len(df)} rows.")
                # get every distinct combination of secret_column and known_columns
                self.df_already_attacked = df[['secret_column', 'known_columns']].drop_duplicates()
                self.df_already_attacked = self.df_already_attacked.reset_index(drop=True)
                self.logger.info(f"Found {len(self.df_already_attacked)} already attacked secret_column / known_columns combinations.")

    def add_result(self, row: dict) -> None:
        # Reset the results dataframes because they will be out of date after this add
        self.list_results.append(row)

    def already_attacked(self,
                       secret_column: str,
                       known_columns: List[str],
                       ) -> bool:
        if self.df_already_attacked is None:
            return False
        known_columns_str = self._make_known_columns_str(known_columns)
        count = self.df_already_attacked[
                (self.df_already_attacked['secret_column'] == secret_column) &
                (self.df_already_attacked['known_columns'] == known_columns_str)
                 ].shape[0]
        if count > 0:
            return True
        return False

    def add_known_columns(self, known_columns: List[str]) -> None:
        self.all_used_known_columns += known_columns
        self.all_used_known_columns = sorted(list(set(self.all_used_known_columns)))

    def add_secret_column(self, secret_column: str) -> None:
        self.all_used_secret_columns.append(secret_column)
        self.all_used_secret_columns = sorted(list(set(self.all_used_secret_columns)))

    def get_results_df(self,
                       known_columns: Optional[List[str]] = None,
                       secret_column: Optional[str] = None) -> pd.DataFrame:
        df_results = pd.DataFrame(self.list_results_done)
        return self._filter_df(df_results, known_columns, secret_column)

    def alc_per_secret_and_known_df(self,
                                 known_columns: Optional[List[str]] = None,
                                 secret_column: Optional[str] = None) -> pd.DataFrame:
        df_secret_known_results = pd.DataFrame(self.list_secret_known_results_done)
        return self._filter_df(df_secret_known_results, known_columns, secret_column)

    def _alc_per_secret_and_known(self, score_info: List[Dict]) -> List[Dict]:
        # self.list_results contains the results of the latest attack only
        df_in = pd.DataFrame(self.list_results)
        df_in['prediction'] = df_in['predicted_value'] == df_in['true_value']
        rows = []
        base_group = df_in[df_in['predict_type'] == 'base']
        attack_group = df_in[df_in['predict_type'] == 'attack']
        base_count = len(base_group)
        attack_count = len(attack_group)
        num_known_columns = df_in['num_known_columns'].iloc[0]
        # Note that known_columns is a string at this point
        secret_column = df_in['secret_column'].iloc[0]
        known_columns = df_in['known_columns'].iloc[0]
        for score in score_info:
            score['secret_column'] = secret_column
            score['known_columns'] = known_columns
            score['num_known_columns'] = num_known_columns
            score['base_count'] = base_count
            score['attack_count'] = attack_count
            rows.append(score)
        return rows


    def _filter_df(self, df: pd.DataFrame,
                  known_columns: Optional[List[str]] = None,
                  secret_column: Optional[str] = None) -> pd.DataFrame:
        if known_columns is not None:
            known_columns_str = self._make_known_columns_str(known_columns)
            df = df[df['known_columns'] == known_columns_str]
        if secret_column is not None:
            df = df[df['secret_column'] == secret_column]
        return df

    def _remove_file(self, file_path: str) -> None:
        self.logger.info(f"Flushing {file_path}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                self.logger.warning(f"Warning: The file at {file_path} is currently open in another application. You might want to make a copy of the file in order to view it while the attack is still executing.")
            except Exception as e:
                self.logger.error(f"Error: Failed to delete {file_path}: {e}")

    def _make_known_columns_str(self, known_columns: List[str]) -> str:
        return json.dumps(sorted(known_columns))

    def consolidate_results(self, score_info: List[Dict]) -> None:
        if len(self.list_results) == 0:
            self.logger.warning("Warning: No results to consolidate.")
            return
        # move the results from the list to a dataframe
        list_secret_known_results = self._alc_per_secret_and_known(score_info)
        # At this point, self.list_results, and list_secret_known_results
        # contain the results of the latest attack only
        self.list_results_done += self.list_results
        self.list_results = []
        self.list_secret_known_results_done += list_secret_known_results

    def summarize_results(self,
                          strong_thresh: float = 0.5,
                          risk_thresh: float = 0.7,
                          with_text: bool = True,
                          with_plot: bool = True,
                          ) -> bool:
        if len(self.list_results_done) == 0:
            self.logger.warning("Warning: No results to summarize.")
            return False
        df_results = pd.DataFrame(self.list_results_done)
        df_secret_known_results = pd.DataFrame(self.list_secret_known_results_done)
        if len(df_results) > 0:
            self.save_to_parquet(self.results_path, df_results, 'summary_raw.parquet')
        if len(df_secret_known_results) > 0:
            self.save_to_csv(self.results_path, df_secret_known_results, 'summary_secret_known.csv')
            if with_text:
                text_summary = make_text_summary(df_secret_known_results,
                                                strong_thresh,
                                                risk_thresh,
                                                self.all_used_secret_columns,
                                                self.all_used_known_columns,
                                                self.attack_name)
                self.save_to_text(self.results_path, text_summary, 'summary.txt')
            if with_plot:
                plot_alc(df_secret_known_results,
                            strong_thresh,
                            risk_thresh,
                            self.attack_name,
                            os.path.join(self.results_path, 'alc_plot.png'))
                plot_alc_prec(df_secret_known_results,
                            strong_thresh,
                            risk_thresh,
                            self.attack_name,
                            os.path.join(self.results_path, 'alc_prec_plot.png'))
                plot_alc_best(df_secret_known_results,
                            strong_thresh,
                            risk_thresh,
                            self.attack_name,
                            os.path.join(self.results_path, 'alc_plot_best.png'))
                plot_alc_prec_best(df_secret_known_results,
                            strong_thresh,
                            risk_thresh,
                            self.attack_name,
                            os.path.join(self.results_path, 'alc_prec_plot_best.png'))
        return True


    def save_to_text(self, results_path: str, text_summary: str, file_name: str) -> None:
        save_path = os.path.join(results_path, file_name)
        try:
            with open(save_path, 'w') as f:
                f.write(text_summary)
        except PermissionError:
            self.logger.warning(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the summary file in order to view it while the attack is still executing.")
        except Exception as e:
            self.logger.error(f"Error: Failed to write {save_path}: {e}")

    def save_to_parquet(self, results_path, df: pd.DataFrame, file_name: str) -> None:
        save_path = os.path.join(results_path, file_name)
        try:
            df.to_parquet(save_path, index=False)
        except PermissionError:
            self.logger.warning(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the file in order to view it while the attack is still executing.")
        except Exception as e:
            self.logger.error(f"Error: Failed to write {save_path}: {e}")

    def save_to_csv(self, results_path, df: pd.DataFrame, file_name: str) -> None:
        save_path = os.path.join(results_path, file_name)
        try:
            df.to_csv(save_path, index=False)
        except PermissionError:
            self.logger.warning(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the summary file in order to view it while the attack is still executing.")
        except Exception as e:
            self.logger.error(f"Error: Failed to write {save_path}: {e}")
    

def clean_up_results_files(results_path: str) -> None:
    pass

def plot_alc_prec_best(df: pd.DataFrame,
                  strong_thresh: float, risk_thresh: float,
                  attack_name: str,
                  file_path: str) -> None:
    if len(df) < 10:
        return
    df_best = df[df['paired'] == False].copy()
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
    # sort the dataframe by 'attack_recall' ascending
    df = df.sort_values(by='attack_recall', ascending=True).reset_index(drop=True)
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
    df_best = df[df['paired'] == False].copy()
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

    # Determine the number of unique categories in the y-axis (secret_column)
    num_categories = df['secret_column'].nunique()
    
    # Dynamically adjust the figure height based on the number of categories
    figure_height = max(2, num_categories * 0.2)
    
    plt.figure(figsize=(6, figure_height))

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
