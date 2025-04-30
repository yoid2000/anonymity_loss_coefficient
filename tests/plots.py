import numpy as np
import os
import matplotlib.pyplot as plt
from alc import AnonymityLossCoefficient
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pprint
pp = pprint.PrettyPrinter(indent=4)

plots_path = os.path.join('alc_plots')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

def savefigs(plt, name):
    for suffix in ['.png', '.pdf']:
        path_name = name + suffix
        out_path = os.path.join(plots_path, path_name)
        plt.savefig(out_path)
    plt.close()

def plot_recall_adjust(out_name):
    alc = AnonymityLossCoefficient()
    ranges = [[0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    recall_values = np.concatenate(arrays)
    strength_vals = [1.5, 2.0, 3.0]

    fig, ax1 = plt.subplots(figsize=((8, 5)))

    for n in strength_vals:
        alc.set_param('recall_adjust_strength', n)
        adj_values = [alc._recall_adjust(recall) for recall in recall_values]
        ax1.scatter(recall_values, adj_values, label=f'recall_adjust_strength = {n}', s=5)

    ax1.set_xscale('log')  # Set the scale of the second x-axis to logarithmic
    ax1.set_xlabel('Recall (Log Scale)', fontsize=12)
    ax1.set_ylabel('Adjustment', fontsize=12)

    ax2 = ax1.twiny()  # Create a second x-axis
    #ax2.set_xscale('log')  # Set the scale of the second x-axis to logarithmic

    for n in strength_vals:
        alc.set_param('recall_adjust_strength', n)
        adj_values = [alc._recall_adjust(recall) for recall in recall_values]
        ax2.scatter(recall_values, adj_values, label=f'recall_adjust_strength = {n}', s=5)

    ax2.set_xlabel('Recall (Linear Scale)', fontsize=12)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    savefigs(plt, out_name)

def plot_base_adjusted_prc(out_name):
    alc = AnonymityLossCoefficient()
    increase_values = [0.2, 0.5, 0.8, 0.98]
    prc_base_values = np.linspace(0, 0.999, 1000)
    fig, ax = plt.subplots(figsize=((8, 5)))
    # For each increase value, calculate prc_attack and prc_adj for each prc_base and plot the results
    for increase in increase_values:
        prc_attack_values = prc_base_values + increase * (1.0 - prc_base_values)
        prc_adj_values = [alc._prc_improve(prc_base, prc_attack) for prc_base, prc_attack in zip(prc_base_values, prc_attack_values)]
        ax.plot(prc_base_values, prc_adj_values, label=f'Improvement = {increase}')

    # Add labels and a legend
    ax.set_ylim(0, 1)
    ax.set_xlabel('Base PRC', fontsize=12)
    ax.set_ylabel('Attack PRC', fontsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.25))

    # Create an inset axes in the upper left corner of the current axes
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=2, borderpad=4)

    # Plot the same data on the inset axes with the specified x-axis range
    for increase in increase_values:
        prc_attack_values = prc_base_values + increase * (1.0 - prc_base_values)
        prc_adj_values = [alc._prc_improve(prc_base, prc_attack) for prc_base, prc_attack in zip(prc_base_values, prc_attack_values)]
        ax_inset.plot(prc_base_values, prc_adj_values)

    # Set the x-axis range of the inset axes
    ax_inset.set_xlim(0.94, 1.0)
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_identical_recall(out_name, limit=1.0):
    ''' In this plot, we hold the precision improvement of the attack over the base constant, and given both attack and base identical recall. We find that the
    constant precision improvement puts an upper bound on the ALC. We also find that
    the recall also places an upper bound.
    '''
    alc = AnonymityLossCoefficient()
    recall_values = np.logspace(np.log10(0.0001), np.log10(1), 5000)
    p_base_values = np.random.uniform(0, limit, len(recall_values))

    # Run several different relative improvements between base and attack
    increase_values = [0.2, 0.5, 0.8, 0.98]
    plt.figure(figsize=((8, 5)))
    for increase in increase_values:
        p_attack_values = p_base_values + (increase * (1.0 - p_base_values))
        scores = [alc.alc(p_base=p_base_value, r_base=recall_value, p_attack=p_attack_value, r_attack=recall_value) for p_base_value, recall_value, p_attack_value, recall_value in zip(p_base_values, recall_values, p_attack_values, recall_values)]
        plt.scatter(recall_values, scores, label=f'precision increase = {increase}', s=2)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel(f'Recall (base precision limit = {limit})', fontsize=12)
    plt.ylabel('Anonymity Loss Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    savefigs(plt, out_name)

def prec_from_fscore_recll(fscore, recll, beta):
    if recll == 0:
        return 0  # Avoid division by zero
    beta_squared = beta ** 2
    precision = (fscore * recll) / (((1+beta_squared) * recll) - (fscore * beta_squared))
    return precision

def compute_fscore(prec, recll, beta):
    if prec == 0 and recll == 0:
        return 0  # Avoid division by zero
    beta_squared = beta ** 2
    fscore = (1 + beta_squared) * (prec * recll) / (beta_squared * prec + recll)
    return fscore

def plot_fscore_prec_for_equal_recall(out_name, beta=0.1):
    reclls = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    prec_values = np.random.uniform(0.01, 1, 5000)

    plt.figure(figsize=((6, 3.5)))
    for recll in reclls:
        fscore_values = [compute_fscore(prec, recll, beta) for prec in prec_values]
        plt.scatter(prec_values, fscore_values, label=f'recll = {recll}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel(f'Fscore (beta = {beta})', fontsize=12)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower left')
    plt.tight_layout()
    savefigs(plt, out_name)


def plot_prec_recall_for_equal_fscore(out_name, beta=1.001):
    ''' The purpose of this plot is to see how different values of prec
        and recall can have the same prc.
    '''
    print(f'for beta = {beta}, fscore = 0.5, and recll 0.5, prec = {prec_from_fscore_recll(0.5, 0.5, beta)}')
    fscores = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    recall_base_values = np.logspace(np.log10(0.0001), np.log10(1), 10000)

    plt.figure(figsize=((6, 3.5)))
    for fscore in fscores:
        prec_values = [prec_from_fscore_recll(fscore, recall_value, beta) for recall_value in recall_base_values]
        prec_recall_pairs = [(prec, recall) for prec, recall in zip(prec_values, recall_base_values)]
        prec_recall_pairs = sorted(prec_recall_pairs, key=lambda x: x[0])
        #prec_recall_pairs = [(prec, recall) for prec, recall in prec_recall_pairs if prec <= 1.0]
        prec_values, recall_values = zip(*prec_recall_pairs)
        plt.scatter(recall_values, prec_values, label=f'Fscore = {fscore}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel(f'Recall (beta = {beta})', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower left')
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_prec_recall_for_equal_prc(out_name):
    ''' The purpose of this plot is to see how different values of prec
        and recall can have the same prc.
    '''
    alc = AnonymityLossCoefficient()
    alpha = alc.get_param('recall_adjust_strength')
    print(f'for prc = 0.5, and prec 1.0, recall = {alc.recall_from_prc_prec(0.5, 1.0)}')
    print(f'for prc = 0.5, and recall 0.001484, prec = {alc.prec_from_prc_recall(0.5, 0.001484)}')
    print(f'for prc = 0.5, and prec 0.6, recall = {alc.recall_from_prc_prec(0.5, 0.6)}')
    print(f'for prc = 0.5, and recall 0.0233, prec = {alc.prec_from_prc_recall(0.5, 0.0233)}')
    print(f'for prc = 0.5, and prec 0.5, recall = {alc.recall_from_prc_prec(0.5, 0.5)}')
    print(f'for prc = 0.5, and recall 1.0, prec = {alc.prec_from_prc_recall(0.5, 1.0)}')
    prc_vals = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    recall_base_values = np.concatenate(arrays)

    plt.figure(figsize=((6, 3.5)))
    for prc_val in prc_vals:
        prec_values = [alc.prec_from_prc_recall(prc_val, recall_value) for recall_value in recall_base_values]
        prec_recall_pairs = [(prec, recall) for prec, recall in zip(prec_values, recall_base_values)]
        prec_recall_pairs = sorted(prec_recall_pairs, key=lambda x: x[0])
        prec_recall_pairs = [(prec, recall) for prec, recall in prec_recall_pairs if prec <= 1.0]
        prec_values, recall_values = zip(*prec_recall_pairs)
        plt.scatter(recall_values, prec_values, label=f'PRC = {prc_val}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.text(0.05, 0.98, f'alpha = {alpha}, Cmin = 0.0001', ha='left', va='top', fontsize=9, transform=plt.gca().transAxes)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower center')
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_varying_base_recall(out_name):
    ''' The purpose of this plot is to see the effect of having a different
        base recall than attack recall. We vary the base recall from 1/10K to 1 while keeping all other parameters constant. What this shows is that the ALC varies substantially when the recall values are not similar.
    '''
    alc = AnonymityLossCoefficient()
    recall_values = np.logspace(np.log10(0.0001), np.log10(1), 5000)
    p_base = 0.5
    r_attack = 0.01

    # Run several different relative improvements between base and attack
    increase_values = [0.2, 0.5, 0.8, 0.98]
    plt.figure(figsize=((8, 5)))
    for increase in increase_values:
        p_attack = p_base + (increase * (1.0 - p_base))
        scores = [alc.alc(p_base=p_base, r_base=recall_value, p_attack=p_attack, r_attack=r_attack) for recall_value in recall_values]
        plt.scatter(recall_values, scores, label=f'precision increase = {increase}', s=2)
    plt.xscale('log')
    plt.axvline(x=0.01, color='black', linestyle='dashed')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel(f'Base Recall (Attack Recall = {r_attack})', fontsize=12)
    plt.ylabel('Anonymity Loss Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    savefigs(plt, out_name)

def run_prc_checks(alc, p_base, r_base, prc_base):
    if r_base <= 0.0001:
        return
    p_base_test = round(alc.prec_from_prc_recall(prc_base, r_base),3)
    if round(p_base,3) != p_base_test:
        print(f'Error: prec_from_prc_recall({prc_base}, {r_base})')
        print(f'Expected: {round(p_base,3)}, got: {p_base_test}')
        quit()
    r_base_test = round(alc.recall_from_prc_prec(prc_base, p_base),3)
    if round(r_base,3) != r_base_test:
        print(f'Error: recall_from_prc_prec({prc_base}, {p_base})')
        print(f'Expected: {round(r_base,3)}, got: {r_base_test}')
        quit()

def do_alc_test(alc, p_base, r_base, increase, r_attack):
    print('------------------------------------')
    p_attack = p_base + increase * (1.0 - p_base)
    print(f'Base precision: {p_base}, base recall: {r_base}\nattack precision: {p_attack}, attack recall: {r_attack}')
    print(f'prec increase: {increase}')
    prc_atk = alc.prc(prec=p_attack, recall=r_attack)
    print(f'prc_atk: {prc_atk}')
    prc_base = alc.prc(prec=p_base, recall=r_base)
    print(f'prc_base: {prc_base}')
    run_prc_checks(alc, p_base, r_base, prc_base)
    print(f'ALC: {round(alc.alc(p_base=p_base, r_base=r_base, p_attack=p_attack, r_attack=r_attack),3)}')

def make_alc_plots(recall_adjust_strength=3.0, pairs='v3'):
    alc = AnonymityLossCoefficient()
    alc.set_param('recall_adjust_strength', recall_adjust_strength)
    if pairs == 'v1':
        Catk_Cbase_pairs = [(1, 1), (0.01, 0.01), (0.7, 1.0), (0.01, 0.05)]
        fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    elif pairs == 'v2':
        Catk_Cbase_pairs = [(1, 1), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001)]
        fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    elif pairs == 'v3':
        Catk_Cbase_pairs = [(1, 1), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001), (0.075, 0.1), (0.05, 0.01)]
        fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    Pbase_values = [0.01, 0.1, 0.4, 0.7, 0.9]
    Patk = np.arange(0, 1.01, 0.01)
    
    axs = axs.flatten()
    
    for i, (Catk, Cbase) in enumerate(Catk_Cbase_pairs):
        for Pbase in Pbase_values:
            ALC = [alc.alc(p_base=Pbase, r_base=Cbase, p_attack=p, r_attack=Catk) for p in Patk]
            axs[i].plot(Patk, ALC, label=f'Pbase={Pbase}')
        
        axs[i].text(0.05, 0.95, f'Catk = {Catk}, Cbase = {Cbase}\nalpha = {recall_adjust_strength}\nCmin = 0.0001', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(-0.5, 1)
        
        # Remove x-axis labels and ticks for the upper two subplots
        if i < len(Catk_Cbase_pairs) - 2:
            axs[i].set_xlabel('')
            #axs[i].set_xticklabels([])
        
        # Remove y-axis labels and ticks for the right subplots
        if i % 2 == 1:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])
        
        if i % 2 == 0:
            axs[i].set_ylabel('ALC')
        
        if i >= len(Catk_Cbase_pairs) - 2:
            axs[i].set_xlabel('Patk')
        
        axs[i].legend(fontsize='small', loc='lower right')
        axs[i].grid(True)
    
    plt.tight_layout()
    
    # Save the plot in both PNG and PDF formats
    plt.savefig(f'alc_plots/alc_plot_{recall_adjust_strength}_{pairs}.png')
    plt.savefig(f'alc_plots/alc_plot_{recall_adjust_strength}_{pairs}.pdf')

alc = AnonymityLossCoefficient()
do_alc_test(alc, p_base=0.5, r_base=1.0, increase=0.2, r_attack=1.0)
do_alc_test(alc, p_base=0.2, r_base=1.0, increase=0.8, r_attack=1.0)
do_alc_test(alc, p_base=0.999, r_base=1.0, increase=0.9, r_attack=1.0)
do_alc_test(alc, p_base=0.5, r_base=0.1, increase=0.2, r_attack=0.1)
do_alc_test(alc, p_base=0.2, r_base=0.1, increase=0.8, r_attack=0.1)
do_alc_test(alc, p_base=0.5, r_base=0.01, increase=0.2, r_attack=0.01)
do_alc_test(alc, p_base=0.2, r_base=0.01, increase=0.8, r_attack=0.01)
do_alc_test(alc, p_base=0.5, r_base=0.001, increase=0.2, r_attack=0.001)
do_alc_test(alc, p_base=0.2, r_base=0.001, increase=0.8, r_attack=0.001)
do_alc_test(alc, p_base=0.5, r_base=0.0001, increase=0.2, r_attack=0.0001)
do_alc_test(alc, p_base=0.2, r_base=0.0001, increase=0.8, r_attack=0.0001)
do_alc_test(alc, p_base=1.0, r_base=0.00001, increase=0, r_attack=0.00001)
plot_prec_recall_for_equal_prc('prec_recall_for_equal_prc')
make_alc_plots(pairs='v3')
for recall_adjust_strength in [1.0, 2.0, 3.0, 4.0]:
    make_alc_plots(recall_adjust_strength=recall_adjust_strength, pairs='v1')
    make_alc_plots(recall_adjust_strength=recall_adjust_strength, pairs='v2')
plot_varying_base_recall('varying_base_recall')
plot_prec_recall_for_equal_fscore('prec_recall_for_equal_fscore')
plot_fscore_prec_for_equal_recall('fscore_prec_for_equal_recall')
plot_identical_recall('identical_recall')
plot_identical_recall('identical_recall_limit', limit=0.5)
plot_recall_adjust('recall_adjust')
plot_base_adjusted_prc('base_adjusted_prc')