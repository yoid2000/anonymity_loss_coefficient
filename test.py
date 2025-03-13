import numpy as np
import os
import matplotlib.pyplot as plt
import anonymity_loss_coefficient
from mpl_toolkits.mplot3d import Axes3D
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

def plot_abs_weights(out_name):
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    pb_values = np.concatenate([np.arange(0, 0.99, 0.01), np.arange(0.991, 0.999, 0.001)])
    weight_values = [0.5, 1.0, 2.0]
    plt.figure(figsize=((8, 5)))
    for weight in weight_values:
        alc.set_param('pcc_abs_weight_strength', weight)
        abs_weight_values = [alc._get_pcc_abs_weight(pb) for pb in pb_values]
        plt.plot(pb_values, abs_weight_values, label=f'pcc_abs_weight_strength = {weight}')
    plt.xlim(0.5, 1.0)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel('PCC', fontsize=12)
    plt.ylabel('abs_weight', fontsize=12)
    plt.legend()
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_cov_adjust(out_name):
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    ranges = [[0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    cov_values = np.concatenate(arrays)
    strength_vals = [1.5, 2.0, 3.0]

    fig, ax1 = plt.subplots(figsize=((8, 5)))

    for n in strength_vals:
        alc.set_param('cov_adjust_strength', n)
        adj_values = [alc._cov_adjust(cov) for cov in cov_values]
        ax1.scatter(cov_values, adj_values, label=f'cov_adjust_strength = {n}', s=5)

    ax1.set_xscale('log')  # Set the scale of the second x-axis to logarithmic
    ax1.set_xlabel('COV (Log Scale)', fontsize=12)
    ax1.set_ylabel('Adjustment', fontsize=12)

    ax2 = ax1.twiny()  # Create a second x-axis
    #ax2.set_xscale('log')  # Set the scale of the second x-axis to logarithmic

    for n in strength_vals:
        alc.set_param('cov_adjust_strength', n)
        adj_values = [alc._cov_adjust(cov) for cov in cov_values]
        ax2.scatter(cov_values, adj_values, label=f'cov_adjust_strength = {n}', s=5)

    ax2.set_xlabel('COV (Linear Scale)', fontsize=12)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    savefigs(plt, out_name)

def plot_base_adjusted_pcc(out_name):
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    increase_values = [0.2, 0.5, 0.8, 0.98]
    pcc_base_values = np.linspace(0, 0.999, 1000)
    fig, ax = plt.subplots(figsize=((8, 5)))
    # For each increase value, calculate pcc_attack and pcc_adj for each pcc_base and plot the results
    for increase in increase_values:
        pcc_attack_values = pcc_base_values + increase * (1.0 - pcc_base_values)
        pcc_adj_values = [alc._pcc_improve(pcc_base, pcc_attack) for pcc_base, pcc_attack in zip(pcc_base_values, pcc_attack_values)]
        ax.plot(pcc_base_values, pcc_adj_values, label=f'Improvement = {increase}')

    # Add labels and a legend
    ax.set_ylim(0, 1)
    ax.set_xlabel('Base PCC', fontsize=12)
    ax.set_ylabel('Attack PCC', fontsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.25))

    # Create an inset axes in the upper left corner of the current axes
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=2, borderpad=4)

    # Plot the same data on the inset axes with the specified x-axis range
    for increase in increase_values:
        pcc_attack_values = pcc_base_values + increase * (1.0 - pcc_base_values)
        pcc_adj_values = [alc._pcc_improve(pcc_base, pcc_attack) for pcc_base, pcc_attack in zip(pcc_base_values, pcc_attack_values)]
        ax_inset.plot(pcc_base_values, pcc_adj_values)

    # Set the x-axis range of the inset axes
    ax_inset.set_xlim(0.94, 1.0)
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_identical_cov(out_name, limit=1.0):
    ''' In this plot, we hold the precision improvement of the attack over the base constant, and given both attack and base identical coverage. We find that the
    constant precision improvement puts an upper bound on the ALC. We also find that
    the coverage also places an upper bound.
    '''
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    cov_values = np.logspace(np.log10(0.0001), np.log10(1), 5000)
    p_base_values = np.random.uniform(0, limit, len(cov_values))

    # Run several different relative improvements between base and attack
    increase_values = [0.2, 0.5, 0.8, 0.98]
    plt.figure(figsize=((8, 5)))
    for increase in increase_values:
        p_attack_values = p_base_values + (increase * (1.0 - p_base_values))
        scores = [alc.alc(p_base=p_base_value, c_base=cov_value, p_attack=p_attack_value, c_attack=cov_value) for p_base_value, cov_value, p_attack_value, cov_value in zip(p_base_values, cov_values, p_attack_values, cov_values)]
        plt.scatter(cov_values, scores, label=f'precision increase = {increase}', s=2)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel(f'Coverage (base precision limit = {limit})', fontsize=12)
    plt.ylabel('Anonymity Loss Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    savefigs(plt, out_name)

def prec_from_fscore_recall(fscore, recall, beta):
    if recall == 0:
        return 0  # Avoid division by zero
    beta_squared = beta ** 2
    precision = (fscore * recall) / (((1+beta_squared) * recall) - (fscore * beta_squared))
    return precision

def compute_fscore(prec, recall, beta):
    if prec == 0 and recall == 0:
        return 0  # Avoid division by zero
    beta_squared = beta ** 2
    fscore = (1 + beta_squared) * (prec * recall) / (beta_squared * prec + recall)
    return fscore

def plot_fscore_prec_for_equal_cov(out_name, beta=0.1):
    recalls = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    prec_values = np.random.uniform(0.01, 1, 5000)

    plt.figure(figsize=((6, 3.5)))
    for recall in recalls:
        fscore_values = [compute_fscore(prec, recall, beta) for prec in prec_values]
        plt.scatter(prec_values, fscore_values, label=f'Recall = {recall}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel(f'Fscore (beta = {beta})', fontsize=12)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower left')
    plt.tight_layout()
    savefigs(plt, out_name)


def plot_prec_cov_for_equal_fscore(out_name, beta=1.001):
    ''' The purpose of this plot is to see how different values of prec
        and cov can have the same pcc.
    '''
    print(f'for beta = {beta}, fscore = 0.5, and recall 0.5, prec = {prec_from_fscore_recall(0.5, 0.5, beta)}')
    fscores = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    cov_base_values = np.logspace(np.log10(0.0001), np.log10(1), 10000)

    plt.figure(figsize=((6, 3.5)))
    for fscore in fscores:
        prec_values = [prec_from_fscore_recall(fscore, cov_value, beta) for cov_value in cov_base_values]
        prec_cov_pairs = [(prec, cov) for prec, cov in zip(prec_values, cov_base_values)]
        prec_cov_pairs = sorted(prec_cov_pairs, key=lambda x: x[0])
        #prec_cov_pairs = [(prec, cov) for prec, cov in prec_cov_pairs if prec <= 1.0]
        prec_values, cov_values = zip(*prec_cov_pairs)
        plt.scatter(cov_values, prec_values, label=f'Fscore = {fscore}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel(f'Coverage (beta = {beta})', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower left')
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_prec_cov_for_equal_pcc(out_name):
    ''' The purpose of this plot is to see how different values of prec
        and cov can have the same pcc.
    '''
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    alpha = alc.get_param('cov_adjust_strength')
    print(f'for pcc = 0.5, and prec 1.0, cov = {alc.cov_from_pcc_prec(0.5, 1.0)}')
    print(f'for pcc = 0.5, and cov 0.001484, prec = {alc.prec_from_pcc_cov(0.5, 0.001484)}')
    print(f'for pcc = 0.5, and prec 0.6, cov = {alc.cov_from_pcc_prec(0.5, 0.6)}')
    print(f'for pcc = 0.5, and cov 0.0233, prec = {alc.prec_from_pcc_cov(0.5, 0.0233)}')
    print(f'for pcc = 0.5, and prec 0.5, cov = {alc.cov_from_pcc_prec(0.5, 0.5)}')
    print(f'for pcc = 0.5, and cov 1.0, prec = {alc.prec_from_pcc_cov(0.5, 1.0)}')
    pcc_vals = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001]
    ranges = [[0.0001, 0.00011], [0.00011, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    cov_base_values = np.concatenate(arrays)

    plt.figure(figsize=((6, 3.5)))
    for pcc_val in pcc_vals:
        prec_values = [alc.prec_from_pcc_cov(pcc_val, cov_value) for cov_value in cov_base_values]
        prec_cov_pairs = [(prec, cov) for prec, cov in zip(prec_values, cov_base_values)]
        prec_cov_pairs = sorted(prec_cov_pairs, key=lambda x: x[0])
        prec_cov_pairs = [(prec, cov) for prec, cov in prec_cov_pairs if prec <= 1.0]
        prec_values, cov_values = zip(*prec_cov_pairs)
        plt.scatter(cov_values, prec_values, label=f'PCC = {pcc_val}', s=1)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.text(0.05, 0.98, f'alpha = {alpha}, Cmin = 0.0001', ha='left', va='top', fontsize=9, transform=plt.gca().transAxes)
    plt.legend(scatterpoints=1, markerscale=7, handletextpad=0.5, labelspacing=0.5, fontsize='small', loc='lower center')
    plt.tight_layout()
    savefigs(plt, out_name)

def plot_varying_base_coverage(out_name):
    ''' The purpose of this plot is to see the effect of having a different
        base coverage than attack coverage. We vary the base coverage from 1/10K to 1 while keeping all other parameters constant. What this shows is that the ALC varies substantially when the coverage values are not similar.
    '''
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    cov_values = np.logspace(np.log10(0.0001), np.log10(1), 5000)
    p_base = 0.5
    c_attack = 0.01

    # Run several different relative improvements between base and attack
    increase_values = [0.2, 0.5, 0.8, 0.98]
    plt.figure(figsize=((8, 5)))
    for increase in increase_values:
        p_attack = p_base + (increase * (1.0 - p_base))
        scores = [alc.alc(p_base=p_base, c_base=cov_value, p_attack=p_attack, c_attack=c_attack) for cov_value in cov_values]
        plt.scatter(cov_values, scores, label=f'precision increase = {increase}', s=2)
    plt.xscale('log')
    plt.axvline(x=0.01, color='black', linestyle='dashed')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel(f'Base Coverage (Attack Coverage = {c_attack})', fontsize=12)
    plt.ylabel('Anonymity Loss Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    savefigs(plt, out_name)

def run_pcc_checks(alc, p_base, c_base, pcc_base):
    if c_base <= 0.0001:
        return
    p_base_test = round(alc.prec_from_pcc_cov(pcc_base, c_base),3)
    if round(p_base,3) != p_base_test:
        print(f'Error: prec_from_pcc_cov({pcc_base}, {c_base})')
        print(f'Expected: {round(p_base,3)}, got: {p_base_test}')
        quit()
    c_base_test = round(alc.cov_from_pcc_prec(pcc_base, p_base),3)
    if round(c_base,3) != c_base_test:
        print(f'Error: cov_from_pcc_prec({pcc_base}, {p_base})')
        print(f'Expected: {round(c_base,3)}, got: {c_base_test}')
        quit()

def do_alc_test(alc, p_base, c_base, increase, c_attack):
    print('------------------------------------')
    p_attack = p_base + increase * (1.0 - p_base)
    print(f'Base precision: {p_base}, base coverage: {c_base}\nattack precision: {p_attack}, attack coverage: {c_attack}')
    print(f'prec increase: {increase}')
    pcc_atk = alc.pcc(prec=p_attack, cov=c_attack)
    print(f'pcc_atk: {pcc_atk}')
    pcc_base = alc.pcc(prec=p_base, cov=c_base)
    print(f'pcc_base: {pcc_base}')
    run_pcc_checks(alc, p_base, c_base, pcc_base)
    print(f'ALC: {round(alc.alc(p_base=p_base, c_base=c_base, p_attack=p_attack, c_attack=c_attack),3)}')

def make_alc_plots(cov_adjust_strength=3.0, pairs='v3'):
    alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
    alc.set_param('cov_adjust_strength', cov_adjust_strength)
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
            ALC = [alc.alc(p_base=Pbase, c_base=Cbase, p_attack=p, c_attack=Catk) for p in Patk]
            axs[i].plot(Patk, ALC, label=f'Pbase={Pbase}')
        
        axs[i].text(0.05, 0.95, f'Catk = {Catk}, Cbase = {Cbase}\nalpha = {cov_adjust_strength}\nCmin = 0.0001', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
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
    plt.savefig(f'alc_plots/alc_plot_{cov_adjust_strength}_{pairs}.png')
    plt.savefig(f'alc_plots/alc_plot_{cov_adjust_strength}_{pairs}.pdf')

alc = anonymity_loss_coefficient.AnonymityLossCoefficient()
do_alc_test(alc, p_base=0.5, c_base=1.0, increase=0.2, c_attack=1.0)
do_alc_test(alc, p_base=0.2, c_base=1.0, increase=0.8, c_attack=1.0)
do_alc_test(alc, p_base=0.999, c_base=1.0, increase=0.9, c_attack=1.0)
do_alc_test(alc, p_base=0.5, c_base=0.1, increase=0.2, c_attack=0.1)
do_alc_test(alc, p_base=0.2, c_base=0.1, increase=0.8, c_attack=0.1)
do_alc_test(alc, p_base=0.5, c_base=0.01, increase=0.2, c_attack=0.01)
do_alc_test(alc, p_base=0.2, c_base=0.01, increase=0.8, c_attack=0.01)
do_alc_test(alc, p_base=0.5, c_base=0.001, increase=0.2, c_attack=0.001)
do_alc_test(alc, p_base=0.2, c_base=0.001, increase=0.8, c_attack=0.001)
do_alc_test(alc, p_base=0.5, c_base=0.0001, increase=0.2, c_attack=0.0001)
do_alc_test(alc, p_base=0.2, c_base=0.0001, increase=0.8, c_attack=0.0001)
do_alc_test(alc, p_base=1.0, c_base=0.00001, increase=0, c_attack=0.00001)
plot_prec_cov_for_equal_pcc('prec_cov_for_equal_pcc')
make_alc_plots(pairs='v3')
for cov_adjust_strength in [1.0, 2.0, 3.0, 4.0]:
    make_alc_plots(cov_adjust_strength=cov_adjust_strength, pairs='v1')
    make_alc_plots(cov_adjust_strength=cov_adjust_strength, pairs='v2')
plot_varying_base_coverage('varying_base_coverage')
plot_prec_cov_for_equal_fscore('prec_cov_for_equal_fscore')
plot_fscore_prec_for_equal_cov('fscore_prec_for_equal_cov')
plot_identical_cov('identical_cov')
plot_identical_cov('identical_cov_limit', limit=0.5)
#plot_abs_weights('abs_weights')
plot_cov_adjust('cov_adjust')
plot_base_adjusted_pcc('base_adjusted_pcc')