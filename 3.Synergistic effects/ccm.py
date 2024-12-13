import matplotlib.pyplot as plt
# import jdc
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import pyEDM

# for showing progress bar in for loops


feature1 = "HF"
feature2 = "GBCD"
feature3 = "IMS"

yos1 = "HF"
yos2 = "GBCD"
yos3 = "IMS"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman' # 全局字体样式
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
E = 0
def shadow_manifold(X, tau, E, L):
    X = X[:L]
    M = {t:[] for t in range((E-1) * tau, L)}
    for t in range((E-1) * tau, L):
        x_lag = []
        for t2 in range(0, E-1 + 1):
            x_lag.append(X[t-t2*tau])
        M[t] = x_lag
    return M
def get_distances(Mx):
    t_vec = [(k, v) for k,v in Mx.items()]
    t_steps = np.array([i[0] for i in t_vec])
    vecs = np.array([i[1] for i in t_vec])
    dists = distance.cdist(vecs, vecs)
    return t_steps, dists
def get_nearest_distances(t, t_steps, dists, E):
    t_ind = np.where(t_steps == t)
    dist_t = dists[t_ind].squeeze()
    nearest_inds = np.argsort(dist_t)[1:E + 1 + 1]
    nearest_timesteps = t_steps[nearest_inds]
    nearest_distances = dist_t[nearest_inds]
    return nearest_timesteps, nearest_distances
class ccm:
    def __init__(self, X, Y, tau=1, E=2, L=500):
        self.X = X
        self.Y = Y
        self.tau = tau
        self.E = E
        self.L = L
        self.My = shadow_manifold(self.Y, self.tau, self.E,self.L)
        self.t_steps, self.dists = get_distances(self.My)
    def causality(self):
        X_true_list = []
        X_hat_list = []
        for t in list(self.My.keys()):
            X_true, X_hat = self.predict(t)
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)
        x, y = X_true_list, X_hat_list
        correl = np.corrcoef(x, y)[0][1]
        return correl
    def predict(self, t):
        eps = 0.000001
        t_ind = np.where(self.t_steps == t)
        dist_t = self.dists[t_ind].squeeze()
        nearest_timesteps, nearest_distances = get_nearest_distances(t, self.t_steps, self.dists, E)
        u = np.exp(
            -nearest_distances / np.max([eps, nearest_distances[0]]))
        w = u / np.sum(u)
        X_true = self.X[t]
        X_cor = np.array(self.X)[nearest_timesteps]
        X_hat = (w * X_cor).sum()
        return X_true, X_hat


    def visualize_cross_mapping(self):
        f, axs = plt.subplots(1, 2, figsize=(12, 6))
        for i, ax in zip((0, 1), axs):
            X_lag, Y_lag = [], []
            for t in range(1, len(self.X)):
                X_lag.append(X[t - tau])
                Y_lag.append(Y[t - tau])
            X_t, Y_t = self.X[1:], self.Y[1:]
            ax.scatter(X_t, X_lag, s=5, label='$M_x$')
            ax.scatter(Y_t, Y_lag, s=5, label='$M_y$', c='y')
            A, B = [(self.Y, self.X), (self.X, self.Y)][i]
            cm_direction = ['Mx to My', 'My to Mx'][i]
            Ma = shadow_manifold(A, tau, E, L)
            Mb = shadow_manifold(B, tau, E, L)

            t_steps_A, dists_A = get_distances(Ma)
            t_steps_B, dists_B = get_distances(Mb)
            timesteps = list(Ma.keys())
            for t in np.random.choice(timesteps, size=3, replace=False):
                Ma_t = Ma[t]
                near_t_A, near_d_A = get_nearest_distances(t, t_steps_A, dists_A, E)
                for i in range(E + 1):
                    A_t = Ma[near_t_A[i]][0]
                    A_lag = Ma[near_t_A[i]][1]
                    ax.scatter(A_t, A_lag, c='b', marker='s')
                    B_t = Mb[near_t_A[i]][0]
                    B_lag = Mb[near_t_A[i]][1]
                    ax.scatter(B_t, B_lag, c='r', marker='*', s=50)
                    ax.plot([A_t, B_t], [A_lag, B_lag], c='r', linestyle=':')
            ax.set_title(f'{cm_direction} cross mapping. time lag, tau = {tau}, E = 3')
            ax.legend(prop={'size': 14})
            ax.set_xlabel('$X_t$, $Y_t$', size=15)
            ax.set_ylabel('$X_{t-1}$, $Y_{t-1}$', size=15)

def plot_ccm_correls(X, Y, tau, E, L,ax1,ax2,yos1,yos2):
    M = shadow_manifold(Y, tau, E, L)
    t_steps, dists = get_distances(M)
    ccm_XY = ccm(X, Y, tau, E, L)
    ccm_YX = ccm(Y, X, tau, E, L)

    X_My_true, X_My_pred = [], []
    Y_Mx_true, Y_Mx_pred = [], []
    for t in range(tau, L):
        true, pred = ccm_XY.predict(t)
        X_My_true.append(true)
        X_My_pred.append(pred)
        true, pred = ccm_YX.predict(t)
        Y_Mx_true.append(true)
        Y_Mx_pred.append(pred)

    coeff = np.round(np.corrcoef(X_My_true, X_My_pred)[0][1], 2)
    ax2.scatter(X_My_true, X_My_pred, s=60, color='#3B8791',marker='+',linewidths=1)   #散点换颜色  '#A07936'
    ax2.set_xlabel(f'{yos1}( ) (observed)', size=18)
    # axs[1].set_ylabel('$\hat{X}(t)|M_y$ (estimated)', size=15)
    ax2.set_ylabel(f'{yos2}( ) (estimated)', size=18)
    # ax2.set_title(f'tau={tau}, E={E}, L={L}, Correlation coeff = {coeff}')
    coeff = np.round(np.corrcoef(Y_Mx_true, Y_Mx_pred)[0][1], 2)
    ax1.scatter(Y_Mx_true, Y_Mx_pred, s=60, color='#A07936',marker='+',linewidths=1)
    ax1.set_xlabel(f'{yos2}( ) (observed)', size=18)
    # axs[0].set_ylabel('$\hat{Y}(t)|M_x$ (estimated)', size=15)
    ax1.set_ylabel(f'{yos1}( ) (estimated)', size=18)
    # ax1.set_title(f'tau={tau}, E={E}, L={L}, Correlation coeff = {coeff}')



# 删除边框和标签
def delBorder(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

def plotSubpanelA(ax1,ax2,ax3,file_path):

    #ax1
    xlsx_file2 = file_path
    xlsx_file1 = file_path
    Original_data = pd.read_excel(xlsx_file1, sheet_name="Sheet2", usecols=[0, 1, 2])
    CCM_data = pd.read_excel(xlsx_file2, sheet_name="Sheet3", usecols=[1, 2, 3])
    l1, = ax1.plot(CCM_data[["LibSize"]], CCM_data[[f"{feature3}:{feature2}"]]
                   , color='#3B8791')
    l2, = ax1.plot(CCM_data[["LibSize"]], CCM_data[[f"{feature2}:{feature3}"]]
                   , color='#A07936')
    ax1.legend(handles=[l1, l2], loc='upper right'
               , labels=[f'{yos3} versus {yos2}', f'{yos2} versus {yos3}']
               , handlelength=0.6  # 图例句柄的长度
               , fontsize=15, frameon=False,bbox_to_anchor=(1, 0.7))
    ax1.set_xticklabels([0, 0,20,40,60,80,100], fontsize=15)
    ax1.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    ax1.set_xlim(0-5, 100+5)
    ax1.set_ylim(0 - 0.16, 1 + 0.06)
    ax1.set_xlabel('Timestamp (L)', fontsize=18)
    ax1.set_ylabel('Correlation coefficient (ρ)', fontsize=18)

    X = Original_data[f'{feature2}']
    X = np.array(X).tolist()
    # print(X)
    Y = Original_data[f'{feature3}']
    Y = np.array(Y).tolist()
    # print(Y)
    L = 100
    tau = 1
    E = 2
    plot_ccm_correls(X, Y, tau, E, L, ax2, ax3,yos3,yos2)

    ax2.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    # ax4.set_yticklabels([0,0,0.05,0.10,0.15,0.20,0.25],fontsize=15)
    ax2.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    ax2.set_xticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax4.set_yticks([0,0,0.05,0.10,0.15,0.20,0.25])
    ax2.set_yticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlim(0 - 0.06, 1 + 0.06)
    # ax4.set_ylim(0 - 0.015, 0.25 + 0.015)
    ax2.set_ylim(0 - 0.06, 1 + 0.06)

    ax2.text(0.42, -0.22, 't', fontstyle='italic', fontsize=18)
    ax2.text(-0.24, 0.39, 't', fontstyle='italic', rotation=90, fontsize=18)

    ax3.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    # ax4.set_yticklabels([0,0,0.05,0.10,0.15,0.20,0.25],fontsize=15)
    ax3.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    ax3.set_xticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax4.set_yticks([0,0,0.05,0.10,0.15,0.20,0.25])
    ax3.set_yticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_xlim(0 - 0.06, 1 + 0.06)
    # ax4.set_ylim(0 - 0.015, 0.25 + 0.015)
    ax3.set_ylim(0 - 0.06, 1 + 0.06)
    # ax4.text(0.348,-0.058,'t', fontstyle='italic',fontsize=18)
    ax3.text(0.378, -0.22, 't', fontstyle='italic', fontsize=18)
    # ax4.text(-0.31,0.09,'t', fontstyle='italic',rotation=90,fontsize=18)
    ax3.text(-0.24, 0.43, 't', fontstyle='italic', rotation=90, fontsize=18)

def plotSubpanelB(ax1,ax2,ax3,file_path):
    #ax1
    xlsx_file2 = file_path
    xlsx_file1 = file_path
    Original_data = pd.read_excel(xlsx_file1, sheet_name="Sheet2", usecols=[0, 1, 2])
    CCM_data = pd.read_excel(xlsx_file2, sheet_name="Sheet3", usecols=[1, 2, 3])
    l1, = ax1.plot(CCM_data[["LibSize"]], CCM_data[[f"{feature1}:{feature2}"]]
                   , color='#3B8791')  #换了颜色 #'#3B8791'
    l2, = ax1.plot(CCM_data[["LibSize"]], CCM_data[[f"{feature2}:{feature1}"]]
                   , color='#A07936')
    ax1.legend(handles=[l1, l2], loc='upper right'
               , labels=[f'{yos1} versus {yos2}', f'{yos2} versus {yos1}']
               , handlelength=0.6  # 图例句柄的长度
               , fontsize=15, frameon=False,bbox_to_anchor=(1, 0.7))
    ax1.set_xticklabels([0, 0, 20, 40, 60, 80, 100], fontsize=15)
    ax1.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    ax1.set_xlim(0-5, 82+5)
    ax1.set_ylim(0 - 0.16, 1 + 0.06)
    ax1.set_xlabel('Timestamp (L)', fontsize=18)
    ax1.set_ylabel('Correlation coefficient (ρ)', fontsize=18)

    #ax2
    X = Original_data[f'{feature2}']
    X = np.array(X).tolist()
    # print(X)
    Y = Original_data[f'{feature1}']
    Y = np.array(Y).tolist()
    # print(Y)
    L = 82
    tau = 1
    E = 2
    plot_ccm_correls(X, Y, tau, E, L, ax2, ax3,yos1,yos2)

    ax2.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    # ax4.set_yticklabels([0,0,0.05,0.10,0.15,0.20,0.25],fontsize=15)
    ax2.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    ax2.set_xticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax4.set_yticks([0,0,0.05,0.10,0.15,0.20,0.25])
    ax2.set_yticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlim(0 - 0.06, 1 + 0.06)
    # ax4.set_ylim(0 - 0.015, 0.25 + 0.015)
    ax2.set_ylim(0 - 0.06, 1 + 0.06)

    #调整了的图2尺度
    # ax2.set_xticklabels([0.4, 0.5, 0.6, 0.7, 0.8], fontsize=15)
    # ax2.set_yticklabels([0.4, 0.5, 0.6, 0.7, 0.8], fontsize=15)
    # ax2.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    # ax2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    # # ax3.set_xticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # # ax3.set_yticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax2.set_xlim(0.4 - 0.02, 0.8 + 0.02)
    # ax2.set_ylim(0.4 - 0.02, 0.8 + 0.02)
    ax2.text(0.42, -0.22, 't', fontstyle='italic', fontsize=18)
    ax2.text(-0.24, 0.37, 't', fontstyle='italic', rotation=90, fontsize=18)

    # ax2.text(0.547, 0.315, 't', fontstyle='italic', fontsize=18)
    # ax2.text(0.307, 0.547, 't', fontstyle='italic', rotation=90, fontsize=18)


    ax3.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    # ax4.set_yticklabels([0,0,0.05,0.10,0.15,0.20,0.25],fontsize=15)
    ax3.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    ax3.set_xticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax4.set_yticks([0,0,0.05,0.10,0.15,0.20,0.25])
    ax3.set_yticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_xlim(0 - 0.06, 1 + 0.06)
    # ax4.set_ylim(0 - 0.015, 0.25 + 0.015)
    ax3.set_ylim(0 - 0.06, 1 + 0.06)

    # # 调整了的图3尺度
    # ax3.set_xticklabels([0,0,0.1,0.2,0.3,0.4], fontsize=15)
    # ax3.set_yticklabels([0,0,0.1,0.2,0.3,0.4], fontsize=15)
    # ax3.set_xticks([0,0,0.1,0.2,0.3,0.4, 0.5])
    # ax3.set_yticks([0,0,0.1,0.2,0.3,0.4, 0.5])
    # # ax3.set_xticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # # ax3.set_yticks([0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax3.set_xlim(0.0-0.02,0.2 + 0.02)
    # ax3.set_ylim(0.0-0.02,0.2 + 0.02)

    # ax4.text(0.348,-0.058,'t', fontstyle='italic',fontsize=18)
    ax3.text(0.359, -0.22, 't', fontstyle='italic', fontsize=18)
    # ax4.text(-0.31,0.09,'t', fontstyle='italic',rotation=90,fontsize=18)
    ax3.text(-0.24, 0.43, 't', fontstyle='italic', rotation=90, fontsize=18)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 300

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    ax1 = [axs[0, 0], axs[0, 1], axs[0, 2]]
    ax2 = [axs[1, 0], axs[1, 1], axs[1, 2]]
    # 第一行的子图
    file_path_a = fr"E:\scrapy\pubmed\ccm_project\MyCCM\data\tran_data\sco\ccm_circulation_scolar.xlsx"
    plotSubpanelA(ax1[0], ax1[1], ax1[2], file_path_a)
    file_path_b = fr"E:\scrapy\pubmed\ccm_project\MyCCM\data\tran_data\sco\ccm_circulation_foot.xlsx"
    plotSubpanelB(ax2[0], ax2[1], ax2[2], file_path_b)

    fig.subplots_adjust(wspace=0.3, hspace=0.24, left=0.08)
    fig.text(0.042, 0.93, 'a', fontsize=36, va='top', fontweight='bold')
    fig.text(0.33, 0.93, 'c', fontsize=36, va='top', fontweight='bold')
    fig.text(0.63, 0.93, 'e', fontsize=36, va='top', fontweight='bold')

    fig.text(0.042, 0.5, 'b', fontsize=36, va='top', fontweight='bold')
    fig.text(0.33, 0.5, 'd', fontsize=36, va='top', fontweight='bold')
    fig.text(0.63, 0.5, 'f', fontsize=36, va='top', fontweight='bold')

    plt.show()
    plt.savefig('GBCD_plot.jpg', format='jpg')