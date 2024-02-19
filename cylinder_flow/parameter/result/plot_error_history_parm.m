clear; clc
format short
format compact

%%
error = textread('error_u0_history.txt');
semilogy(0:2000,error(1:2001,:),'-','color',[0.8,0.0,0.0],'linewidth',1.5); hold on

error = textread('error_u1_history.txt');
semilogy(0:2000,error(1:2001,:),'-','color',[0.0,0.5,0.5],'linewidth',1.5); hold on

error = textread('error_p_history.txt');
semilogy(0:2000,error(1:2001,:),'-','color',[0.0,0.0,0.8],'linewidth',1.5); hold on

legend('error: u_1', 'error: u_2', 'error: p', 'fontsize',14)
xticks([0,400,800,1200,1600,2000])
xticklabels({'0','40000','80000','120000','160000','200000'})
xlabel('iteration count','fontsize',16); ylabel('relative error','fontsize',16)