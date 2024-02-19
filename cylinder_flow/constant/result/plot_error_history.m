clear; clc
format short
format compact

%%
error = textread('error_u0_history_xfno.txt');
semilogy(0:200,error(1:201,:),'-','color',[0.8,0.0,0.0],'linewidth',1.5); hold on

error = textread('error_u1_history_xfno.txt');
semilogy(0:200,error(1:201,:),'-','color',[0.0,0.5,0.5],'linewidth',1.5); hold on

error = textread('error_p_history_xfno.txt');
semilogy(0:200,error(1:201,:),'-','color',[0.0,0.0,0.8],'linewidth',1.5); hold on

error = textread('error_u0_history_pinns_10.txt');
semilogy(0:200,error(1:201,:),'-','color',[1.0,0.4,0.4],'linewidth',1.5); hold on

error = textread('error_u1_history_pinns_10.txt');
semilogy(0:200,error(1:201,:),'-','color',[0.5,0.0,0.5],'linewidth',1.5); hold on

error = textread('error_p_history_pinns_10.txt');
semilogy(0:200,error(1:201,:),'-','color',[0.0,0.0,0.0],'linewidth',1.5); hold on

legend('XFNO error: u_1', 'XFNO error: u_2', 'XFNO error: p', ...
       'PINNs error: u_1', 'PINNs error: u_2', 'PINNs error: p', 'fontsize',10)
xticks([0,40,80,120,160,200])
xticklabels({'0','4000','8000','12000','16000','20000'})
xlabel('iteration count','fontsize',16); ylabel('relative error','fontsize',16)