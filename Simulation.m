%% The file reproduces the simulation for m_{ora}, m_{nai}, m_{MLE}, m_{CMLE} and m_{EE}
%% with the linear g_{r^x,r^y}'s of Tan (2022+).
 
% References: 
% Tan, R. (2022+) Nonparametric regression with nonignorable missing covariates
% and outcomes using bounded inverse weighting. Journal of Nonparametric Statistics. 

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

n_sample = [300,1000];% Sample sizes
n_r = 200;% Number of simulation repetition
xi = 0.01;% Tuning parameter for CMLE.

for k = 1:2
    n = n_sample(k);% Sample size
   
    S = zeros(200,n_r,5);% To store estimated regression curves.
    ISE = zeros(n_r,5);% To store the resulting ISEs.
 
    a = -1;
    b = 1;

    % Uncomment the particular one and comment the other ones to select a particular model.
    
    % Models for true m;
    % Models (i) and (iv) -------------------------------------------------
    m = @(x) x.^2+0.5;
    % ---------------------------------------------------------------------
    
    % Model (ii) ----------------------------------------------------------
    %m = @(x) (normpdf(x,-0.3,0.15)-normpdf(x,0.3,0.15))./3;
    % ---------------------------------------------------------------------
    
    % Model (iii) ---------------------------------------------------------
    %m = @(x) sin(pi.*x)+0.5;
    % ---------------------------------------------------------------------

    for i = 1:n_r
        rng(3*i)
        
        % Generate data.
        % Models (i), (ii) and (iv) for X----------------------------------
        X = random('Normal',0,0.5,[n,1]);
        %------------------------------------------------------------------
        
        % Model (iii) for X------------------------------------------------
        %X = (random('Chisquare',4,[n,1])-5)./4;
        %------------------------------------------------------------------
        
        Y = m(X)+random('Normal',0,0.5,[n,1]);

        U = random('Normal',0,1,[n,1]);

        % The missing data model for (i), (ii) and (iii)-------------------
        %[ RX,RY ] = MisDM( X,Y,U );
        %------------------------------------------------------------------
        
        % The missing data model for (iv)----------------------------------
        [ RX,RY ] = MisDM_vb( X,Y,U );
        %------------------------------------------------------------------

        alpha_MLE = zeros(1,3*size(U,2)+5);
        alpha_CMLE = alpha_MLE;
        alpha_EE = alpha_MLE;

        alpha_nai = alpha_MLE;
        incpt_ind = [1,2+size(U,2),4+2*size(U,2)];
        alpha_nai(incpt_ind) = -Inf;

        % Estimation for the missing data model.
        % (r^x,r^y) = (0,0)
        U_sub = U(RX==RY,:);
        RU_sub = RX(RX==RY);
        [alpha_MLE(1:1+size(U,2))] = LogMLE(U_sub,RU_sub);
        [alpha_CMLE(1:1+size(U,2))] = LogCMLE(U_sub,RU_sub,xi);
        [alpha_EE(1:1+size(U,2))] = LogEE(U_sub,RU_sub);

        % (r^x,r^y) = (1,0)
        X_subx = [X,U];
        X_subx = X_subx(RX==1,:);
        RY_subx = RY(RX==1);
        [alpha_MLE(2+size(U,2):3+2*size(U,2))] = LogMLE(X_subx,RY_subx);
        [alpha_CMLE(2+size(U,2):3+2*size(U,2))] = LogCMLE(X_subx,RY_subx,xi);
        [alpha_EE(2+size(U,2):3+2*size(U,2))] = LogEE(X_subx,RY_subx);

        % (r^x,r^y) = (0,1)
        Y_suby = [Y,U];
        Y_suby = Y_suby(RY==1,:);
        RX_suby = RX(RY==1);
        [alpha_MLE(4+2*size(U,2):5+3*size(U,2))] = LogMLE(Y_suby,RX_suby);
        [alpha_CMLE(4+2*size(U,2):5+3*size(U,2))] = LogCMLE(Y_suby,RX_suby,xi);
        [alpha_EE(4+2*size(U,2):5+3*size(U,2))] = LogEE(Y_suby,RX_suby);

        % Regression estimators
        [ ~,S(:,i,1),~] = loclin( X,Y,a,b );%m_oracle
        [ ~,S(:,i,2)] = loclin_IPW( a,b,X,Y,RX,RY,U,alpha_nai );%m_naive
        
        [ ~,S(:,i,3) ] = loclin_IPW( a,b,X,Y,RX,RY,U,alpha_MLE );%m_MLE
        [ ~,S(:,i,4) ] = loclin_IPW( a,b,X,Y,RX,RY,U,alpha_CMLE );%m_CMLE
        [ ~,S(:,i,5) ] = loclin_IPW( a,b,X,Y,RX,RY,U,alpha_EE );%m_EE

        t = linspace(a,b,200)';
        Yt = m(t);
        ISE(i,:) = trapz(t,(S(:,i,:)-Yt).^2);
    end
    
    % Store results.
    fname = sprintf('BM221128_(iv)newCMLEn%d',n);
    save(fname,'ISE','S');

    % Print mean and standard deviations of the ISEs.
    A = mean(ISE,1,'omitnan')*100;
    B = std(ISE,0,1,'omitnan')*100;
    fprintf([fname ': ']);
    fprintf('%0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f)\n',...
        A(1),B(1),A(2),B(2),A(3),B(3),A(4),B(4),A(5),B(5)) 
end
