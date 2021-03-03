clear all
close all
clc

%-------------------------------------------------------------------------%
% based on the papers:
%
% 1) "A very fast time scale of human motor adaptation: Within movement 
% adjustments of internal representations during reaching", Crevecoeur
% 2020.
%
% 2) "Optimal Task-Dependent Changes of Bimanual Feedback Control and 
% Adaptation", Diedrichsen 2007.

%% %---PARAMETERS---% %% 

m              = 2.5;  % [kg]
k              = 0.1;  % [Nsm^-1]
tau            = 0.1;  % [s]
delta          = 0.01; % [s]  
theta          = 15;   % [N/(m/s)] - coeff pert force (F = +-L*dy/dt)
alpha          = [1000 1000 20 20 0 0];% [PosX, PosY, VelX, VelY, Fx, Fy]
learning_rates = [.1 .1];% [right left]
coeffQ         = 1;      % increase or decrease Q matrix during trials
time           = 0.6;    % [s] - experiment 1 - reaching the target
stab           = 0.01;   % [s] - experiment 1 - stabilization 
nStep          = round((time+stab)/delta)-1;
N              = round(time/delta);


% Protocol parameters

right_perturbation = 'CCW';     % CCW, CW or BASELINE (no FFs)
left_perturbation  = 'BASELINE';% CCW, CW or BASELINE (no FFs)
numoftrials        = 15;        % number of trials 
catch_trials       = 0;         % number of catch trials


%% %---SYSTEM CREATION---% %%

% STATE VECTOR REPRESENTATION:
% (1-6)  RIGHT HAND x, y, dx, dy, fx and fy
% (7-12) LEFT HAND  x, y, dx, dy, fx and fy
% (13-14) MID POINT x and y

xinit  = [.06 0 0 0 0 0 -.06 0 0 0 0 0 0 0]';    % [right left]
xfinal = [.06 .15 0 0 0 0 -.06 .15 0 0 0 0 0 0.15]';% [right left] 

%---System---%
A = [0 0 1 0 0 0; 0 0 0 1 0 0;...
	 0 0 -k/m 0 1/m 0;...
	 0 0 0 -k/m 0 1/m; 0 0 0 0 -1/tau 0;...
	 0 0 0 0 0 -1/tau];
A = blkdiag(A,A);
A = [A zeros(12,2); zeros(2,14)];
A(13,:) = [0 0 0.5 0 0 0 0 0 0.5 0 0 0 0 0];
A(14,:) = [0 0 0 0.5  0 0 0 0 0 0.5 0 0 0 0];
B = zeros(6,2);
B(5,1) = 1/tau;
B(6,2) = 1/tau;
B = blkdiag(B,B);
B = [B; zeros(2,4)];

ns = size(A,1);
nc = size(B,2);

A_hat = A;
DA = (A-A_hat)*delta; % Used when there is a model error
A = eye(size(A))+delta*A;
A_hat = eye(size(A_hat))+delta*A_hat;
B = delta*B;

% Observability Matrix
H = eye(size(A,1));
E = eye(ns,1)';          %See Basar and Tamer, pp. 171


%% %---COST FUNCTION---% %%

Q = zeros(size(A,1),size(A,2),nStep);
M = Q;
TM = Q;
Id = eye(ns);

runningalpha = zeros(ns,nStep); 
for i = 1:nStep
    
    fact = min(1,(i*delta/time))^3;%2
    temp_R = [1 1 fact*10^6 fact*10^6 1 1];%2 - 4 - 4.2 - 8.2
    temp_L = [1 1 fact*10^6 fact*10^6 1 1];%4 - 4 - 4 - 4
    runningalpha(:,i) = [temp_R temp_L fact*10^8 fact*10^8]';
    
end

%Filling in the cost matrices
for j = 1:nStep
    for i = 1:ns
        
        Q(:,:,j) = Q(:,:,j) + runningalpha(i,j)*Id(:,i)*Id(:,i)';
        
    end
end

%Signal Dependent Noise
nc = size(B,2);
Csdn = zeros(size(B,1),nc,nc);
for i = 1:nc
    Csdn(:,i,i) = .1*B(:,i);    
end

M = Q;
TM = Q;
D = eye(ns);

% Implementing the backwards recursions

M(:,:,end) = Q(:,:,end);
L = zeros(size(B,2),size(A,1),nStep-1);  % Optimal Minimax Gains
Lambda = zeros(size(A,1),size(A,2),nStep-1);

% Optimization of gamma
gamma = 50000;
minlambda = zeros(nStep-1,1);
gammaK = 0.5;
reduceStep = 1;
positive = false;
relGamma = 1;

while (relGamma > .001 || ~positive)

    for k = nStep-1:-1:1

        % Minimax Feedback Control
        TM(:,:,k) = gamma^2*eye(size(A))-D'*M(:,:,k+1)*D;
        minlambda(k) = min(eig(TM(:,:,k)));

        Lambda(:,:,k) = eye(size(A_hat))+(B*B'-gamma^-2*(D*D'))*M(:,:,k+1);
        M(:,:,k) = Q(:,:,k)+A_hat'*(M(:,:,k+1)^-1+B*B'-gamma^-2*D*D')^-1*A_hat;
        L(:,:,k) = B'*M(:,:,k+1)*Lambda(:,:,k)^-1*A_hat;

    end

    oldGamma = gamma;

    if min(real(minlambda)) >= 0

        gamma = (1-gammaK)*gamma;
        relGamma = (oldGamma-gamma)/oldGamma;
        positive = true;

    elseif min(real(minlambda)) < 0

        gamma = (1-gammaK)^-1*gamma;
        reduceStep = reduceStep + 0.5;
        relGamma = -(oldGamma-gamma)/oldGamma;
        gammaK = gammaK^reduceStep;
        positive = false;

    end
end

%% %---SIMULATION---% %%

% Add perturbation (curl FFs) to the matrix A
switch right_perturbation
    case 'CCW'
        A(3,4) = -delta*(theta/m);
        A(4,3) = delta*(theta/m);
    case 'CW'
        A(3,4) = delta*(theta/m);
        A(4,3) = -delta*(theta/m);
    case 'BASELINE'
        A(3,4) = 0;
        A(4,3) = 0;
    otherwise
        error('The perturbation choice is incorrect !')
end

switch left_perturbation
    case 'CCW'
        A(9,10) = -delta*(theta/m);
        A(10,9) = delta*(theta/m);
    case 'CW'
        A(9,10) = delta*(theta/m);
        A(10,9) = -delta*(theta/m);
    case 'BASELINE'
        A(9,10) = 0;
        A(10,9) = 0;
    otherwise
        error('The perturbation choice is incorrect !')
end

% Initialization simulation

 % Robust
x = zeros(ns,nStep+1,numoftrials);
xhat = x;

control     = zeros(nc,nStep,numoftrials);    % Initialize control
avControl   = zeros(nc,nStep);                % Average Control variable

% Random indexes for catch trials
catch_trials_idx = [];

if catch_trials ~= 0
    while length(catch_trials_idx) ~= catch_trials
        random = randi(numoftrials, 1, 1); 
        catch_trials_idx = [catch_trials_idx random];
        catch_trials_idx = unique(catch_trials_idx);
    end
end

for p = 1:numoftrials

    % Robust
    x(:,1,p) = xinit - xfinal;
    xhat(:,1,p) = x(:,1,p);
    u = zeros(nStep-1,size(B,2)); % size(B,2) is the control dimension
    w = zeros(ns,1);
    Oxi = 0.001*B*B';
    Omega = eye(6)*Oxi(5,5);

    %Parameters for State Estimation
    Sigma = zeros(ns,ns,nStep);
    Sigma(:,:,1) = eye(ns)*10^-2;
    SigmaK = Sigma;

    for i = 1:nStep-1

        %--- Robust ---%
        A_old = A;
        
        if (~isempty(catch_trials_idx)) & (k == catch_trials_idx)
            A(3,4)  = 0;
            A(9,10) = 0;
            A(4,3)  = 0;
            A(10,9) = 0;
        end

        sensoryNoise = mvnrnd(zeros(size(Omega,1),1),Omega)';
        sensoryNoise = [sensoryNoise; sensoryNoise; 0;0];
        motorNoise = mvnrnd(zeros(size(Oxi,1),1),Oxi)';

        %MINMAX HINFTY CONTROL ------------------------------------------------
        %Riccati Equation for the State Estimator
        Sigma(:,:,i+1) = A_hat*(Sigma(:,:,i)^-1+H'*(E*E')^-1*H-gamma^-2*Q(:,:,i))^-1*A_hat'+D*D';

        %Feedback Eequation
        yx = H*x(:,i,p) + sensoryNoise;

        %Minmax Simulation with State Estimator
        u(i,:) = -B'*(M(:,:,i+1)^-1+B*B'-gamma^-2*(D*D'))^-1*A_hat*...   %Control
        (eye(ns)-gamma^-2*Sigma(:,:,i)*M(:,:,1))^-1*xhat(:,i,p);

        %Signal Dependent Noise - Robust Control
        sdn = 0;

        for isdn = 1:nc
            sdn = sdn + normrnd(0,1)*Csdn(:,:,isdn)*u(i,:)';
        end

        xhat(:,i+1,p) = A_hat*xhat(:,i,p) + B*u(i,:)'+...
        A_hat*(Sigma(:,:,i)^-1+H'*(E*E')^-1*H-gamma^-2*Q(:,:,i))^-1*(gamma^-2*Q(:,:,i)*xhat(:,i,p)+H'*(E*E')^-1*(yx-H*xhat(:,i,p)));

        % Minmax Simulation
        DA = A - A_hat;          
        wx = DA*x(:,i,p); % Non zero if there is a model error. 
        x(:,i+1,p) = A_hat*x(:,i,p) + B*u(i,:)'+ D*wx + motorNoise + sdn;

        % Update the A matrix
        
        % Update the A matrix (see eq.(9))
        eps = x(1:6,i+1,p)- xhat(1:6,i+1,p);
        
        theta_up_R = A_hat(3,4);
        dzhat_dL = zeros(1,6);
        dzhat_dL(1,3) = xhat(4,i+1,p);% x_hat check
        theta_up_R = theta_up_R + learning_rates(1)*dzhat_dL*eps;
        A_hat(3,4) = theta_up_R;
        
        theta_up_R = A_hat(4,3);
        dzhat_dL = zeros(1,6);
        dzhat_dL(1,4) = xhat(3,i+1,p);% x_hat check
        theta_up_R = theta_up_R + learning_rates(1)*dzhat_dL*eps;
        A_hat(4,3) = theta_up_R;
        
        eps = x(7:12,i+1,p)- xhat(7:12,i+1,p);
        
        theta_up_L = A_hat(9,10);
        dzhat_dL = zeros(1,6);
        dzhat_dL(1,3) = xhat(10,i+1,p);
        theta_up_L  = theta_up_L + learning_rates(2)*dzhat_dL*eps;
        A_hat(9,10) = theta_up_L;
        
        theta_up_L = A_hat(10,9);
        dzhat_dL = zeros(1,6);
        dzhat_dL(1,4) = xhat(9,i+1,p);
        theta_up_L  = theta_up_L + learning_rates(2)*dzhat_dL*eps;
        A_hat(10,9) = theta_up_L;

        u_temp = [u(i,1) u(i,2) u(i,3) u(i,4)]';
        control(:,i,p) = u_temp;
        A = A_old;
    end
    
    avControl = avControl + control(:,:,p)/numoftrials;

end

%% %---GRAPHS---% %%

translate = repmat([.06 .15 0 0 0 0 -.06 .15 0 0 0 0 0 0.15]',1,N,numoftrials);
x     = x(:,1:N,:) + translate;

for trial = 1:numoftrials
    
    figure(1)
    
    % Position
    subplot(2,2,[1,2])
    hold on;
    midx = x(13,1:N,trial);
    midy = x(14,1:N,trial);
    plot(x(1,1:N,trial), x(2,1:N,trial)); 
    plot(x(7,1:N,trial), x(8,1:N,trial));
    plot(midx, midy);
    plot(0,0,'ro','LineWidth',2);
    plot(0,.15,'ro','MarkerSize',10,'LineWidth',2);
    plot(0.06,0,'ro','LineWidth',2);
    plot(0.06,.15,'ro','MarkerSize',10,'LineWidth',2);
    plot(-0.06,0,'ro','LineWidth',2);
    plot(-0.06,.15,'ro','MarkerSize',10,'LineWidth',2);
    xlabel('x-coord [m]'); ylabel('y-coord [m]'); title(['Robust model - one cursor - trajectories'],'FontSize',14);
    axis([-(max(x(1,:,1)) + 0.04) (max(x(1,:,1)) + 0.04)  -0.01 0.16])

    % Control
    subplot(2,2,4)
    plot([.01:.01:(nStep)*.01],control(1,:,trial)), hold on;
    plot([.01:.01:(nStep)*.01],avControl(1,:),'k','Linewidth',2)
    xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector - Right','FontSize',14);
    %axis square

    subplot(2,2,3)
    plot([.01:.01:(nStep)*.01],control(3,:,trial)), hold on;
    plot([.01:.01:(nStep)*.01],avControl(3,:),'k','Linewidth',2)
    xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector - Left','FontSize',14);
    %axis square

    %input(' ');    

end

% Velocity profiles

figure(2)
subplot(131)
plot([.01:.01:(nStep)*.01], x(3,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(9,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('X Velocity [m/s]');
legend('right', 'left');

subplot(132)
plot([.01:.01:(nStep)*.01], x(4,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(10,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('Y Velocity [m/s]');
legend('right', 'left');

subplot(133)
plot([.01:.01:(nStep)*.01], sqrt(x(3,:,end).^2+x(4,:,end).^2), 'b');hold on;
plot([.01:.01:(nStep)*.01], sqrt(x(9,:,end).^2+x(10,:,end).^2), 'r');hold off;
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('right', 'left');

% Force profiles

figure(3)
subplot(131)
plot([.01:.01:(nStep)*.01], x(5,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(11,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('X Force [N]');
legend('right', 'left');

subplot(132)
plot([.01:.01:(nStep)*.01], x(6,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(12,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('Y Force [N]');
legend('right', 'left');

subplot(133)
plot([.01:.01:(nStep)*.01], sqrt(x(5,:,end).^2+x(11,:,end).^2), 'b');hold on;
plot([.01:.01:(nStep)*.01], sqrt(x(6,:,end).^2+x(12,:,end).^2), 'r');hold off;
xlabel('Time [s]');
ylabel('Force [N]');
legend('right', 'left');

