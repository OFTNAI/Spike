% Find all polychronous groups that start with the anchor neurons v0 fired
% at the times t0. Typically, should be called from the polychron.m
% (Kalman Katlowitz fixed a bug related to new MATLAB release in 2011)

function group=polygroup(v0,t0,neuronModel)
global a b c d N D pp s post ppre dpre pre delay T timestep di nLayers  ExcitDim InhibDim
model_izhikevich = 1;
model_conductanceLIAF = 2;
maxFiringCount = 1000;

if (neuronModel==model_conductanceLIAF)
    
    restV0 = -0.074; %resting_potential_v0
    
    %params for neurons
    somatic_capcitance_Cm_excit = 500.0*10^-12;
    somatic_capcitance_Cm_inhib = 214.0*10^-12;
    
    somatic_leakage_conductance_g0_excit = 25.0*10^-9;
    somatic_leakage_conductance_g0_inhib = 18.0*10^-9;

    membrane_resistance_R_excit = 1.0/ somatic_capcitance_Cm_excit;
    membrane_resistance_R_inhib = 1.0/ somatic_capcitance_Cm_inhib;

    membrane_time_constant_tau_m_excit = somatic_capcitance_Cm_excit / somatic_leakage_conductance_g0_excit;
    membrane_time_constant_tau_m_inhib = somatic_capcitance_Cm_inhib / somatic_leakage_conductance_g0_inhib;
    
    threshold_for_action_potential_spike = -0.053;
    
    v = restV0*ones(N,1);
    r = ones(N,1);
    tau_m = ones(N,1);

    for l = 1:nLayers 
        excit_begin = (ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1)+1;
        excit_end = ExcitDim*ExcitDim*l + (InhibDim*InhibDim)*(l-1);
        r(excit_begin:excit_end,1)=membrane_resistance_R_excit;
        tau_m(excit_begin:excit_end,1)=membrane_time_constant_tau_m_excit;
        
        inhib_begin = ExcitDim*ExcitDim*l + (InhibDim*InhibDim)*(l-1) + 1;
        inhib_end = (ExcitDim*ExcitDim+InhibDim*InhibDim)*l;
        r(inhib_begin:inhib_end,1)=membrane_resistance_R_inhib;
        tau_m(inhib_begin:inhib_end,1)=membrane_time_constant_tau_m_inhib;
    end

    I_rev=zeros(N,T+D);
    s_sum=zeros(N,T+D);
    I_tmp = zeros(N,1);
    
    group.firings=[];                             % spike timings
    last_fired=-T+zeros(N,1);               % assume that no neurons fired yet
    group.gr=[];                                  % the group connectivity will be here

    I_rev(v0+N*(t0-1))=0.001; %set input current
    recentTh = 0.002/timestep; %s
    
    
    for t=1:T
        %conductance_calculate_postsynaptic_current_injection_kernel
        I_tmp = I_rev(:,t) - s_sum(:,t).*v;
        v = timestep./tau_m(:,1).*(restV0+r(:,1).*I_tmp(:,1)) + (1 - (timestep./tau_m(:,1))).*v(:,1);
        
        fired = find(v>=threshold_for_action_potential_spike);                % indices of fired neurons

        %reset to resting pot.
        last_fired(fired)=t;
%         disp([t max(v) max(I_tmp) length(fired)]);
        v(fired)=restV0;
        
        

        for k=1:length(fired)
            
%             if(fired(k)>32*32 && fired(k)<32*32+16*16)
%                 disp('Inhibitory neuron activated')
%             end
            
%             I(pp{fired(k)}+t*N)=I(pp{fired(k)}+t*N)+s(fired(k),:);%increment by the size of synaptic weight
%             I_rev(pp{fired(k)}+t*N)=I_rev(pp{fired(k)}+t*N)+di(fired(k),:);%increment by the size of synaptic weight
%             s_sum(pp{fired(k)}+t*N)=s_sum(pp{fired(k)}+t*N)+s(fired(k),:);%increment by the size of synaptic weight
            pp_index = find(pp{fired(k)}>0);
            I_rev(pp{fired(k)}(pp_index)+(t)*N)=I_rev(pp{fired(k)}(pp_index)+(t)*N)+di(fired(k),pp_index);%increment by the size of synaptic weight
            s_sum(pp{fired(k)}(pp_index)+(t)*N)=s_sum(pp{fired(k)}(pp_index)+(t)*N)+s(fired(k),pp_index);%increment by the size of synaptic weight


%             tmp1 = find(post(fired(k),:)>0 & s(fired(k),:)>0 & di(fired(k),:)==0);
%             tmp2 = post(fired(k),:);
%             tmp3 = delay(fired(k),:);
%             plot([ones(1,length(tmp2(tmp1)))*t;t+tmp3(tmp1)],[ones(1,length(tmp2(tmp1)))*fired(k); tmp2(tmp1)]);
%             hold on;
            
            %The times of arrival of PSPs to this neuron
            PSP_times= last_fired(ppre{fired(k)}) + dpre{fired(k)};
            
            recent=find(PSP_times <= t & PSP_times > t-recentTh & di(pre{fired(k)})==0.0 & s(pre{fired(k)}) > 0);
            pprecentIDs = ppre{fired(k)}(recent);
            group.gr = [group.gr; last_fired(pprecentIDs),  pprecentIDs, ...  % presynaptic (time, neuron #)
                      last_fired(pprecentIDs) + dpre{fired(k)}(recent),...   % arrival of PSP (time)
                      fired(k)*(ones(length(recent),1)), ...                            % postsynaptic (neuron)
                      t*(ones(length(recent),1))];                                      % firing (time)

            group.firings=[group.firings; t, fired(k)];
        end;
        if(length(group.firings)>maxFiringCount)
            break;
        end
    %     [min(I(:)) max(I(:))]
    %     ['test'];
    end;
%     hold off;
    
    
    
elseif(neuronModel==model_izhikevich)
    
    restingPot = -60.0;
    v = -70*ones(N,1);                      % initial values
    u = 0.2.*v;                             % initial values
    I=zeros(N,T+D);

    group.firings=[];                             % spike timings
    last_fired=-T+zeros(N,1);               % assume that no neurons fired yet
    group.gr=[];                                  % the group connectivity will be here

    I(v0+N*(t0-1))=1000;                        % fire the anchor neurons at the right times
    recentTh = 10;
    %debug part begins
    % v_debug = [];
    % u_debug = [];
    % v_rec = zeros(N,T+D)+c;
    %debug part end

    for t=1:T
        v=v+0.5*((0.04*v+5).*v+140-u+ I(:,t));    % for numerical 
        v=v+0.5*((0.04*v+5).*v+140-u+ I(:,t));    % stability time 
        u=u+a.*(b*v-u);                   % step is 0.5 ms
        fired = find(v>=30);                % indices of fired neurons

    %     v_rec(:,t) = v(:,1);

    %     v_debug = [v_debug v(v0(1))];
    %     u_debug = [u_debug u(v0(1))];
    %     if t > t0(1)
    %          listPostCells = post(v0(1),(s(v0(1),:)>0.9));
    %          reshape(v(listPostCells),1,[])
    %     end

    %     length(fired)

        %reset to resting pot.
        v(fired)=c;  
        u(fired)=u(fired)+d(fired);
        last_fired(fired)=t;


        for k=1:length(fired)
%             I(pp{fired(k)}+t*N)=I(pp{fired(k)}+t*N)+s(fired(k),:);%increment by the size of synaptic weight
            pp_index = find(pp{fired(k)}>0);
            I(pp{fired(k)}(pp_index)+(t-1)*N)=I(pp{fired(k)}(pp_index)+(t-1)*N)+s(fired(k),pp_index);%increment by the size of synaptic weight

            %The times of arrival of PSPs to this neuron
            PSP_times= last_fired(ppre{fired(k)}) + dpre{fired(k)};
            recent=find(PSP_times < t & PSP_times > t-recentTh & s(pre{fired(k)}) > 0 );
            pprecentIDs = ppre{fired(k)}(recent);
            group.gr = [group.gr; last_fired(pprecentIDs),  pprecentIDs, ...  % presynaptic (time, neuron #)
                      last_fired(pprecentIDs) + dpre{fired(k)}(recent),...   % arrival of PSP (time)
                      fired(k)*(ones(length(recent),1)), ...                            % postsynaptic (neuron)
                      t*(ones(length(recent),1))];                                      % firing (time)

            group.firings=[group.firings; t, fired(k)];
        end;
    %     [min(I(:)) max(I(:))]
    %     ['test'];
        if(length(group.firings)>maxFiringCount)
            break;
        end
    end;

end

%figure;
% v0
% t0
% plot(1:T,[v_debug;u_debug]);
% xlim([1, T])
% v_debug;