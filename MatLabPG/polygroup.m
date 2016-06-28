% Find all polychronous groups that start with the anchor neurons v0 fired
% at the times t0. Typically, should be called from the polychron.m
% (Kalman Katlowitz fixed a bug related to new MATLAB release in 2011)

function group=polygroup(v0,t0)
global a d N D pp s post ppre dpre pre delay T


v = -70*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
I=zeros(N,T+D);

group.firings=[];                             % spike timings
last_fired=-T+zeros(N,1);               % assume that no neurons fired yet
group.gr=[];                                  % the group connectivity will be here

I(v0+N*t0)=1000;                        % fire the anchor neurons at the right times

for t=1:T    
    v=v+0.5*((0.04*v+5).*v+140-u+ I(:,t));    % for numerical 
    v=v+0.5*((0.04*v+5).*v+140-u+ I(:,t));    % stability time 
    u=u+a.*(0.2*v-u);                   % step is 0.5 ms
    fired = find(v>=30);                % indices of fired neurons

    v(fired)=-65;  
    u(fired)=u(fired)+d(fired);
    last_fired(fired)=t;
    
    for k=1:length(fired)
        I(pp{fired(k)}+t*N)=I(pp{fired(k)}+t*N)+s(fired(k),:);
        
        %The times of arrival of PSPs to this neuron
        %PSP_times= last_fired(ppre{fired(k)}) + dpre{fired(k)}';
        PSP_times= last_fired(ppre{fired(k)}) + dpre{fired(k)};
        %recent=find(PSP_times < t & PSP_times > t-10 & s(pre{fired(k)})' > 0 );      % Select those that are relevant
        recent=find(PSP_times < t & PSP_times > t-10 & s(pre{fired(k)}) > 0 );
        group.gr = [group.gr; last_fired(ppre{fired(k)}(recent)),  ppre{fired(k)}(recent), ...  % presynaptic (time, neuron #)
                  last_fired(ppre{fired(k)}(recent)) + dpre{fired(k)}(recent),...   % arrival of PSP (time)
                  fired(k)*(ones(length(recent),1)), ...                            % postsynaptic (neuron)
                  t*(ones(length(recent),1))];                                      % firing (time)
               
        group.firings=[group.firings; t, fired(k)];
    end;
end;   