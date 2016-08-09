% Find all polychronous groups that start with the anchor neurons v0 fired
% at the times t0. Typically, should be called from the polychron.m
% (Kalman Katlowitz fixed a bug related to new MATLAB release in 2011)

function group=polygroup(v0,t0)
global a b c d N D pp s post ppre dpre pre delay T


v = -70*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
I=zeros(N,T+D);

group.firings=[];                             % spike timings
last_fired=-T+zeros(N,1);               % assume that no neurons fired yet
group.gr=[];                                  % the group connectivity will be here

I(v0+N*(t0-1))=1000;                        % fire the anchor neurons at the right times
recentTh = 20;
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
        I(pp{fired(k)}+t*N)=I(pp{fired(k)}+t*N)+s(fired(k),:);%increment by the size of synaptic weight
%        I(1 + pp{fired(k)}+(t-1)*N)=I(1 + pp{fired(k)}+(t-1)*N)+s(fired(k),:);%increment by the size of synaptic weight
        
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
end;

%figure;
% v0
% t0
% plot(1:T,[v_debug;u_debug]);
% xlim([1, T])
% v_debug;