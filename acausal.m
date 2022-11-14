function [accff] = acausal(LP,HP,nroll,acc,dt)
%ACAUSAL Summary of this function goes here
%   Detailed explanation goes here
sampfreq = 1/dt;
Wn = [2*HP/sampfreq,2*LP/sampfreq];
%%% butterworth filter design
%%% If Wn is a two-element vector, Wn = [w1 w2], 
%%% butter returns an order 2*n digital bandpass filter 
%%% with passband w1 < ? < w2

 
option=1;
if option==1
%     disp(sprintf('Wn=[%f,%f], nroll=2 used in acausal filter',Wn))
    %%[b,a] = butter(2*nroll,Wn,'stop');  %%% stop passing through 
    order=4;  %%% nroll=2
    [b,a] = butter(order,Wn); 
    accff = filtfilt(b,a,double(acc));
elseif option==2
%     disp(sprintf('Wn=[%f,%f], nroll=%f used in acausal filter',Wn, nroll))
    %%%% high pass first, then low pass
    Wn = [2*HP/sampfreq];
    [b,a] = butter(2*nroll,Wn,'high');  %%% pass through
    accff = filtfilt(b,a,double(acc));
    Wn = [2*LP/sampfreq];
    [b,a] = butter(2*nroll,Wn,'low');  %%% pass through
    accff = filtfilt(b,a,double(accff));
elseif option==3
%     disp(sprintf('Wn=[%f,%f], nroll=%f used in acausal filter',Wn, nroll))
    %%%% low pass first, then high pass
    Wn = [2*LP/sampfreq];
    [b,a] = butter(2*nroll,Wn,'low');  %%% pass through
    accff = filtfilt(b,a,double(acc));
    Wn = [2*HP/sampfreq];
    [b,a] = butter(2*nroll,Wn,'high');  %%% pass through
    accff = filtfilt(b,a,double(accff));
elseif option==4
%     disp(sprintf('Option 4:::Wn=[%f,%f], nroll=%f used in acausal filter',Wn, nroll))
    %%%% high pass first, then low pass
    Wn = [2*HP/sampfreq,2*LP/sampfreq];
    order=4;
    [b,a] = butter(order,Wn);  
    accff1 = filter(b,a,double(acc));
    accff_reverse=fliplr(accff1);
    accff2 = filter(b,a,double(accff_reverse));
    accff=fliplr(accff2);
end

%%%% nroll*2 = order
%%%% nroll=slope/4 for acausal
%%%% slope=2*order

