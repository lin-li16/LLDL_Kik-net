%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CIVL5750 Homework 2 
%%% Matlab code to compute acceleration response spectrum
%%% Gang Wang, HKUST Mar. 08, 2011
%%% add comments on Mar. 25, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
function PSA=getResponseSpectrum(acc, dt) 
 
%%%% input 
%%%% AccFileName: File name (character) for data in PEER format
%%%% dt is obtained automatically from the PEER format
%%%% can not work for other format
%%%% example: 
%%%% PlotResponseSpectrum('YBI090.AT2')
%%%% output:
%%%% PSA: psudo-spectral acceleration

 damp=0.05;  % use 5 percent damping
 %AccFileName=['TRI090.AT2']; % Loma Prieta, Treasure Island, E-W component
 %AccFileName=['YBI090.AT2']; % Loma Prieta, Yerba Buena Island, E-W component
  
 % specify different periods to calculate spectrum
Period = logspace(-2, 1, 101);


%%%%%%%%%%%%%%%%% no need to change below %%%%%%%%%%%%%%%%%

% read in Strong Motion format
% % %  data=load(AccFileName);
% % %  
% % %  acc=data(:,2);
% % %  dt=data(2,1)-data(1,1);
% % %  np=length(acc);
 
 
 %%% integrate time histories
 time=[1:length(acc)]*dt;
 % acc=acc'; % acc is now a row vector
 acc2=[0, acc(1:(length(acc)-1))];
 accAvg= (acc+acc2)/2;
 vel= cumsum(accAvg).*dt .*981;  % in unit of cm/s
 velAvg= vel + (acc./3+acc2./6).*dt.*981 ;
 displ = cumsum(velAvg).* dt;
 
 PGA=max(abs(acc));   % in unit of g
 PGV=max(abs(vel));   % in unit of cm/s
 PGD=max(abs(displ)); % in unit of cm
 
 Ia_time=pi/2*cumsum(abs(acc).^2)*dt; % unit g-sec  %%% corrected Apr. 25, 2017
 
 

%%%%%%%%%%%%% Calculate spectra STARTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:length(Period)
    T=Period(i);
    Omega=2*pi/T;
    %%%%%%%% use algorithm from SPECTAE (Geomatrix) %%%%%%
    kug=length(time)-1;    
    if T < 10*dt   % stiff system, reduce time step
        [z]=ucmpmx(kug,acc,time,T,Omega,damp);
    else
        [z]=cmpmax(kug,acc,T,Omega,damp,dt);
    end
    rd(i)=z(1);  % max relative disp. for period T, disp response spectrum
    rv(i)=z(2);  % max relative vel. for period T, vel response spectrum
    aa(i)=z(3);  % max absolute acc. for period T, Acc response spectrum
    PsedoSV(i)=2*pi/T*rd(i);      % psedo-velocity response spectrum
    PsedoSA(i)=(2*pi/T)^2*rd(i);  % psedo-acceleration response spectrum
end
%%%%%%%% use algorithm from SPECTAE (Geomatrix) END %%%%%%


%%%%% the final results %%%%%%%%%%%%%%
Sa=aa;                  % max absolution acc, unit: g
Sv=rv *981;             % max relative vel.,  unit: cm/s
Sd=rd *981;             % max relative disp., unit: cm
PSV=PsedoSV *981;       % Psedo-velocity,     unit: cm/s
PSA=PsedoSA;            % Psedo-acceleration, unit: g


% compute response of single-DOF system under arbituary time step
function [z]=ucmpmx(kug,ug,time,pr,w,d)
 % inputs:
 %    kug -- number of time increment
 %    ug -- input ground acceleration
 %    time -- time sequence
 %    pr -- period at which spectra are calculated
 %    w -- frequency
 %    d -- damping ratio, 
 % outputs:
 %    z - z(1);   maximum relative displacement
 %        z(2);   maximum relative velocity
 %        z(3);   maximum absolute acceleration
 %    x is time history, not outputed yet

  wd=sqrt(1.-d*d)*w;
  w2=w*w;
  w3=w2*w;
  for i=1:3
     x(1,i)=0.0;
     z(i)=0.0;
  end
  f2=1./w2;
  f3=d*w;
  f4=1./wd;
  f5=f3*f4;
  f6=2.*f3;
  for k=1:kug
        dt=time(k+1)-time(k);
        ns=round(10.*dt/pr-0.01);
        dt=dt/real(ns);  % reduce time step for STIFF system
        f1=2.*d/w3/dt;
        e=exp(-f3*dt);
        g1=e*sin(wd*dt);
        g2=e*cos(wd*dt);
        h1=wd*g2-f3*g1;
        h2=wd*g1+f3*g2;
        dug=(ug(k+1)-ug(k))/real(ns);
        g=ug(k);
        z1=f2*dug;
        z3=f1*dug;
        z4=z1/dt;
        for is=1:ns      % march over reduced substeps
          z2=f2*g;
          b=x(1,1)+z2-z3;
          a=f4*x(1,2)+f5*b+f4*z4;
          x(2,1)=a*g1+b*g2+z3-z2-z1;
          x(2,2)=a*h1-b*h2-z4;
          x(2,3)=-f6*x(2,2)-w2*x(2,1);
          for l=1:3
            c(l)=abs(x(2,l));
            if(c(l)>=z(l)) 
              z(l)=c(l);
              t(l)=time(k)+is*dt;
            else
            end
            x(1,l)=x(2,l);
          end
          g=g+dug;
        end
  end
 
 
 
 % compute response of single-DOF system under fixed time step
 function [z]=cmpmax(kug,ug,pr,w,d,dt)
  % inputs:
  %    kug -- number of time increment
  %    ug -- input ground acceleration
  %    pr -- period at which spectra are calculated
  %    w- frequency
  %    d -- damping ratio, 
  %    dt -- time step, 
  % outputs:
  %    z - z(1);   maximum relative displacement
  %        z(2);   maximum relative velocity
  %        z(3);   maximum absolute acceleration
  %    x is time history, not outputed yet
  
       wd=sqrt(1.-d*d)*w;
       w2=w*w;
       w3=w2*w;
       for i=1:3
         x(1,i)=0.0;
         z(i)=0.0;
       end
       f1=2.*d/(w3*dt);
       f2=1./w2;
       f3=d*w;
       f4=1./wd;
       f5=f3*f4;
       f6=2.*f3;
       e=exp(-f3*dt);
       g1=e*sin(wd*dt);
       g2=e*cos(wd*dt);
       h1=wd*g2-f3*g1;
       h2=wd*g1+f3*g2;
       for k=1:kug
         dug=ug(k+1)-ug(k);
         z1=f2*dug;
         z2=f2*ug(k);
         z3=f1*dug;
         z4=z1/dt;
         b=x(1,1)+z2-z3;
         a=f4*x(1,2)+f5*b+f4*z4;
         x(2,1)=a*g1+b*g2+z3-z2-z1;   % relative disp.
         x(2,2)=a*h1-b*h2-z4;         % relative vel.
         x(2,3)=-f6*x(2,2)-w2*x(2,1); % absolute acc.
         
         % find the maximum of each
          for l=1:3
           c(l)=abs(x(2,l));
           if(c(l)>=z(l)) 
             z(l)=c(l);
             t(l)=dt*real(k);
           else
           end
           x(1,l)=x(2,l);
          end
      end