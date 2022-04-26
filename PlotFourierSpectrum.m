%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CIVL5750 Homework 2 
%%% Matlab code to compute Fourier amplitude spectrum
%%% Gang Wang, HKUST 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
function [outf,outh]=PlotFourierSpectrum(acc, dt) 
%%% input 
%%%% AccFileName: File name (character) for data in PEER format
%%%% dt is obtained automatically from the PEER format
%%%% can not work for other format
%%%% example: 
%%%% PlotResponseSpectrum('YBI090.AT2')
%%%% output:
%%%% f: frequency (hz); h: Fourier amplitude

 %AccFileName=['G01090.AT2']; % Loma Prieta, Gilroy No. 1 (rock) E-W components
 %AccFileName=['G02090.AT2']; % Loma Prieta, Gilroy No. 2 (soil) E-W components


%%%%%%%%%%%%%%%%%%%%%%%%% no need change below %%%%%%%%%%%%%%%%%%%

np = length(acc);
 
 %%%% reinterprete dt=0.02 sec. 
 Newdt=0.02;
 Newnp=floor(np*dt/Newdt);
 acc=interp1([1:np]*dt, acc,[1:Newnp]*Newdt);
 dt=Newdt;
 np=Newnp;

time=[0:np-1]*dt;
 
%%% check if delta_f =< 0.05 Hz as required by Rathje for accuracy
%%% To ensure a stable value of Tm is calculated for recorded strong 
%%% ground motions, motions should contain at least the minimum number 
%%% of points indicated above or should be augmented with zeroes to
%%% attain these minimum values.
%%% delta_f=1/(np*dt)=< 0.05 Hz, therefore, 
%%% requires np*dt>20, i.e., np>20/dt; otherwise, append zeros in the back

%%% input: acc -- time sequence; np -- no. of points in acc; dt--delta time
 NFFT = 2^nextpow2(np); 

 Y = fft(acc,NFFT);       %% FFT(X,N) is the N-point FFT, padded with zeros if X has less
                          %% than N points and truncated if it has more
                          % changed by G. Wang, Fourier coefficient in complex form, unit g-sec
%%%%%%%%%%%%%%%%%%%%%%%%%%%
 h=2*abs(Y(1:NFFT/2+1));  % this is Fourier coefficient. Norm of complex number, unit g-sec
                          % multiple by factor of 2, cf. Kramer, p.540, eq. (A17)
 hh=h.^2;                 % Ci^2 in Rathje paper
 F=1/dt;
 f=F/2*linspace(0,1,NFFT/2+1); % the maximum frequency of FFT is related to delta time
 f=f';                    % this is Fourier frequency
 
 df=1/(NFFT*dt);
 
% figure(); 
% 
% subplot(3,1,1)
% plot(time, acc,'r'); hold on;
% xlabel('Time (sec)'); ylabel('Acc (g)');
% 
% subplot(3,1,2)
% plot(f,h);  xlim([0 25])
% xlabel('Freq. (HZ)'); ylabel('Fourier Amplitude (g*sec)');
% 
% subplot(3,1,3)
% plot(1./f,h);  xlim([0 10])
% xlabel('Period (sec)'); ylabel('Fourier Amplitude (g*sec)');


if 1==2

	% inverse FFT to recover time history
	acc_ifft=ifft(Y,NFFT);
	dt=1/(df*NFFT);
	time_ifft=[0:NFFT-1]*dt;  % the sequence is appended with zeros in back

	figure;
	plot(time_ifft,acc_ifft,'b')

end
% h = kohmachi(h', f', 100);
h = smoothdata(h, 'movmean', 5);
outf = logspace(log10(0.1), log10(50), 1000);
outh = interp1(f, h, outf);
