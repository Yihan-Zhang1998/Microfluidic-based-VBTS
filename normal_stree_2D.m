%% 2D simulation
% Always utilize SI units
clc
clear
close all
date = datestr(now,'yyyymmdd_HHMMSS');
gpuDeviceTable
GPU1 = gpuDevice(1);
%% Import image for hologram recording
msource = imresize(imread('msource.png'),1);
sensor = imresize(imread('sensing_unit_K1D4.png'),1);
%% Initialize video
movvar=1;
fps=30;
if movvar==1
    writerObj = VideoWriter(sprintf('%s.mp4',date),'MPEG-4' );
    writerObj.FrameRate = fps;
    open(writerObj);
end

%% Universal constants
epsilon0=8.854187817e-12;
mu0=12.566370e-7;
c=1/sqrt(epsilon0*mu0);

%% Plot definitions
brightness=0.1;  % brightness of plot
nt=1000; % total number os time steps
waitdisp=20; % wait to display

lengthx=5e-3;  % size of the image in x axis SI
lengthy=lengthx;  % size of the image in y axis SI

[sy,sx]=size(msource(:,:,1));

emkc = zeros(1000,3); % EM colormap
emkc(1,:) = [1 0 0];
emkc(1000,:) = [0 0 1];
r = linspace(1,0,499);
for j = 1:length(r)
    emkc(j,1) = r(j);
end
b = linspace(0,1,500);
for j = 1:length(b)
    emkc(500+j,3) = b(j);
end

%% Source constants
wavelength=355e-9;
omega=2*pi()*c/wavelength;

%% Cell size
dx=lengthx/sx;  %cell size in x
dy=lengthy/sy;  %cell size in y

%% Coordinates definition
xmin=-lengthx/2;
ymin=-lengthy/2;
xmax=xmin+lengthx;
ymax=ymin+lengthy;

x=(xmin+dx/2:dx:xmax-dx/2);
y=flip((ymin+dx/2:dy:ymax-dy/2));

[X,Y]=meshgrid(x,y);

%% Source Mask
source=round(double(msource(:,:,1))/255);

%% Permittivity
obj1 = sensor;
obj1=round(double(obj1(:,:,2))/255);
RI0=1;  %Background refractive index
alpha1=log(2)/100e-9;  %Absorbtion coeficient with half loss in 100 nm
RI1=1+1i*c*alpha1/2/omega; %Refractive indexwith absorption coeficient
RI2=1.4+1i*2.5e-2;    %Refractive index of sensor

sabs=10;    %Thickness of the absorbing layer in pixels (sigma of Gaussian)
alphab=log(100)/sabs/dx/0.3;    %Absorption after crossing the layer
gsf=repmat(exp(-(-(sx-1)/2:(sx-1)/2).^2/sabs.^2),sy,1).*repmat(exp(-(-(sy-1)/2:(sy-1)/2).^2/sabs.^2).',1,sx);
absmask=ones(size(X));
absmask(2:end-1,2:end-1)=zeros(size(X)-2);
absmask=ifftshift(abs(ifft2(fft2(absmask).*fft2(gsf))));
absmask=absmask/max(absmask(:));

RIbackground=1i*absmask*c*alphab/2/omega;    %Imaginary part
RIbackground=RIbackground+sqrt(real(RI0).^2-imag(RI0).^2+imag(RIbackground).^2);   %Real part to maintain speed

RIall=RIbackground;
RIall=RIall.*(1-obj1)+RI2.*obj1;
% RIall=RIall.*(1-obj2)+RIAl.*obj2;

epsilonc=epsilon0*RIall.^2;
epsilon=abs(real(epsilonc));    %Real permittivity
sigma=omega*imag(epsilonc); %Imaginary permittivity (Conductivity)
mu=mu0*ones(size(X));

%% Time step
cfl=1/sqrt(2); %Condition in Fourier domain
dt=cfl*min(real(sqrt(epsilon(:).*mu(:))))*0.5*dx; %Courant criteria for minimum timestep

%% Initialization
Hx=zeros(size(X));
Hy=zeros(size(X));
Ez=zeros(size(X));

JEz=zeros(size(Ez));

ceb=1./(1+(sigma.*dt./(2*epsilon)));
cea=(1-(sigma.*dt./(2*epsilon))).*ceb;

chbx=1./(1+(sigma.*dt./(2*mu)));
chax=(1-(sigma.*dt./(2*mu))).*chbx;  

chby=1./(1+(sigma.*dt./(2*mu)));
chay=(1-(sigma.*dt./(2*mu))).*chby;  

%% Fourier constants for derivative
xk = -1i*2*pi()*fftshift(-sx/2:sx/2-1)/sx;
yk = flip(-1i*2*pi()*fftshift(-sy/2:sy/2-1)/sy);
[Xk,Yk]=meshgrid(xk,yk);

%% Convert to GPU object
GHx = gpuArray(Hx);
GHy = gpuArray(Hy);
GEz = gpuArray(Ez);
GJEz = gpuArray(JEz);
Gchax = gpuArray(chax);
Gchbx = gpuArray(chbx);
Gchay = gpuArray(chay);
Gchby = gpuArray(chby);
Gcea = gpuArray(cea);
Gceb = gpuArray(ceb);

%% This is the field propagation loop
ki=0;
figure(1)

nt_end = 3500;
exposure_intensity = zeros(size(X));
tic
while ki<nt
    ki=ki+1;
    %Define source
    if ki <= nt_end
        GJEz=source*cos(omega*dt*ki);    
    else
        GJEz = 0;
    end
    
    %Update of Ez
    DyHx=real(ifft2(Yk.*fft2(GHx))/dy); % Fourier derivative
    DxHy=real(ifft2(Xk.*fft2(GHy))/dx); 
    
    GEz  = Gcea .* GEz   + Gceb .* (dt./epsilon) .* (DyHx-DxHy) + GJEz;
    
    %Update of Hx,Hy    
    DyEz=real(ifft2(Yk.*fft2(GEz))/dy); 
    DxEz=real(ifft2(Xk.*fft2(GEz))/dx);
    
    GHx=Gchax .* GHx + Gchbx .* (dt./mu).*(DyEz);    
    GHy=Gchay .* GHy + Gchby .* (dt./mu).*(-DxEz);
        
    if ~mod(ki,waitdisp)

        wait(GPU1);
        disp(['Time per cycle: ' num2str(toc/waitdisp) ' s']);
        clf

        Ez = gather(GEz);
        Hx = gather(GHx);
        Hy = gather(GHy);
        
        imagesc(x,y,Ez,[-1 1]/brightness)  
        axis image
        title(sprintf('TimeFrame %s',num2str(ki)))
        
        set(gca,'ydir','normal' )
        colormap(gca,emkc)
        alpha(1-obj1*0.1)
        drawnow
        if movvar==1
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
        end
        tic
    end
    
end

 

%% Close video
if movvar==1
    close(writerObj);
end

