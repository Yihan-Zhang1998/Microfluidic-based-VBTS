%% 3D simulation of VBTS
% Always utilize SI units

clear
close all
clc
date = datestr(now,'yyyymmdd_HHMMSS');
gpuDeviceTable
GPU1 = gpuDevice(1);
%% Load 3d model for mirror
minx=-25;
maxx=25;
miny=-25;
maxy=25;
minz=-25;
maxz=25;
tp=500;
stp=min([abs(maxx-minx),abs(maxy-miny),abs(maxz-minz)])/tp;
x=minx:stp:maxx;
y=miny:stp:maxy;
z=minz:stp:maxz;

[mx,my,mz]=meshgrid(x,y,z);
points=[mx(:),my(:),mz(:)];
% S_mirror=stlread('Ball_R100.stl');
% mirror = inpolyhedron(S_mirror,x,y,z);

%% Load 3d model for sensor
S_sensor=stlread('sensing_unit_K1D4.stl');
sensor = inpolyhedron(S_sensor,x,y,z);

%% Extract surface groups
S_PDMS = extractMeshSurfaces(S_sensor);
PDMS = inpolyhedron(S_PDMS,x,y,z);

%% Load 3d model for light source
S_msource=stlread('msource.stl');
msource = inpolyhedron(S_msource, x, y, z);

%% Plot
figure, hold on, view(3)        % Display the result
set(gcf, 'Position', get(gcf, 'Position').*[0 0 1.5 1.5])
patch(S_sensor,'FaceColor','g','FaceAlpha',0.2)
hold on
patch(S_msource,'FaceColor','r','FaceAlpha',0.2)
%legend({'volume', 'points inside', 'points outside'}, 'Location', 'southoutside')
axis image
axis off

%% Initialize video
movvar=1;
fps=15;
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
brightness=100;  %brigthness of plot
nt=100; %total number os time steps
waitdisp=20; %wait to display

lengthx=5e-3;  %size of the image in x axis SI
lengthy=lengthx;  %size of the image in y axis SI
lengthz=lengthx;  %size of the image in z axis SI

dims = size(msource);
sx = dims(1);
sy = dims(2);
sz = dims(3);

emkc = zeros(900,3); % EM colormap
r = sin((1:1:450)./2*pi()./(450/2)); % linspace(1,0,299);
for j = 1:length(r)
    emkc(j,1) = r(j);
end
b = -sin((451:1:900)./2*pi()./(450/2)); % linspace(0,1,500);
for j = 1:length(b)
    emkc(450+j,3) = b(j);
end

%% Source constants
wavelength=450e-9;
omega=2*pi()*c/wavelength;

%% Cell size
dx=lengthx/sx;  %cell size in x
dy=lengthy/sy;  %cell size in y
dz=lengthz/sz;  %cell size in z

%% Coordinates definition
xmin=-lengthx/2;
ymin=-lengthy/2;
zmin=-lengthz/2;
xmax=xmin+lengthx;
ymax=ymin+lengthy;
zmax=zmin+lengthz;

x=(xmin+dx/2:dx:xmax-dx/2);
y=(ymin+dy/2:dy:ymax-dy/2);
z=(zmin+dz/2:dz:zmax-dz/2);

[X,Y,Z]=meshgrid(x,y,z);

%% Permittivities
RI0=1;  %Background refractive index
RI_soln = 1.33334;
RI_PDMS = 1.4; %Refractive index of hydrogel

%% Objects
RIobj = [RI_soln, RI_PDMS];

obj(:,:,:,1) = double(sensor);
obj(:,:,:,2) = double(PDMS);

%% Adding objects
objs = sum(obj,4);
objb = 1 - objs.*double(objs<1) - double(objs>=1);
objt = objs.*double(objs>1) + double(objs<=1|objs<=0);

RIbackground = RI0;
RIall = zeros(size(X));
for ki = 1:min(size(obj,4),size(RIobj,2))
    RIall = RIall + RIobj(ki).*obj(:,:,:,ki).*objt;
end
RIall = objb.*RIbackground+RIall;

%% Absorbing boundary

sabs=10;    %Thickness of the absorbing layer in pixels (sigma of Gaussian)
alphab=log(100000)/sabs/dx/0.3;    %Absorption after crossing the layer
gsf_x = zeros(1,sx,1);
gsf_x(1,:,1) = sabs - (1:sx);
gsf_x = sabs - gsf_x.*double(gsf_x>0);
gsf_y = zeros(sy,1,1);
gsf_y(:,1,1) = sabs - (1:sy).';
gsf_y = sabs - gsf_y.*double(gsf_y>0);
gsf_z = zeros(1,1,sz);
gsf_z(1,1,:) = sabs - (1:sz);
gsf_z = sabs - gsf_z.*double(gsf_z>0);
gsf_x = repmat(gsf_x,sy,1,sz)/sabs;
gsf_y = repmat(gsf_y,1,sx,sz)/sabs;
gsf_z = repmat(gsf_z,sy,sx,1)/sabs;
absmask = 1-gsf_x.*flip(gsf_x,2).*gsf_y.*flip(gsf_y,1).*gsf_z.*flip(gsf_z,3);
absbackground=1i*1./(1-absmask)/sabs/(4*pi()*dx/wavelength).*double(absmask>0); %Imaginary part

RIbackground=1i*imag(RIall).*double(imag(RIall)>=imag(absbackground));%Mantain absorbing parts of design
RIbackground=RIbackground+absbackground.*double(imag(RIall)<imag(absbackground));%Add absorbing boundaries
RIall=RIbackground+sqrt(real(RIall.^2)+imag(RIbackground).^2);   %Real part to maintain speed

epsilonc=epsilon0*RIall.^2;
epsilon=abs(real(epsilonc));    %Real permittivity
sigma=omega*imag(epsilonc); %Imaginary permittivity (Conductivity)
mu=mu0*ones(size(X));

%% Time step
cfl=1/sqrt(3); %Condition in Fourier domain
dt=cfl*min(sqrt(epsilon(:).*mu(:)))*0.5*dx; %Courant criteria for minimum timestep

%% Initialization
Ex=zeros(size(X));
Ey=zeros(size(X));
Ez=zeros(size(X));

Hx=zeros(size(X));
Hy=zeros(size(X));
Hz=zeros(size(X));

% definition of the matrices for the current like sources
source = double(msource);
JEx=source*1;
JEy=source*0;
JEz=source*0;

%% This is the field propagation loop

Ez0=Ez;
Ez1=Ez;
Ez2=Ez;

Hx0=Hx;
Hx1=Hx;
Hx2=Hx;
Hy0=Hy;
Hy1=Hy;
Hy2=Hy;
Hz0=Hz;
Hz1=Hz;
Hz2=Hz;

%% Fourier constants for derivative
xk = 1i*2*pi()*fftshift(-sx/2:sx/2-1)/sx;
yk = 1i*2*pi()*fftshift(-sy/2:sy/2-1)/sy;
zk = 1i*2*pi()*fftshift(-sz/2:sz/2-1)/sz;
[Xk,Yk,Zk]=meshgrid(xk,yk,zk);
Xkdx=Xk/dx;
Ykdy=Yk/dy;
Zkdz=Zk/dz;

XYZk2=(Xk.^2+Yk.^2+Zk.^2);

%% Contstants
sigx=(dt/dx).^2./epsilon./mu;
gamx=dt*sigma./epsilon;
exp1=exp(-1i*omega*dt);

% curls
cJEx=ifftn(Ykdy.*fftn(JEz)-Zkdz.*fftn(JEy));
cJEy=ifftn(Zkdz.*fftn(JEx)-Xkdx.*fftn(JEz));
cJEz=ifftn(Xkdx.*fftn(JEy)-Ykdy.*fftn(JEx));

c1=exp1*2./(1+gamx./2);
c2=exp1*exp1*(gamx/2-1)./(gamx./2+1);
c3=exp1*sigx./(1+gamx./2);
% c4Ez=exp1*JEz*dt./(1+gamx./2);

c4Hx=exp1*cJEx*dt./(1+gamx./2);
c4Hy=exp1*cJEy*dt./(1+gamx./2);
c4Hz=exp1*cJEz*dt./(1+gamx./2);

%% Convert to GPU object
GXkdx = gpuArray(Xkdx);
GYkdy = gpuArray(Ykdy);
GZkdz = gpuArray(Zkdz);
GXYZk2 = gpuArray(XYZk2);

GHx0 = gpuArray(Hx0);
GHy0 = gpuArray(Hy0);
GHz0 = gpuArray(Hz0);
GHx1 = gpuArray(Hx1);
GHy1 = gpuArray(Hy1);
GHz1 = gpuArray(Hz1);
GHx2 = gpuArray(Hx2);
GHy2 = gpuArray(Hy2);
GHz2 = gpuArray(Hz2);

Gc1 = gpuArray(c1);
Gc2 = gpuArray(c2);
Gc3 = gpuArray(c3);

Gc4Hx = gpuArray(c4Hx);
Gc4Hy = gpuArray(c4Hy);
Gc4Hz = gpuArray(c4Hz);

% Gexp1 = gpuArray(exp1);
% GJEx = gpuArray(JEx);
% GJEy = gpuArray(JEy);
% GJEz = gpuArray(JEz);
% GcJEx = gpuArray(cJEx);
% GcJEy = gpuArray(cJEy);
% GcJEz = gpuArray(cJEz);
% Ggamx = gpuArray(gamx);
figure;
ki=0;
overallintensity=0;
tic
set(gcf, 'Position',[1 49 1536 740.8])
pos = get(gcf, 'Position');

while ki<nt
    
    ki=ki+1;
    
    if ki > 1000
%         GJEx = 0*source;
%         GJEy = 0*source;
%         GJEz = 0*source;
%         GcJEx=ifftn(GYkdy.*fftn(GJEz)-GZkdz.*fftn(GJEy));
%         GcJEy=ifftn(GZkdz.*fftn(GJEx)-GXkdx.*fftn(GJEz));
%         GcJEz=ifftn(GXkdx.*fftn(GJEy)-GYkdy.*fftn(GJEx));
        Gc4Hx = 0;% Gexp1*GcJEx*dt./(1+Ggamx./2);
        Gc4Hy = 0;% Gexp1*GcJEy*dt./(1+Ggamx./2);
        Gc4Hz = 0;% Gexp1*GcJEz*dt./(1+Ggamx./2);
    end

    GHx0=GHx1;
    GHx1=GHx2;        
    GHx2= GHx1.*Gc1 + GHx0.*Gc2 + ifftn(GXYZk2.*fftn(GHx1)).*Gc3 + Gc4Hx;
    
    GHy0=GHy1;
    GHy1=GHy2;       
    GHy2= GHy1.*Gc1 + GHy0.*Gc2 + ifftn(GXYZk2.*fftn(GHy1)).*Gc3 + Gc4Hy;
    
    GHz0=GHz1;
    GHz1=GHz2;       
    GHz2= GHz1.*Gc1 + GHz0.*Gc2 + ifftn(GXYZk2.*fftn(GHz1)).*Gc3 + Gc4Hz;

    if mod(ki,waitdisp)==0 || ki == 1
        wait(GPU1);
        GEx=ifftn(GYkdy.*fftn(GHz2)-GZkdz.*fftn(GHy2));
        GEy=ifftn(GZkdz.*fftn(GHx2)-GXkdx.*fftn(GHz2));
        GEz=ifftn(GXkdx.*fftn(GHy2)-GYkdy.*fftn(GHx2));
        GHx=ifftn(GZkdz.*fftn(GEy)-GYkdy.*fftn(GEz));
        GHy=ifftn(GXkdx.*fftn(GEz)-GZkdz.*fftn(GEx));
        GHz=ifftn(GYkdy.*fftn(GEx)-GXkdx.*fftn(GEy));
        
        wait(GPU1);
        disp(['Time per cycle: ' num2str(toc/waitdisp) ' s']); 

        Ex = gather(GEx);
        Ey = gather(GEy);
        Ez = gather(GEz);
        Hx = gather(GHx);
        Hy = gather(GHy);
        Hz = gather(GHz);
        E(:,:,:,1) = Ex;
        E(:,:,:,2) = Ey;
        E(:,:,:,3) = Ez;
        H(:,:,:,1) = Hx;
        H(:,:,:,2) = Hy;
        H(:,:,:,3) = Hz;

        intensity = abs(Ex).^2+abs(Ey).^2+abs(Ez).^2;
        if ki == 1000
            overallintensity = intensity;
        end

        clf     
        
        Efield = sum(E,4);
        Hfield = sum(H,4);
        
%         subplot(2,3,1)
%         imagesc(z,y,real(squeeze(Efield(round(sx/2),:,:))),[-1 1]./brightness)
%         axis image
%         set(gca,'ydir','normal')
%         colormap(gca,emkc)
%         alpha(1-squeeze(sensor(round(sx/2),:,:))*0.1-squeeze(mirror(round(sx/2),:,:))*0.1)
%         xlabel('z ($\mu$m)','Interpreter','latex')
%         ylabel('y ($\mu$m)','Interpreter','latex')
%         title('Electrical Field on y-z plane')
% 
%         subplot(2,3,4)
%         imagesc(z,x,real(squeeze(Efield(:,round(sy/2),:))),[-1 1]./brightness)
%         axis image
%         set(gca,'ydir','normal')
%         colormap(gca,emkc)
%         alpha(1-squeeze(sensor(:,round(sy/2),:))*0.1-squeeze(mirror(:,round(sy/2),:))*0.1)
%         xlabel('z ($\mu$m)','Interpreter','latex')
%         ylabel('y ($\mu$m)','Interpreter','latex')
%         title('Electrical Field on x-z plane')

        subplot(2,2,1)
        imagesc(z,x,squeeze(intensity(:,round(sy/2),:)),[0 1])
        axis image
        set(gca,'ydir','normal')
        colormap(gca,hot)
        alpha(1-squeeze(sensor(:,round(sy/2),:))*0.2)
        xlabel('z ($\mu$m)','Interpreter','latex')
        ylabel('x ($\mu$m)','Interpreter','latex')
        title('Intensity on x-z plane')

        subplot(2,2,3)
        imagesc(z,y,squeeze(intensity(round(sx/2),:,:)),[0 1])
        axis image
        set(gca,'ydir','normal')
        colormap(gca,hot)
        alpha(1-squeeze(sensor(round(sx/2),:,:))*0.2)
        xlabel('z ($\mu$m)','Interpreter','latex')
        ylabel('y ($\mu$m)','Interpreter','latex')
        title('Intensity on y-z plane')

        subplot(2,2,2)
        imagesc(z,x,squeeze(angle(sum(E(:,round(sy/2),:,:),4)/exp(1i*omega*dt*ki))),[-1 1].*pi()) 
        axis image
        set(gca,'ydir','normal' )
        colormap(gca,emkc)
        alpha(1-squeeze(sensor(:,round(sy/2),:))*0.2)
        xlabel('z ($\mu$m)','Interpreter','latex')
        ylabel('x ($\mu$m)','Interpreter','latex')
        title('Phase on x-z plane')

        subplot(2,2,4)
        imagesc(z,y,squeeze(angle(sum(E(round(sx/2),:,:,:),4)/exp(1i*omega*dt*ki))),[-1 1].*pi())
        axis image
        set(gca,'ydir','normal' )
        colormap(gca,emkc)
        alpha(1-squeeze(sensor(round(sx/2),:,:))*0.2)
        xlabel('z ($\mu$m)','Interpreter','latex')
        ylabel('y ($\mu$m)','Interpreter','latex')
        title('Phase on y-z plane')

        sgtitle(sprintf('timeFrame = %d',ki))

        
        % set(gcf, 'Position',pos+[0 -500 0 500])
        set(gcf, 'Position',[1 49 1536 740.8])

        drawnow      
        
        if movvar==1
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
        end
        tic
    end
    
end

% Ensure the final field state is captured for downstream analysis.
wait(GPU1);
GEx = ifftn(GYkdy.*fftn(GHz2) - GZkdz.*fftn(GHy2));
GEy = ifftn(GZkdz.*fftn(GHx2) - GXkdx.*fftn(GHz2));
GEz = ifftn(GXkdx.*fftn(GHy2) - GYkdy.*fftn(GHx2));

Ex = gather(GEx);
Ey = gather(GEy);
Ez = gather(GEz);

intensity = abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2;
overallintensity = intensity;

if movvar==1
    close(writerObj);
end
%% Absorbance evaluation for the sensing unit
sensorPresence = squeeze(any(any(sensor,1),2));

if any(sensorPresence)
    firstSensorIdx = find(sensorPresence, 1, 'first');
    lastSensorIdx = find(sensorPresence, 1, 'last');

    preIdx = max(firstSensorIdx - 1, 1);
    postIdx = min(lastSensorIdx + 1, sz);

    sensorFootprint = squeeze(any(sensor, 3));
    if ~any(sensorFootprint(:))
        sensorFootprint = true(sx, sy);
    end

    footprintCount = nnz(sensorFootprint);

    incidentSlice = intensity(:, :, preIdx);
    transmittedSlice = intensity(:, :, postIdx);

    incidentMean = sum(incidentSlice(sensorFootprint)) / footprintCount;
    transmittedMean = sum(transmittedSlice(sensorFootprint)) / footprintCount;

    denom = max(incidentMean, realmin);
    transmittance = transmittedMean / denom;
    absorbance = -log10(max(transmittance, realmin));

    fprintf('Average incident intensity: %g\n', incidentMean);
    fprintf('Average transmitted intensity: %g\n', transmittedMean);
    fprintf('Transmittance through sensor: %g\n', transmittance);
    fprintf('Absorbance through sensor: %g\n', absorbance);

    voxelCounts = squeeze(sum(sum(sensor, 1), 2));
    sensorIntensity = squeeze(sum(sum(intensity .* sensor, 1), 2));

    intensityProfile = NaN(sz, 1);
    validVoxels = voxelCounts > 0;
    intensityProfile(validVoxels) = sensorIntensity(validVoxels) ./ voxelCounts(validVoxels);

    absorbanceProfile = NaN(sz, 1);
    absorbanceProfile(validVoxels) = -log10(max(intensityProfile(validVoxels) ./ denom, realmin));

    absorbanceVolume = NaN(size(intensity));
    sensorIntensityValues = intensity(sensor);
    absorbanceValues = -log10(max(sensorIntensityValues ./ denom, realmin));
    absorbanceVolume(sensor) = absorbanceValues;

    figure;
    validIdx = find(validVoxels);
    plot(z(validIdx) * 1e6, absorbanceProfile(validIdx), 'LineWidth', 1.5);
    grid on;
    xlabel('z ($\mu$m)', 'Interpreter', 'latex');
    ylabel('Absorbance', 'Interpreter', 'latex');
    title('Absorbance profile through the sensing unit');

    figure;
    subplot(1, 2, 1);
    absorbanceSliceXZ = squeeze(absorbanceVolume(:, round(sy/2), :));
    alphaSliceXZ = squeeze(sensor(:, round(sy/2), :));
    imagesc(z, x, absorbanceSliceXZ, 'AlphaData', alphaSliceXZ);
    axis image;
    set(gca, 'ydir', 'normal');
    colormap(gca, hot);
    colorbar;
    xlabel('z ($\mu$m)', 'Interpreter', 'latex');
    ylabel('x ($\mu$m)', 'Interpreter', 'latex');
    title('Absorbance on x-z cross-section');

    subplot(1, 2, 2);
    absorbanceSliceYZ = squeeze(absorbanceVolume(round(sx/2), :, :));
    alphaSliceYZ = squeeze(sensor(round(sx/2), :, :));
    imagesc(z, y, absorbanceSliceYZ, 'AlphaData', alphaSliceYZ);
    axis image;
    set(gca, 'ydir', 'normal');
    colormap(gca, hot);
    colorbar;
    xlabel('z ($\mu$m)', 'Interpreter', 'latex');
    ylabel('y ($\mu$m)', 'Interpreter', 'latex');
    title('Absorbance on y-z cross-section');
else
    warning('Sensing unit geometry is empty; absorbance cannot be evaluated.');
    absorbance = NaN;
    transmittance = NaN;
    absorbanceProfile = [];
end
%% Save the exposure intensity for hologram reading simulation
% FileName = sprintf('%s_exposure_intensity.mat',date);
% save(FileName,'-mat','-v7.3','-nocompression')
