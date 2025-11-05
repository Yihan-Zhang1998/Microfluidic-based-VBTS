%% 2D simulation of ray optics
% Always utilize SI units
clear;
clc;
close all;
date = datestr(now,'yyyymmdd_HHMMSS');
gpuDeviceTable
GPU1 = gpuDevice(1);
%% Import image for hologram recording
msource = imresize(imread('msource.png'),1);
sensor = imresize(imread('sensing_unit_K1D4.png'),1);
pdms = imresize(imread('PDMS.png'),1);
%% Initialize video
movvar=1;
fps=30;
if movvar==1
    writerObj = VideoWriter(sprintf('%s_test1.mp4',date),'MPEG-4' );
    writerObj.FrameRate = fps;
    open(writerObj);
end
%% Universal constants
epsilon0=8.854187817e-12;
mu0=12.566370e-7;
c=1/sqrt(epsilon0*mu0);
%% Plot definitions
brightness=1;  % brightness of plot
nt=5000; % total number os time steps
waitdisp=20; % wait to display
lengthx=25e-3;  % size of the image in x axis SI
lengthy=lengthx;  % size of the image in y axis SI
[sy,sx]=size(msource(:,:,1));
%% Source constants
wavelength=450e-9;
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
%% Refractive Indices and Objects Definition
obj1 = sensor;
obj1 = round(double(obj1(:,:,2))/255);
obj2 = pdms;
obj2 = round(double(obj2(:,:,1))/255);
RI0 = 1.0003;  % Background refractive index (air)
RI1 = 1.33334+1i*2.5e-2;    % Refractive index of sensor % This needs to be adjusted.
RI2 = 1.4; % Refractive index of PDMS
RIall = RI0*ones(size(X));
RIall(obj1==1) = RI1;
RIall(obj2==1) = RI2;
n_all = real(RIall);
% Absorption
kappa_all = imag(RIall);
alpha_all = 4*pi*kappa_all/wavelength;
%% Edge Smoothing (Boundary Layer)
sigma_px =2;
n_all_s = imgaussfilt(n_all, sigma_px);
%% Plot the Geometry
figure;
subplot(1,2,1);
imagesc(x,y,n_all);
axis image
set(gca,'ydir','normal' )
% colormap(gca,emkc)
alpha(1-obj1*0.2-obj2*0.1)
subplot(1,2,2);
imagesc(x,y,alpha_all);
axis image
set(gca,'ydir','normal' )
% colormap(gca,emkc)
alpha(1-obj1*0.2-obj2*0.1)
%% Gradients of n(x,y) in physical units
% gradient() returns per-pixel differences; scale to [1/m]
[dn_dy_pix, dn_dx_pix] = gradient(n_all_s);  % note MATLAB: first dim = rows (y), second = cols (x)
dn_dx = dn_dx_pix / dx;
dn_dy = dn_dy_pix / dy;
%% Ray Definition
aperture = 2e-3;
x0 = linspace(-aperture/2, aperture/2, 9);
r0 = [x0; repmat(-20e-3, 1, numel(x0))];
k0 = repmat([0; 1], 1, numel(x0));
normalize = @(v) v./vecnorm(v);
% intersectSphere = @(p,k,C,R) deal((-dot(k,(p-C)) - sqrt( (dot(k,(p-C)))^2 - (vecnorm(p-C)^2 - R^2) )), [] ); % Intersect ray with sphere (2D: x-z treated as 3D with y=0)
refract = @(k_in, n_hat, n1, n2) normalize((n1/n2)*k_in + ( (n1/n2)*(-dot(n_hat,k_in)) - sqrt( 1 - (n1/n2)^2*(1-(-dot(n_hat,k_in))^2) ) ) * n_hat ); % Refract: k_out = (n1/n2)k_in + ( (n1/n2)*cosi - sqrt(1 - (n1/n2)^2*(1-cosi^2)) ) n
reflect = @(k_in, n_hat) normalize( k_in - 2*dot(k_in,n_hat)*n_hat ); 
% Trace
r = r0;
k = k0;
%% Coordinate mappers (world [x,y] <-> pixel [row,col])
% You built x from xmin:dx:xmax and y as a flipped linspace; so:
world2pix = @(xw,yw) deal( ...
    (xw - xmin)/dx + 1, ...                % col index (j)
    (ymax - yw)/dy + 1 );                   % row index (i) because y was flipped
in_bounds = @(xw,yw) (xw>xmin & xw<xmax & yw>ymin & yw<ymax);
%% Initialize rays (copy to arrays we'll mutate)
xr = r(1,:);        % x position [m]
yr = r(2,:);        % y position [m]  (your r0 uses z in 2D; here we treat it as y for the image grid)
% Direction unit vectors from angle in your k (0,1). Convert to components:
kx = k(1,:); kz = k(2,:);                 % using your names; treat kz as +y
tx = kx; ty = kz;                          % rename to (tx,ty) for clarity; must be unit
nrm = sqrt(tx.^2 + ty.^2); tx = tx./nrm; ty = ty./nrm;
I  = ones(size(xr));                       % ray intensities (start at 1)
%% March settings
ds      = min(dx,dy)*0.5;                  % step length in meters (CFL-like; 0.3–1.0 px is good)
nSteps  = 5000;                             % safety cap
sample_every = 2;                           % draw every N steps to video
traj_x = cell(1,numel(xr)); traj_y = traj_x;
%% March
for s = 1:nSteps
    % Sample n and grad n at current positions (bilinear)
    [j, i] = world2pix(xr, yr);            % (j=cols, i=rows)
    % guard: rays out of bounds
    alive = in_bounds(xr, yr);
    if ~any(alive), break; end

    ii = i(alive); jj = j(alive);
    n_here   = interp2(n_all_s, jj, ii, 'linear', n_air);
    dndx     = interp2(dn_dx,  jj, ii, 'linear', 0);
    dndy     = interp2(dn_dy,  jj, ii, 'linear', 0);
    alpha    = interp2(alpha_all, jj, ii, 'linear', 0);

    % ODE: d/ds (n * t_hat) = grad n  =>  update t_hat
    % Discretize: n*(t_new - t_old)/ds ≈ grad n  =>  t_new ≈ t_old + (grad n / n)*ds, then renormalize
    tx(alive) = tx(alive) + (dndx ./ n_here) * ds;
    ty(alive) = ty(alive) + (dndy ./ n_here) * ds;
    % renormalize direction
    L = sqrt(tx.^2 + ty.^2); tx = tx./L; ty = ty./L;

    % Advance positions
    xr(alive) = xr(alive) + tx(alive)*ds;
    yr(alive) = yr(alive) + ty(alive)*ds;

    % Absorption
    I(alive) = I(alive) .* exp(-alpha * ds);

    % Record sparse trajectory for plotting
    for rr = find(alive)
        traj_x{rr}(end+1) = xr(rr);
        traj_y{rr}(end+1) = yr(rr);
    end

    % Draw + video
    if mod(s, sample_every)==0
        if s==sample_every
            figure('Color','w'); imshow(nmap_s,[]); hold on; title('Ray paths over refractive index map');
        else
            cla; imshow(nmap_s,[]); hold on;
        end
        % overlay sensor/PDMS outlines
        visboundaries(bwperim(obj2),'Color','c');    % PDMS edges
        visboundaries(bwperim(obj1),'Color','y');    % sensor edges

        % plot rays (intensity -> alpha)
        for rr = 1:numel(traj_x)
            if ~isempty(traj_x{rr})
                px = (traj_x{rr}-xmin)/dx + 1;
                py = (ymax - traj_y{rr})/dy + 1;
                plot(px, py, '-', 'LineWidth', 1.0, 'Color', [1 0 0 max(0.05, min(1,I(rr)))]);
            end
        end
        drawnow;
        if movvar==1
            frame = getframe(gcf);
            writeVideo(writerObj, frame);
        end
    end
end
%% Final spot analysis on an image/sensor line (example at y = ymax - 10 mm)
y_img = ymax - 10e-3;  % choose your detection plane height
hit = abs(yr - y_img) < ds;                % crude gating near plane
x_hits = xr(hit); I_hits = I(hit);
if ~isempty(x_hits)
    fprintf('Hit count on y=%.1f mm: %d   RMS spot = %.3f mm\n', 1e3*y_img, numel(x_hits), 1e3*std(x_hits));
end
%% Close video
if movvar==1
    close(writerObj);
end




