%% 2D simulation of ray optics
% Always utilize SI units
% In test2, we made sure that the refraction follows Snell's law by
% defining a function at the end.
clear;
clc;
close all;
date = datestr(now,'yyyymmdd_HHMMSS');
%% GPU (optional)
useGPU = false;
if exist('gpuDeviceCount','file') == 2 && gpuDeviceCount('available') > 0 %#ok<*IMPLC>
    gpuDeviceTable;
    GPU1 = gpuDevice(1);
    useGPU = true;
    fprintf('Using GPU: %s\n', GPU1.Name);
else
    fprintf('No compatible GPU detected or Parallel Computing Toolbox unavailable. Running on CPU.\n');
end
if useGPU
    toDevice = @(x) gpuArray(x);
    fromDevice = @(x) gather(x);
else
    toDevice = @(x) x;
    fromDevice = @(x) x;
end
%% Import image for hologram recording
msource_cpu = readOrGenerateMask('msource.png');
sensor_cpu = readOrGenerateMask('sensing_unit_K1D4.png');
pdms_cpu    = readOrGenerateMask('PDMS.png');

msource = toDevice(msource_cpu);
sensor  = toDevice(sensor_cpu);
pdms    = toDevice(pdms_cpu);
%% Initialize video
movvar=1;
fps=30;
if movvar==1
    writerObj = VideoWriter(sprintf('%s_test2.mp4',date),'MPEG-4' );
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
x_lin = (xmin+dx/2:dx:xmax-dx/2);
y_lin = flip((ymin+dx/2:dy:ymax-dy/2));
[X_cpu,Y_cpu]=meshgrid(x_lin,y_lin);
X = toDevice(X_cpu);
Y = toDevice(Y_cpu);
x = toDevice(x_lin);
y = toDevice(y_lin);
%% Source Mask
source = msource(:,:,1) > 0.5;
%% Refractive Indices and Objects Definition
obj1 = sensor(:,:,2) > 0.5;
obj2 = pdms(:,:,1) > 0.5;
RI0 = 1.0003;  % Background refractive index (air)
RI1 = 1.33334+1i*2.5e-2;    % Refractive index of sensor % This needs to be adjusted.
RI2 = 1.4; % Refractive index of PDMS
RIall = RI0*ones(size(X),'like',X);
RIall(obj1==1) = RI1;
RIall(obj2==1) = RI2;
n_all = real(RIall);
% Absorption
kappa_all = imag(RIall);
alpha_all = 4*pi*kappa_all/wavelength;
% CPU copies for plotting and interface calculations
n_all_cpu     = fromDevice(n_all);
alpha_all_cpu = fromDevice(alpha_all);
obj1_cpu      = fromDevice(obj1);
obj2_cpu      = fromDevice(obj2);
x_plot        = fromDevice(x);
y_plot        = fromDevice(y);
%% Plot the Geometry
figure;
subplot(1,2,1);
imagesc(x_plot,y_plot,n_all_cpu);
axis image
set(gca,'ydir','normal' )
% colormap(gca,emkc)
alpha(1-obj1_cpu*0.2-obj2_cpu*0.1)
subplot(1,2,2);
imagesc(x_plot,y_plot,alpha_all_cpu);
axis image
set(gca,'ydir','normal' )
% colormap(gca,emkc)
alpha(1-obj1_cpu*0.2-obj2_cpu*0.1)
%% Gradients of n(x,y) in physical units
% gradient() returns per-pixel differences; scale to [1/m]
[dn_dx_pix, dn_dy_pix] = gradient(n_all);  % note MATLAB: first dim = rows (y), second = cols (x)
dn_dx = dn_dx_pix / dx;
dn_dy = dn_dy_pix / dy;
% Raw (unsmoothed) gradients to estimate interface normals accurately
dn_dx_cpu      = fromDevice(dn_dx);
dn_dy_cpu      = fromDevice(dn_dy);
figure;
contour(x_plot,y_plot,n_all)
hold on
quiver(x_plot,y_plot,dn_dx_cpu,dn_dy_cpu)
hold off
%% Ray Definition
aperture = 5e-3;
y0 = linspace(-aperture/2, aperture/2, 9);
x_start = xmin + 5*dx;                 % small offset from the left boundary
r0 = [repmat(x_start, 1, numel(y0)); y0];
k0 = repmat([1; 0], 1, numel(y0));
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
xr = toDevice(r(1,:));        % x position [m]
yr = toDevice(r(2,:));        % y position [m]
% Direction unit vectors from angle in your k. Convert to components:
tx = toDevice(k(1,:)); ty = toDevice(k(2,:));
nrm = sqrt(tx.^2 + ty.^2); tx = tx./nrm; ty = ty./nrm;
I  = toDevice(ones(size(xr)));             % ray intensities (start at 1)
n_medium = toDevice(RI0 * ones(size(xr))); % refractive index of current medium per ray
traj_x = cell(1,numel(xr));
traj_y = cell(1,numel(xr));
traj_I = cell(1,numel(xr));
for idx = 1:numel(xr)
    traj_x{idx} = fromDevice(xr(idx));
    traj_y{idx} = fromDevice(yr(idx));
    traj_I{idx} = fromDevice(I(idx));
end
%% March settings
ds      = min(dx,dy)*0.5;                  % step length in meters (CFL-like; 0.3–1.0 px is good)
nSteps  = 5000;                             % safety cap
sample_every = 2;                           % draw every N steps to video

%% March
n_background = toDevice(RI0);
for s = 1:nSteps
    % Sample n and grad n at current positions (bilinear)
    [j, i] = world2pix(xr, yr);            % (j=cols, i=rows)
    % guard: rays out of bounds (allow a margin so rays can enter domain)
    margin_y = 0.5*lengthy;               % 12.5 mm for default geometry
    margin_x = dx;                        % one pixel to tolerate numerical drift
    inside = in_bounds(xr, yr);
    pre_or_post = (xr > xmin - margin_x) & (xr < xmax + margin_x) & ...
                 (yr > ymin - margin_y) & (yr < ymax + margin_y) & ...
                 (ty > 0);
    alive = inside | pre_or_post;
    aliveHost = fromDevice(alive);
    if ~any(aliveHost), break; end

    active_idx = find(alive);
    ii = i(active_idx); jj = j(active_idx);
    n_here = interp2(n_all, jj, ii, 'linear', n_background);
    dndx = interp2(dn_dx,  jj, ii, 'linear', 0);
    dndy = interp2(dn_dy,  jj, ii, 'linear', 0);
    alpha_here = interp2(alpha_all, jj, ii, 'linear', 0);
    n_prev = n_medium(active_idx);

    % Store pre-update directions for Snell-law handling at interfaces
    tx_before = tx;
    ty_before = ty;

    % ODE: d/ds (n * t_hat) = grad n  =>  update t_hat
    % Discretize: n*(t_new - t_old)/ds ≈ grad n  =>  t_new ≈ t_old + (grad n / n)*ds, then renormalize
    tx(active_idx) = tx(active_idx) + (dndx ./ n_here) * ds;
    ty(active_idx) = ty(active_idx) + (dndy ./ n_here) * ds;
    % renormalize direction
    L = sqrt(tx.^2 + ty.^2); tx = tx./L; ty = ty./L;

    % Advance positions
    prev_x = xr(active_idx);
    prev_y = yr(active_idx);
    xr(active_idx) = xr(active_idx) + tx(active_idx)*ds;
    yr(active_idx) = yr(active_idx) + ty(active_idx)*ds;

    % Sample new medium properties after the step
    [j_new, i_new] = world2pix(xr(active_idx), yr(active_idx));
    n_next     = interp2(n_all, j_new, i_new, 'linear', n_background);
    alpha_next = interp2(alpha_all, j_new, i_new, 'linear', 0);

    % Apply Snell's law when crossing an interface
    crossing = abs(n_next - n_prev) > 1e-6;
    crossingHost = fromDevice(crossing);
    if any(crossingHost)
        cross_idx = find(crossingHost);
        jj_new = fromDevice(j_new(crossingHost));
        ii_new = fromDevice(i_new(crossingHost));
        nx = interp2(dn_dx_cpu, jj_new, ii_new, 'linear', 0);
        ny = interp2(dn_dy_cpu, jj_new, ii_new, 'linear', 0);
        norm_mag = hypot(nx, ny);
        valid = norm_mag > 1e-9;
        n_next_cross = fromDevice(n_next(crossingHost));
        tir_flags = false(1, numel(cross_idx));
        for idxCross = 1:numel(cross_idx)
            rr = active_idx(cross_idx(idxCross));
            if ~valid(idxCross)
                continue; % fallback: keep gradient-based direction
            end
            normal_vec = [nx(idxCross); ny(idxCross)] ./ norm_mag(idxCross);
            k_in = fromDevice([tx_before(rr); ty_before(rr)]);
            if dot(normal_vec, k_in) > 0
                normal_vec = -normal_vec;
            end
            n1 = fromDevice(n_prev(cross_idx(idxCross)));
            n2 = n_next_cross(idxCross);
            [k_out, tir] = snellVector(k_in, normal_vec, n1, n2);
            k_out_gpu = toDevice(k_out);
            tx(rr) = k_out_gpu(1);
            ty(rr) = k_out_gpu(2);
            prev_x_val = prev_x(cross_idx(idxCross));
            prev_y_val = prev_y(cross_idx(idxCross));
            xr(rr) = prev_x_val + k_out_gpu(1)*ds;
            yr(rr) = prev_y_val + k_out_gpu(2)*ds;
            if tir
                n_next_cross(idxCross) = n1;
                alpha_next(cross_idx(idxCross)) = alpha_here(cross_idx(idxCross));
            end
            tir_flags(idxCross) = tir;
        end
        transmit_mask = ~tir_flags & valid;
        if any(transmit_mask)
            idx_transmit = cross_idx(transmit_mask);
            rays_transmit = active_idx(idx_transmit);
            [j_tx, i_tx] = world2pix(xr(rays_transmit), yr(rays_transmit));
            j_tx_cpu = fromDevice(j_tx);
            i_tx_cpu = fromDevice(i_tx);
            n_next_vals = interp2(n_all_cpu, j_tx_cpu, i_tx_cpu, 'linear', fromDevice(n_background));
            alpha_next_vals = interp2(alpha_all_cpu, j_tx_cpu, i_tx_cpu, 'linear', 0);
            n_next_cross(transmit_mask) = n_next_vals;
            alpha_next(idx_transmit) = toDevice(alpha_next_vals);
        end
        n_next(cross_idx) = toDevice(n_next_cross);
    end

    n_medium(active_idx) = n_next;

    % Absorption (use average attenuation across the step)
    alpha_step = 0.5*(alpha_here + alpha_next);
    alpha_step = alpha_step(:).';
    I(active_idx) = I(active_idx) .* exp(-alpha_step * ds);

    % Record sparse trajectory for plotting
    for rr = active_idx
        traj_x{rr}(end+1) = fromDevice(xr(rr));
        traj_y{rr}(end+1) = fromDevice(yr(rr));
        traj_I{rr}(end+1) = fromDevice(I(rr));
    end

    % Draw + video
    if mod(s, sample_every)==0
        if s==sample_every
            figure('Color','w');
            imagesc(x, y, n_all_cpu, [1 1.5]);
            axis image; set(gca,'YDir','normal'); hold on;
            title('Ray paths over refractive index map');
        else
            cla;
            imagesc(x, y, n_all_cpu, [1 1.5]);
            axis image; set(gca,'YDir','normal'); hold on;
        end
        % overlay sensor/PDMS outlines
        contour(x_plot, y_plot, double(obj2_cpu), [0.5 0.5], 'c', 'LineWidth', 1.0);    % PDMS edges
        contour(x_plot, y_plot, double(obj1_cpu), [0.5 0.5], 'y', 'LineWidth', 1.0);    % sensor edges

        % plot rays (intensity -> alpha)
        for rr = 1:numel(traj_x)
            xx = traj_x{rr};
            yy = traj_y{rr};
            ii = traj_I{rr};
            if numel(xx) > 1
                for seg = 1:numel(xx)-1
                    seg_intensity = max(0.05, min(1, mean(ii(seg:seg+1))));
                    line(xx(seg:seg+1), yy(seg:seg+1), 'LineWidth', 1.0, ...
                        'Color', seg_intensity*[1 0 0]);
                end
            end
        end
        drawnow;
        if movvar==1
            frame = getframe(gcf);
            writeVideo(writerObj, frame);
        end
    end
end
%% Final spot analysis on an image/sensor line (example at x = xmax - 10 mm)
x_img = xmax - 10e-3;  % choose your detection plane location
hit = fromDevice(abs(xr - x_img) < ds);                % crude gating near plane
yr_cpu = fromDevice(yr);
I_cpu = fromDevice(I);
y_hits = yr_cpu(hit); I_hits = I_cpu(hit);
if ~isempty(y_hits)
    fprintf('Hit count on x=%.1f mm: %d   RMS spot = %.3f mm\n', 1e3*x_img, numel(y_hits), 1e3*std(y_hits));
end
%% Close video
if movvar==1
    close(writerObj);
end
%% 
function [k_out, tir] = snellVector(k_in, normal_vec, n1, n2)
%SNELLVECTOR Compute transmitted (or reflected) ray direction via Snell's law.
%   [K_OUT, TIR] = SNELLVECTOR(K_IN, NORMAL_VEC, N1, N2) returns the unit
%   direction vector after an interface crossing. NORMAL_VEC points toward
%   the incident medium. TIR is true when total internal reflection occurs.

tir = false;
k_in = k_in(:) / norm(k_in);
normal_vec = normal_vec(:) / max(norm(normal_vec), eps);
cos_theta_i = -dot(normal_vec, k_in);
if cos_theta_i < 0
    normal_vec = -normal_vec;
    cos_theta_i = -cos_theta_i;
end

eta = n1 / n2;
sin2_theta_t = eta^2 * max(0, 1 - cos_theta_i^2);
if sin2_theta_t > 1  % Total internal reflection
    k_out = k_in + 2*cos_theta_i*normal_vec;
    tir = true;
else
    cos_theta_t = sqrt(max(0, 1 - sin2_theta_t));
    k_out = eta * k_in + (eta * cos_theta_i - cos_theta_t) * normal_vec;
end
k_out = k_out / norm(k_out);

end

function img = readOrGenerateMask(filename)
%READORGENERATEMASK Read an image if it exists, otherwise create a synthetic mask.
if exist(filename,'file') == 2
    img = imread(filename);
    if size(img,3) == 1
        img = repmat(img,1,1,3);
    end
    return;
end

warning('File %s not found. Using generated placeholder mask.', filename);
sz = 512;
[X,Y] = meshgrid(linspace(-1,1,sz));

switch lower(filename)
    case 'msource.png'
        pattern = exp(-((X.^2 + Y.^2) / 0.15^2));
    case 'sensing_unit_k1d4.png'
        pattern = double(abs(X) < 0.5 & abs(Y) < 0.3);
    case 'pdms.png'
        pattern = double(X.^2 + Y.^2 < 0.75^2);
    otherwise
        pattern = exp(-0.5*((X.^2 + Y.^2)/0.5^2));
end

pattern = uint8(255 * mat2gray(pattern));
img = repmat(pattern,1,1,3);
end


