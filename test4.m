%% 2D simulation of ray optics
% Always utilize SI units
% In test4, we are going to consider the effect of reflection, beam
% interaction. A screen detector was added.
clear;
clc;
close all;
date = datestr(now,'yyyymmdd_HHMMSS');
%% GPU
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
coating_cpu = readOrGenerateMask('reflection_coating.png');

msource = toDevice(msource_cpu);
sensor  = toDevice(sensor_cpu);
pdms    = toDevice(pdms_cpu);
coating = toDevice(coating_cpu);
%% Initialize video
movvar=1;
fps=30;
if movvar==1
    writerObj = VideoWriter(sprintf('%s_test4.mp4',date),'MPEG-4' );
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
obj3 = coating(:,:,3) > 0.5;
RI0 = 1.0003;  % Background refractive index (air)
RI1 = 1.4158+1i*9.9010e-6;    % Refractive index of sensor % This needs to be adjusted.
RI2 = 1.4; % Refractive index of PDMS
RI3 = 0.18104+1i*3.068099; % Refractive index of gold
RIall = RI0*ones(size(X),'like',X);
RIall(obj1==1) = RI1;
RIall(obj2==1) = RI2;
RIall(obj3==1) = RI3;
n_all = real(RIall);
% Absorption
kappa_all = imag(RIall);
alpha_all = 4*pi*kappa_all/wavelength;
% CPU copies for plotting and interface calculations
n_all_cpu     = fromDevice(n_all);
kappa_all_cpu = fromDevice(kappa_all);
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
% alpha(1-obj1_cpu*0.2-obj2_cpu*0.1)
subplot(1,2,2);
imagesc(x_plot,y_plot,kappa_all_cpu);
axis image
set(gca,'ydir','normal' )
% colormap(gca,emkc)
% alpha(1-obj1_cpu*0.2-obj2_cpu*0.1)
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
% y0 = 0;
y0 = linspace(-aperture/2, aperture/2, 15);
x_start = xmin + 0.4 * lengthx;                 % small offset from the left boundary
r0 = [repmat(x_start, 1, numel(y0)); y0];
k0 = repmat([1; 0], 1, numel(y0));
% Trace
r = r0;
k = k0;
% Screen detector placed slightly to the left of the launch plane so that
% backward-propagating light (e.g., reflections) can be sampled. Keep it
% inside the computational window.
detector_offset = max(2*dx, 0.01 * lengthx);
x_detector = max(xmin + 1.5*dx, x_start - detector_offset);
detector_hits_y = [];
detector_hits_I = [];
detector_hits_color = [];
detector_profile_trans = zeros(size(y_plot));
detector_profile_ref   = zeros(size(y_plot));
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
traj_color = cell(1,numel(xr));
for idx = 1:numel(xr)
    traj_x{idx} = fromDevice(xr(idx));
    traj_y{idx} = fromDevice(yr(idx));
    traj_I{idx} = fromDevice(I(idx));
    traj_color{idx} = [1 0 0];
end
completed_traj_x = {};
completed_traj_y = {};
completed_traj_I = {};
completed_traj_color = {};
irr_map_transmission = zeros(size(n_all_cpu));
irr_map_reflection   = zeros(size(n_all_cpu));
%% March settings
ds      = min(dx,dy)*0.5;                  % step length in meters (CFL-like; 0.3–1.0 px is good)
nSteps  = 230;                             % safety cap
sample_every = 2;                           % draw every N steps to video
intensity_threshold = 1e-3;                % discard rays once their intensity drops below this level

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
    atten_cross_positions = [];
    atten_cross_values    = [];
    ds_old_list           = [];
    ds_new_list           = [];
    alpha_prev_list       = [];
    trans_coeff_list      = [];
    new_rays = struct('xr',{},'yr',{},'tx',{},'ty',{},'I',{},'n',{}, ...
        'traj_x',{},'traj_y',{},'traj_I',{},'color',{});
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
            k_in_unit = k_in / max(norm(k_in), eps);
            cos_theta_i = -dot(normal_vec, k_in_unit);
            [k_out, tir] = snellVector(k_in, normal_vec, n1, n2);
            k_out_gpu = toDevice(k_out);
            tx(rr) = k_out_gpu(1);
            ty(rr) = k_out_gpu(2);
            prev_x_val = prev_x(cross_idx(idxCross));
            prev_y_val = prev_y(cross_idx(idxCross));
            prev_x_cpu = fromDevice(prev_x_val);
            prev_y_cpu = fromDevice(prev_y_val);
            xr(rr) = prev_x_val + k_out_gpu(1)*ds;
            yr(rr) = prev_y_val + k_out_gpu(2)*ds;
            xr_cpu = fromDevice(xr(rr));
            yr_cpu = fromDevice(yr(rr));
            if tir
                n_next_cross(idxCross) = n1;
                alpha_next(cross_idx(idxCross)) = alpha_here(cross_idx(idxCross));
                traj_color{rr} = [1 0 0];
            end
            tir_flags(idxCross) = tir;

            % Estimate partial path before hitting the interface for absorption logging
            alpha_prev_val = fromDevice(alpha_here(cross_idx(idxCross)));
            I_before = fromDevice(I(rr));
            frac_old = estimateInterfaceFraction(prev_x_cpu, prev_y_cpu, xr_cpu, yr_cpu, ...
                obj1_cpu, obj2_cpu, xmin, ymin, dx, dy, ymax);
            ds_old = frac_old * ds;
            ds_new = (1 - frac_old) * ds;
            if ds_old > 0
                                interface_pt = [prev_x_cpu; prev_y_cpu] + k_in_unit * ds_old;
                I_at_interface = I_before * exp(-alpha_prev_val * ds_old);
                traj_x{rr}(end+1) = interface_pt(1);
                traj_y{rr}(end+1) = interface_pt(2);
                traj_I{rr}(end+1) = I_at_interface;
            else
                interface_pt = [prev_x_cpu; prev_y_cpu];
                I_at_interface = I_before;
            end
            if tir
                trans_coeff = 1;
                R_coeff = 0;
            else
                [R_coeff, trans_coeff] = fresnelCoefficients(n1, n2, cos_theta_i);
                trans_coeff = min(max(trans_coeff, 0), 1);
            end
            atten_cross_positions(end+1) = cross_idx(idxCross);
            ds_old_list(end+1)           = ds_old;
            ds_new_list(end+1)           = ds_new;
            alpha_prev_list(end+1)       = alpha_prev_val;
            trans_coeff_list(end+1)      = trans_coeff;

            if ~tir && R_coeff > 1e-6
                reflect_dir = reflectVector(k_in_unit, normal_vec);
                step_reflect = max(ds_new, 0);
                step_reflect = max(step_reflect, 1e-9);
                new_pos = interface_pt + reflect_dir * step_reflect;
                I_ref_start = I_at_interface * R_coeff;
                if I_ref_start < intensity_threshold
                    continue;
                end
                I_ref_end = I_ref_start * exp(-alpha_prev_val * step_reflect);
                if I_ref_end < intensity_threshold
                    continue;
                end
                new_entry.xr = new_pos(1);
                new_entry.yr = new_pos(2);
                new_entry.tx = reflect_dir(1);
                new_entry.ty = reflect_dir(2);
                new_entry.I  = I_ref_end;
                new_entry.n  = n1;
                new_entry.traj_x = [interface_pt(1), new_pos(1)];
                new_entry.traj_y = [interface_pt(2), new_pos(2)];
                new_entry.traj_I = [I_ref_start, I_ref_end];
                new_entry.color  = [1 0 0];
                new_rays(end+1) = new_entry;
            end
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
        if ~isempty(atten_cross_positions)
            alpha_prev_vals = alpha_prev_list(:);
            ds_old_vals = ds_old_list(:);
            ds_new_vals = ds_new_list(:);
            alpha_next_vals = fromDevice(alpha_next(atten_cross_positions));
            trans_coeff_vals = trans_coeff_list(:);
            atten_cross_values = trans_coeff_vals .* exp(-alpha_prev_vals .* ds_old_vals - alpha_next_vals(:) .* ds_new_vals);
        end
    end

    n_medium(active_idx) = n_next;

    % Check whether any rays cross the detector plane on this step.
    prev_x_cpu = fromDevice(prev_x);
    prev_y_cpu = fromDevice(prev_y);
    new_x_cpu = fromDevice(xr(active_idx));
    new_y_cpu = fromDevice(yr(active_idx));
    tx_cpu    = fromDevice(tx(active_idx));
    I_prev_cpu = fromDevice(I(active_idx));
    alpha_prev_cpu = fromDevice(alpha_here);

    crossed_left = (prev_x_cpu > x_detector) & (new_x_cpu <= x_detector) & (tx_cpu < 0);
    if any(crossed_left)
        cross_idx = find(crossed_left);
        denom = prev_x_cpu(cross_idx) - new_x_cpu(cross_idx);
        denom(abs(denom) < eps) = eps;
        frac = (prev_x_cpu(cross_idx) - x_detector) ./ denom;
        frac = min(max(frac, 0), 1);
        ds_cross = frac * ds;
        y_cross = prev_y_cpu(cross_idx) + (new_y_cpu(cross_idx) - prev_y_cpu(cross_idx)) .* frac;
        I_cross = I_prev_cpu(cross_idx) .* exp(-alpha_prev_cpu(cross_idx) .* ds_cross);

        detector_hits_y = [detector_hits_y; y_cross(:)]; %#ok<AGROW>
        detector_hits_I = [detector_hits_I; I_cross(:)]; %#ok<AGROW>

        row_idx = round((ymax - y_cross) / dy + 1);
        for detIdx = 1:numel(cross_idx)
            rr = active_idx(cross_idx(detIdx));
            base_color = traj_color{rr};
            if isempty(base_color)
                base_color = [1 0 0];
            end
            detector_hits_color = [detector_hits_color; base_color(:)']; %#ok<AGROW>
            row = row_idx(detIdx);
            if row < 1 || row > numel(y_plot)
                continue;
            end
            if base_color(1) >= base_color(3)
                detector_profile_trans(row) = detector_profile_trans(row) + I_cross(detIdx);
            else
                detector_profile_ref(row)   = detector_profile_ref(row)   + I_cross(detIdx);
            end
        end
    end

    % Absorption with interface-aware path lengths
    alpha_here_cpu = fromDevice(alpha_here);
    alpha_here_cpu = alpha_here_cpu(:);
    atten_cpu = exp(-alpha_here_cpu * ds);
    for idxCross = 1:numel(atten_cross_positions)
        pos = atten_cross_positions(idxCross);
        atten_cpu(pos) = atten_cross_values(idxCross);
    end
    atten_gpu = toDevice(atten_cpu);
    I(active_idx) = I(active_idx) .* atten_gpu';

    % Accumulate irradiance assuming linear optics (crossing rays superpose)
    % We purposefully do not modify directions when beams cross; instead we sum
    % their intensities in the sampling grid because in a linear medium the
    % fields do not exchange energy directly. This addresses the "shouldn't
    % they interact" concern by recording how overlap increases local power.
    [j_pix, i_pix] = world2pix(xr(active_idx), yr(active_idx));
    i_pix_cpu = round(fromDevice(i_pix));
    j_pix_cpu = round(fromDevice(j_pix));
    I_pix_cpu = fromDevice(I(active_idx));
    for idxActive = 1:numel(active_idx)
        ii_idx = i_pix_cpu(idxActive);
        jj_idx = j_pix_cpu(idxActive);
        if ii_idx < 1 || ii_idx > size(irr_map_transmission,1) || ...
           jj_idx < 1 || jj_idx > size(irr_map_transmission,2)
            continue;
        end
        base_color = traj_color{active_idx(idxActive)};
        if isempty(base_color)
            base_color = [1 0 0];
        end
        if base_color(1) >= base_color(3)
            irr_map_transmission(ii_idx, jj_idx) = irr_map_transmission(ii_idx, jj_idx) + I_pix_cpu(idxActive);
        else
            irr_map_reflection(ii_idx, jj_idx)   = irr_map_reflection(ii_idx, jj_idx)   + I_pix_cpu(idxActive);
        end
    end

    % Record sparse trajectory for plotting
    for rr = active_idx
        traj_x{rr}(end+1) = fromDevice(xr(rr));
        traj_y{rr}(end+1) = fromDevice(yr(rr));
        traj_I{rr}(end+1) = fromDevice(I(rr));
    end

    if ~isempty(new_rays)
        for nr = 1:numel(new_rays)
            xr = [xr, toDevice(new_rays(nr).xr)]; %#ok<AGROW>
            yr = [yr, toDevice(new_rays(nr).yr)]; %#ok<AGROW>
            tx = [tx, toDevice(new_rays(nr).tx)]; %#ok<AGROW>
            ty = [ty, toDevice(new_rays(nr).ty)]; %#ok<AGROW>
            I  = [I,  toDevice(new_rays(nr).I)]; %#ok<AGROW>
            n_medium = [n_medium, toDevice(new_rays(nr).n)]; %#ok<AGROW>
            traj_x{end+1} = new_rays(nr).traj_x;
            traj_y{end+1} = new_rays(nr).traj_y;
            traj_I{end+1} = new_rays(nr).traj_I;
            traj_color{end+1} = new_rays(nr).color;
        end
    end

    % Remove rays whose intensity has diminished below the threshold
    keep_gpu = I >= intensity_threshold;
    keep_host = fromDevice(keep_gpu);
    if ~all(keep_host)
        drop_idx = find(~keep_host);
        completed_traj_x = [completed_traj_x, traj_x(drop_idx)];
        completed_traj_y = [completed_traj_y, traj_y(drop_idx)];
        completed_traj_I = [completed_traj_I, traj_I(drop_idx)]; 
        completed_traj_color = [completed_traj_color, traj_color(drop_idx)];
        xr = xr(keep_gpu);
        yr = yr(keep_gpu);
        tx = tx(keep_gpu);
        ty = ty(keep_gpu);
        I  = I(keep_gpu);
        n_medium = n_medium(keep_gpu);
        traj_x = traj_x(keep_host);
        traj_y = traj_y(keep_host);
        traj_I = traj_I(keep_host);
        traj_color = traj_color(keep_host);
    end

    % Draw + video
    if mod(s, sample_every)==0
        if s==sample_every
            figure('Color','w');
            imagesc(x, y, n_all_cpu);
            axis image; set(gca,'YDir','normal'); hold on;
            title('Ray paths over refractive index map');
        else
            cla;
            imagesc(x, y, n_all_cpu);
            axis image; set(gca,'YDir','normal'); hold on;
        end
        % overlay sensor/PDMS outlines
        contour(x_plot, y_plot, double(obj2_cpu), [0.5 0.5], 'c', 'LineWidth', 1.0);    % PDMS edges
        contour(x_plot, y_plot, double(obj1_cpu), [0.5 0.5], 'y', 'LineWidth', 1.0);    % sensor edges

        % plot rays (intensity -> alpha)
        plot_traj_x = [completed_traj_x, traj_x];
        plot_traj_y = [completed_traj_y, traj_y];
        plot_traj_I = [completed_traj_I, traj_I];
        plot_traj_color = [completed_traj_color, traj_color];
        for rr = 1:numel(plot_traj_x)
            xx = plot_traj_x{rr};
            yy = plot_traj_y{rr};
            ii = plot_traj_I{rr};
            base_color = plot_traj_color{rr};
            if isempty(base_color)
                base_color = [1 0 0];
            end
            base_color = max(min(base_color, 1), 0);
            if numel(xx) > 1
                for seg = 1:numel(xx)-1
                    seg_intensity = max(0, min(1, mean(ii(seg:seg+1))));
                    seg_color = seg_intensity * base_color;
                    line(xx(seg:seg+1), yy(seg:seg+1), 'LineWidth', 1.0, ...
                        'Color', seg_color);
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

% Visualize accumulated irradiance from crossing rays
figure('Color','w');
subplot(1,3,1);
imagesc(x, y, irr_map_transmission);
axis image; set(gca,'YDir','normal');
title('Transmitted irradiance');
colorbar;

subplot(1,3,2);
imagesc(x, y, irr_map_reflection);
axis image; set(gca,'YDir','normal');
title('Reflected irradiance');
colorbar;

subplot(1,3,3);
imagesc(x, y, irr_map_transmission + irr_map_reflection);
axis image; set(gca,'YDir','normal');
title('Total irradiance (superposition)');
colorbar;

% Visualize what the detector plane records.
detector_total = detector_profile_trans + detector_profile_ref;
figure('Color','w');
hold on;
plot(detector_total, y_plot * 1e3, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Total');
plot(detector_profile_trans, y_plot * 1e3, 'r--', 'LineWidth', 1.0, 'DisplayName', 'Transmitted contributions');
plot(detector_profile_ref,   y_plot * 1e3, 'b-.', 'LineWidth', 1.0, 'DisplayName', 'Reflected contributions');
if ~isempty(detector_hits_I)
    scatter(detector_hits_I, detector_hits_y * 1e3, 25, detector_hits_color, 'filled', 'DisplayName', 'Recorded rays');
end
set(gca,'YDir','normal');
xlabel('Accumulated intensity at detector (a.u.)');
ylabel('y position on detector (mm)');
title(sprintf('Detector plane at x = %.2f mm', 1e3 * x_detector));
legend('Location','best');
hold off;

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
function [R, T] = fresnelCoefficients(n1, n2, cos_theta_i)
%FRESNELCOEFFICIENTS Unpolarized Fresnel reflectance and transmittance.
n1r = real(n1);
n2r = real(n2);
cos_theta_i = max(min(real(cos_theta_i), 1), 0);
eta = n1r / max(n2r, eps);
sin2_theta_t = eta^2 * max(0, 1 - cos_theta_i^2);
if sin2_theta_t >= 1
    R = 1;
    T = 0;
    return;
end
cos_theta_t = sqrt(max(0, 1 - sin2_theta_t));
rs = (n1r * cos_theta_i - n2r * cos_theta_t) / max(n1r * cos_theta_i + n2r * cos_theta_t, eps);
rp = (n2r * cos_theta_i - n1r * cos_theta_t) / max(n2r * cos_theta_i + n1r * cos_theta_t, eps);
R = 0.5 * (abs(rs)^2 + abs(rp)^2);
R = min(max(R, 0), 1);
T = 1 - R;
end

function v_ref = reflectVector(k_in, normal_vec)
%REFLECTVECTOR Compute the mirror reflection of direction k_in about normal_vec.
k_in = k_in(:) / max(norm(k_in), eps);
normal_vec = normal_vec(:) / max(norm(normal_vec), eps);
v_ref = k_in - 2 * dot(k_in, normal_vec) * normal_vec;
v_ref = v_ref / max(norm(v_ref), eps);
end

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

function frac = estimateInterfaceFraction(x0, y0, x1, y1, obj1, obj2, xmin, ymin, dx, dy, ymax)
%ESTIMATEINTERFACEFRACTION Approximate the fraction of a step spent in the origin medium.
%   Uses mask transitions to detect where the interface lies along the segment
%   connecting (x0,y0) to (x1,y1). Returns a value in [0,1].

masks = {obj1, obj2};
start_vals = zeros(1, numel(masks));
end_vals   = zeros(1, numel(masks));
delta_vals = zeros(1, numel(masks));
for k = 1:numel(masks)
    start_vals(k) = sampleMask(masks{k}, x0, y0, xmin, ymin, dx, dy, ymax);
    end_vals(k)   = sampleMask(masks{k}, x1, y1, xmin, ymin, dx, dy, ymax);
    delta_vals(k) = end_vals(k) - start_vals(k);
end

[~, idx] = max(abs(delta_vals));
delta = delta_vals(idx);
start_val = start_vals(idx);

if abs(delta) < 1e-6
    frac = 0.5;
else
    frac = (0.5 - start_val) / delta;
    frac = min(max(frac, 0), 1);
end

end

function val = sampleMask(mask, xw, yw, xmin, ymin, dx, dy, ymax)
%SAMPLEMASK Bilinear sample of a binary mask at world coordinates.
col = (xw - xmin)/dx + 1;
row = (ymax - yw)/dy + 1;
val = interp2(double(mask), col, row, 'linear', 0);
end

