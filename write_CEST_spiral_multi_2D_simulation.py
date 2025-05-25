"""Variable density spiral sequence."""

# %%
import time
from pathlib import Path
import json

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from utils.create_ismrmrd_header import create_hdr
from utils.vds import vds
from utils.write_seq_definitions import write_seq_definitions

# choose flags
FLAG_GOLDEN_ANGLE = True  # toggle use of golden angle
FLAG_PLOTS = False  # toggle plotting of gradients/trajectory etc
FLAG_TESTREPORT = False  # toggle advanced test report including timing check (SLOW)
FLAG_TIMINGCHECK = False  # toggle timing check (SlOW)

# define filename
molecule_of_interest = "BA_simulation"
offset_file = Rf"SpiralssCEST/utils/molecule_offsets/{molecule_of_interest}.json"

###############
# CEST settings
###############
GAMMA_HZ = 42.5764
b1: float = 1.2  # B1 power average [µT] (cw power equivalent is calculated below)

with open(offset_file) as off_in:
    offsets = json.load(off_in)
    offsets_ppm = offsets["offsets"]
seq_defs: dict = {}

seq_defs['b0'] = 3  # B0 [T]
seq_defs['tp'] = 100e-3  # pulse duration [s]
seq_defs['offsets_ppm'] = np.array(offsets_ppm)
seq_defs["m0_offset"] = np.array([offsets_ppm[0]]) # offset vector [ppm]
seq_defs['num_meas'] = seq_defs['offsets_ppm'].size  # number of repetition
seq_defs['b1'] = b1
############# END OF CEST SETTINGS ###################
# define filename
filename = f'{time.strftime("%Y%m%d")}_{molecule_of_interest}CEST_spiral_multi_2D'

# define geometry parameters
nx = 196  # number of points per spoke
slice_thickness = 10e-3  # slice thickness [m]
n_slices = 1  # number of slices
fov = 256e-3  # field-of-view
slice_indices = np.arange(0, n_slices)
delta_kz = 1 / (slice_thickness * n_slices)
slice_areas = (np.arange(n_slices) - (n_slices / 2)) / slice_thickness
res = fov / nx  # spatial resolution [m]
#%%
# calculate order of slices to loop through
ordered_slice_indices = []
slice_list = np.arange(start=0,stop=n_slices,step=1)
bottom_slices = slice_list[0:len(slice_list)//2]
top_slices = slice_list[len(slice_list)//2:]

loop_counter = 0
if n_slices == 1:
    ordered_slice_indices = slice_list
for slice in range(0,len(slice_list)//2):
    if loop_counter % 2 == 0 or loop_counter == 0:
        ordered_slice_indices.append(bottom_slices[slice])
        ordered_slice_indices.append(top_slices[slice]) 
    else:
        ordered_slice_indices.append(bottom_slices[slice])
        ordered_slice_indices.append(top_slices[slice])
    loop_counter += 1


#%%
# define number of spirals and variable density parameter
n_spirals_for_traj_calculation = 12  # number of shots/interleaves
n_readouts_per_slice = 12 #number of aqcuisitions per slice
fov_scaling = [fov, -fov *3/4]  # [fov, 0] # fov decreases linearly from fov_scaling[0] to fov_scaling[0]-fov_scaling[1].

# number of dummy readouts to enter steady-state before readout
n_dummies_before_first_offset = 0
n_dummies_between_offsets = 100

seq_defs['n_spirals_for_traj_calculation'] = n_spirals_for_traj_calculation
seq_defs['n_readouts_per_slice'] = n_readouts_per_slice
seq_defs['n_dummies_between_offsets'] = n_dummies_between_offsets

# set repetition time
TR = None  # repetition time [s], None for minimum 

# define rf pulse parameters
rf_angle = 10  # flip angle of excitation pulse [°]
rf_duration = 1.28e-3  # duration of excitation pulse [s]
rf_bwt_product = 4  # bandwidth time product of rf pulses. MUST BE THE SAME FOR ALL PULSES !!!
rf_spoiling_inc = 117  # rf spoiling phase increment. Choose 0 to disable rf spoiling. [°]

# create Pypulseq Sequence object and set system limits
system = pp.Opts(
    max_grad=100,
    grad_unit='mT/m',
    max_slew=100,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

# add system limits to seq object
seq = pp.Sequence(system=system)

# calculate angular increment
delta_angle = (
    2 * np.pi * (1 - 2 / (1 + np.sqrt(5))) if FLAG_GOLDEN_ANGLE else 2 * np.pi / n_readouts_per_slice
)

# create slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
    flip_angle=rf_angle * np.pi / 180,
    duration=rf_duration,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=rf_bwt_product,
    system=system,
    return_gz=True,
    use="excitation"
)

# calculate spiral trajectory
r_max = 0.5 / fov * nx  # [1/m]
k, g, s, timing, r, theta = vds(
    smax=system.max_slew * 0.9,
    gmax=system.max_grad * 0.9,
    T=system.grad_raster_time,
    N=n_spirals_for_traj_calculation,
    Fcoeff=fov_scaling,
    rmax=r_max,
    oversampling=12,  # 12,
)

# calculate ADC
adc_total_samples = np.shape(g)[0] -1
# Ensure that adc_total_samples does not exceed the Siemenes ADC limit. Check pulseq documentation for details.
assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'

adc_dwell = system.grad_raster_time
adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, system=system)

seq_defs['n_k0'] = adc_total_samples

# gradient spoiling
A_gz_spoil = 4 / slice_thickness - gz.area / 2
gz_spoil = pp.make_trapezoid(channel='z', area=A_gz_spoil, system=system)
ramp_spoil_dur = max(pp.calc_duration(gz_spoil), system.max_grad / system.max_slew)

# Post Preparation Spoiler
post_spoil_amp = 0.5 * system.max_grad  # Hz/m
post_spoil_rt = 0.5e-3  # spoiler rise time in seconds
post_spoil_dur = 1.5e-3  # complete spoiler duration in seconds
post_spoil_x, post_spoil_y, post_spoil_z = (
    pp.make_trapezoid(
        channel=c,
        system=system,
        amplitude=post_spoil_amp,
        flat_time=post_spoil_dur - 2 * post_spoil_rt,
        rise_time=post_spoil_rt,
    )
    for c in ['x', 'y', 'z']
)

# # # # # # # # # # # # #
# CREATE ISMRMRD HEADER #
# # # # # # # # # # # # #

# define full filename
str_vds = '_vds' if fov_scaling[1] != 0 else ''
sat_time = f'{seq_defs["tp"]}ms'.replace('.', 'p')
filename += f'_{fov * 1000:.0f}mm_{adc_total_samples}k0_{n_slices}slices_{n_readouts_per_slice}interleaves_{sat_time}_{b1:.1f}uT_{n_spirals_for_traj_calculation}spirals_{rf_angle}angle'

# create folder for seq and header file
output_path = Path.cwd() / 'output' / filename
output_path.mkdir(parents=True, exist_ok=True)

# delete existing header file
if (output_path / f'{filename}_header.h5').exists():
    (output_path / f'{filename}_header.h5').unlink()

# create header
hdr = create_hdr(
    traj_type='spiral',
    fov=fov,
    res=res,
    slice_thickness=slice_thickness,
    dt=adc_dwell,
    n_k1=n_spirals_for_traj_calculation,
)

# write header to file
prot = ismrmrd.Dataset(output_path / f'{filename}_header.h5', 'w')
prot.write_xml_header(hdr.toXML('utf-8'))

# choose initial rf phase offset
rf_phase = 0
rf_inc = 0

# RF Saturation pulse
flip_angle_sat = b1 * GAMMA_HZ * 2 * np.pi * seq_defs['tp']
sat_pulse = pp.make_sinc_pulse(
    flip_angle=flip_angle_sat,
    duration=seq_defs['tp'],
    system=system,
    time_bw_product=2,
    apodization=0.15,
)

# convert offsets from ppm to Hz and round to 3 decimals
offsets_hz = seq_defs['offsets_ppm'] * GAMMA_HZ * seq_defs['b0']
offsets_hz = np.around(offsets_hz, decimals=3)

# Pre-calculate the spiral gradient waveforms, k-space trajectories, and rewinders
n_points_g = np.shape(g)[0]
n_points_k = np.shape(k)[0]

spiral_readout_grad = np.zeros((n_readouts_per_slice, 2, n_points_g))
spiral_trajectory = np.zeros((n_readouts_per_slice, 2, n_points_k))
gx_readout_list = []
gy_readout_list = []
gx_rewinder_list = []
gy_rewinder_list = []
rewinder_duration = 0

for n in range(n_readouts_per_slice):
    delta = n * delta_angle
    exp_delta = np.exp(1j * delta)
    exp_delta_pi = np.exp(1j * (delta + np.pi))

    spiral_readout_grad[n, 0, :] = np.real(g * exp_delta)
    spiral_readout_grad[n, 1, :] = np.imag(g * exp_delta)
    spiral_trajectory[n, 0, :] = np.real(k * exp_delta_pi)
    spiral_trajectory[n, 1, :] = np.imag(k * exp_delta_pi)
    
    gx_readout = pp.make_arbitrary_grad(channel='x', waveform=spiral_readout_grad[n, 0], system=system, delay=adc.delay)
    gy_readout = pp.make_arbitrary_grad(channel='y', waveform=spiral_readout_grad[n, 1], system=system, delay=adc.delay)

    gx_rewinder, _, _ = pp.make_extended_trapezoid_area(
        area=-(gx_readout.waveform * system.grad_raster_time).sum(),
        channel='x',
        grad_start=spiral_readout_grad[n, 0, -1],
        grad_end=0,
        system=system,
    )

    gy_rewinder, _, _ = pp.make_extended_trapezoid_area(
        area=-(gy_readout.waveform * system.grad_raster_time).sum(),
        channel='y',
        grad_start=spiral_readout_grad[n, 1, -1],
        grad_end=0,
        system=system,
    )

    gx_readout_list.append(gx_readout)
    gy_readout_list.append(gy_readout)    
    gx_rewinder_list.append(gx_rewinder)
    gy_rewinder_list.append(gy_rewinder)

    rewinder_duration = max(rewinder_duration, pp.calc_duration(gx_rewinder, gy_rewinder))

# set rewinder duration to maximum duration of all rewinder/spoiling gradients in the corresponding block
rewinder_duration = max(rewinder_duration, ramp_spoil_dur)

# pre-calculate gz_pre for time optimization+
max_gz_pre_dur = 0
for slice in range(n_slices):
    gz_pre = pp.make_trapezoid(
        channel="z", system=system, area=slice_areas[slice]
    )
    if pp.calc_duration(gz_pre) > max_gz_pre_dur:
        max_gz_pre_dur = pp.calc_duration(gz_pre)
print(f"Max gz_pre Duration: {max_gz_pre_dur}")

# calculate minimum repetition time (TR)
min_TR = (
    + pp.calc_duration(post_spoil_x)
    + pp.calc_duration(rf, gz)  # rf pulse
    + max_gz_pre_dur #gz_pre
    + pp.calc_duration(gzr)
    + pp.calc_duration(adc)
    + pp.calc_duration(rewinder_duration)
    + system.grad_raster_time  # adc ramp down time
)

min_TR = np.ceil(min_TR / system.grad_raster_time) * system.grad_raster_time  # put on raster
min_TR = np.ceil(min_TR * 1e9) / 1e9  # round to 2 decimal values in ms

# calculate TR delay
tr_delay = 1e-5 if TR is None else np.ceil((TR - min_TR) / system.grad_raster_time) * system.grad_raster_time
assert tr_delay >= 1e-5, f'TR must be larger than {min_TR * 1000:.2f} ms. Current value is {TR * 1000:.2f} ms.'

# print TR values
final_TR = min_TR + tr_delay
print(f'shortest TR = {min_TR * 1000:.2f} ms')
print(f'final TR = {final_TR * 1000:.2f} ms')

# # # # # # # # # # # # # # # # #
# ADD BLOCKS TO SEQUENCE OBJECT #
# # # # # # # # # # # # # # # # #

for m, offset in enumerate(offsets_hz):
    # print progress/offset
    print(f" {m + 1} / {len(offsets_hz)}: offset = {seq_defs['offsets_ppm'][m]} ppm")
    # set sat_pulse
    sat_pulse.freq_offset = offset

    slice = 0 # set artifical counter for gz
    
        # set number of dummy scans
    if m == 0:
        n_dummies = n_dummies_before_first_offset
    else:
        n_dummies = n_dummies_between_offsets
    
    # loop over dummies before & between offsets
    for _ in range(n_dummies):
        # add global sat pulse and spoiler        
        seq.add_block(sat_pulse)
        seq.add_block(post_spoil_x, post_spoil_y, post_spoil_z)

        # add delay with length of slice selective rf excitation pulse
        seq.add_block(pp.make_delay(rf_duration))
        seq.add_block(gz)
        
        # add readout gradients, no ADC
        seq.add_block(gx_readout_list[0], gy_readout_list[0])

        # add rewinder gradients and spoiler
        gx_rewinder = gx_rewinder_list[0]
        gy_rewinder = gy_rewinder_list[0]
        seq.add_block(gx_rewinder, gy_rewinder, gz_spoil)
        
        # calculate rewinder delay for current shot
        current_rewinder_duration = max(pp.calc_duration(gx_rewinder), pp.calc_duration(gy_rewinder))
        rewinder_delay = rewinder_duration - current_rewinder_duration
        
        # add rewinder delay to ensure constant TR over all shots
        seq.add_block(pp.make_delay(rewinder_delay))
        
        if slice != n_slices - 1:
                slice += 1
        else:
            slice = 0 # reset slice counter if reaches size of n_slices
        
    ## Loop over readouts per slice
    for repetition in range(n_readouts_per_slice):
        
        for i in range(16):
       
            ## Loop over slices
            for slice in ordered_slice_indices:
                
                # CEST Saturation
                # seq.add_block(
                #     pp.make_delay(tr_delay),
                #     pp.make_label(label='LIN', type='SET', value=repetition),
                #     pp.make_label(label='PAR', type='SET', value=int(slice)), 
                #     )
                
                if rf_spoiling_inc > 0:
                    sat_pulse.phase_offset = rf_phase / 180 * np.pi
                    adc.phase_offset = rf_phase / 180 * np.pi
                    # set the phase offset for the excitation pulse
                    rf.phase_offset = rf_phase / 180 * np.pi

                rf.freq_offset = gz.amplitude * slice_thickness * (slice - n_slices / 2)

                seq.add_block(sat_pulse)
                seq.add_block(post_spoil_x, post_spoil_y, post_spoil_z)

                ## READOUT
                # add slice selective excitation pulse
                if i == 8:       
                    seq.add_block(rf, gz)
                    # add slice selection re-phasing gradient (gzr)
                    seq.add_block(gzr)
                            
                else:
                    # add delay with length of slice selective rf excitation pulse
                    seq.add_block(pp.make_delay(rf_duration))

                # add readout gradients and ADC

                seq.add_block(gx_readout_list[repetition], gy_readout_list[repetition])#, adc)

                # add rewinder gradients and spoiler
                gx_rewinder = gx_rewinder_list[repetition]
                gy_rewinder = gy_rewinder_list[repetition]
                seq.add_block(gx_rewinder, gy_rewinder, gz_spoil)
                
                # calculate rewinder delay for current shot
                current_rewinder_duration = max(pp.calc_duration(gx_rewinder), pp.calc_duration(gy_rewinder))
                rewinder_delay = rewinder_duration - current_rewinder_duration

                # add rewinder delay to ensure constant TR over all shots
                seq.add_block(pp.make_delay(rewinder_delay))
                
                # update rf phase offset for the next saturation pulse for "next round"
                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
                
                # add TR delay to ensure the desired TR is achieved
                seq.add_block(pp.make_delay(tr_delay))
                            
                # add acquisitions to metadata
                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
                traj_ismrmrd = np.stack(
                    [
                        spiral_trajectory[repetition, 0, 0:-1] * fov,
                        spiral_trajectory[repetition, 1, 0:-1] * fov,
                    ]
                )
                acq.traj[:] = traj_ismrmrd.T
                prot.append_acquisition(acq)

    #end of offset
    seq.add_block(adc)
            
# close ISMRMRD file
prot.close()

# check timing of the sequence
if FLAG_TIMINGCHECK and not FLAG_TESTREPORT:
    ok, error_report = seq.check_timing()
    if ok:
        print('\nTiming check passed successfully')
    else:
        print('\nTiming check failed! Error listing follows\n')
        print(error_report)

# show advanced rest report
if FLAG_TESTREPORT:
    print('\nCreating advanced test report...')
    print(seq.test_report())

# write additional definitions
for k, v in seq_defs.items():
    seq.set_definition(k, v)

# write all required parameters in the seq-file definitions
write_seq_definitions(
    seq=seq,
    fov=fov,
    slice_thickness=slice_thickness,
    name=filename,
    alpha=rf_angle,
    Nx=nx,
    sampling_scheme='spiral',
    N_slices=n_slices,
    Nr=n_spirals_for_traj_calculation,
    TE=1e-3,
    TR=final_TR,
    delta=delta_angle,
)

# write required readout parameters to definitions
seq.set_definition(key='FOV', value=[fov, fov, slice_thickness*n_slices])

# %%
if FLAG_PLOTS:
    # Plot sequence
    plt.figure()
    seq.plot(time_range=(0, 17))  ###show blocks
    # Plot trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    plt.figure()
    plt.plot(np.transpose(k_traj))
    plt.figure()
    plt.plot(k_traj[0], k_traj[1], 'b')
    plt.plot(k_traj_adc[0], k_traj_adc[1], 'r.')
    plt.show()

    plt.plot(
        k_traj_adc[0, : int(k_traj_adc.shape[1] / n_slices)], k_traj_adc[1, : int(k_traj_adc.shape[1] / n_slices)], 'r.'
    )
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.show()

# save seq-file
print(f"\nSaving sequence file '{filename}.seq' in 'output' folder.")
print(output_path)
seq.write(str(output_path / filename), create_signature=True)

# %%
