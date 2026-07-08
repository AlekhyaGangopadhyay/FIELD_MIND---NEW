Title: Production blast vibration data from Mount Erzberg iron ore mine, Austria

Authors: Bernd Trabi and Florian Bleibinhaus, Montanuniversität Leoben, Austria

Location: Erzberg, Eastern Alps, Austria

Description:
The data are 3-component seismic records of production blasts at the Mount Erzberg iron ore mine acquired in three campaigns in 2019-2020 with an array of 81 sensors.

Data file: blasts.sgy
File Format: SEG-Y
Sampling Rate: 500 Hz (dt = 2000 μs per sample)
Trace Length: 5 s (2500 samples)
Seismic Sensor: SERCEL 3-component Digital acceleration Sensor Units (DSU3-SA)

Data Processing:
- DC-offset removal
- Digital Butterworth low-cut filter (minimum-phase, 0.05/0.5 Hz stop/pass frequency)
- Time-domain numerical integration
- Conversion from counts to Volts, and from Volts to particle velocity [mm/s]

Data header description:
tracr         - trace sequence number in data file
fldr, ep      - production blast sequence number
tracf         - receiver station number
trid          - trid codes (12-Z, 13-EW, 14-NS)
offset        - measured from the hole with the largest charge
gx, gy, gelev - receiver coordinates
sx, sy, selev - coordinate of the blast hole with the largest charge
ns            - number of samples
dt            - sampling rate
year, day, hour, minute, second - inverted time zero (start of record) in local time

Supplementary file:
BLASTS.txt - Lists providing production blast sequence numbers, coordinates (UTM Zone 33N, WGS84), delay time, charge and ignition type of each blasthole.
Each row represents one blasthole fired during a production blast.
Columns:
blast     - Production blast sequence number
sx        - Easting (m)
sy        - Northing (m)
selev     - Elevation (m)
delay     - Delay time after shot start (s)
charge    - Explosive charge per blasthole (kg)
detonator - Ignition Type: Electronic (E) or Non-electric (N)

Notes:
Delay times for non-electric (N) are inaccurate by several milliseconds, including a systematic error of ~3 ms per delay.
For production blasts 10010 and 10018, a subsequent blast was initiated a few seconds after the first one, such that it overlays the recordings of the first blast. The configuration provided in BLASTS.txt refers only to the first blast.
No absolute shot times were available. Time zero was inverted from the recorded seismic data.

Citation / Reference:
Bleibinhaus, F. & Trabi, B., (2023) Source time functions and interference from blast arrays. Geophysical Prospecting, 71, 1325–1337. https://doi.org/10.1111/1365-2478.13365
Trabi, B. & Bleibinhaus, F. (2023) Blast vibration prediction. Geophysical Prospecting, 71, 1312-1324. https://doi.org/10.1111/1365-2478.13361
Trabi, B., (2024) Blast Array Optimization for Vibration Reduction in Heterogeneous Models. PhD Thesis, Montanuniversität Leoben. https://doi.org/10.34901/mul.pub.2025.025
