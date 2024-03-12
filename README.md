# Neptune
Open source low-latency OFDM data communications link for drones and aerospace.

See FlexLink folder for published versions of the FlexLink specification. Contact Leonard on Slack to get involved. 

See below for the current version of the Trueflight specifications, a simplified version with 60 kHz subcarrier spacing. 

## Neptune Trueflight Requirements

1) The primary link design requirement is low latency. Optimizations for low latency are weighted heavier than other metrics when trade-offs are considered.
  
2) High reliability is expected in the following multi path environments.
   
Extended Pedestrian A model (EPA)

Extended Vehicular A model (EVA)

Extended Typical Urban model (ETU)

3) High reliability is expected using base station high speed train scenario.

4) High reliability is expected using the AWGN moving propagation model.

5) Data link can be resumed if interrupted or lost.

## Neptune Trueflight Specifications

Specifications are stated [with related requirements in brackets]

1) subcarrier spacing 60 kHz (30 kHz, 15 kHz) [Requirement 1, 2, 3, 4]

"wider subcarrier spacings are suitable for low-latency and high-reliability critical applications such as autonomous driving and UAVs, whereas narrower subcarrier spacing is suitable for low data rate or machine-type communications such as narrowband IoT applications. In addition, a wider subcarrier spacing is essential for applications operating at higher frequency bands, such as millimeter wave (mmWave), to alleviate the Doppler spreads for high-mobility scenarios." (6G Massive Radio Access Networks: Key Issues Technologies, and Future Challenges)

"NR V2X with subcarrier spacing of 60 kHz enjoys considerable gains at high velocities (280, 500 kmph) compared to that with subcarrier spacing of 15 kHz." (6G for Vehicle-to-Everything (V2X) Communications: Enabling Technologies, Challenges, and Opportunities)

"NR has a number of novel features. The first is flexible sub-carrier spacing, which can be 2^n multiples of 15 KHz for n an integer in the range of 0–5. Known as 5G NR numerology, different sub-carrier spacing can be used to meet different requirements of autonomous driving, such as high mobility and low latency. ...large numerology helps reduce inter-carrier interference caused by the Doppler effect but makes it more vulnerable to inter-symbol interference due to multipath propagation in a V2X scenario." (Autonomous Vehicles Enabled by the Integration of IoT, Edge Intelligence, 5G, and Blockchain)

Using flexible subcarrier spacing, the choices are 15, 30, 60, 120, 240, and 480 kHz. A rule of thumb based on band conditions, 15 kHZ to 60 kHz is used in carrier channels below 6 GHz, and 60 kHz to 120 kHz is used for higher frequency bands. Larger subcarrier spacings mean lower latency and higher frequencies.

- [ ] investigate 60 kHz as a subcarrier spacing for Neptune in order to maximize low latency performance. Publish a video summarizing results. 
 
