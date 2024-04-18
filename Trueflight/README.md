# Neptune Trueflight

5 GHz band drone data communications use case.

## Neptune Trueflight Requirements

1) The primary link design requirement is 10mS round-trip delay. Optimizations for low latency are weighted heavier than other metrics when trade-offs are considered.
  
2) High reliability is expected in the following multi path environments.
   
Extended Pedestrian A model (EPA)

Extended Vehicular A model (EVA)

Extended Typical Urban model (ETU)

3) High reliability is expected using base station high speed train scenario.

4) High reliability is expected using the AWGN moving propagation model.

5) Data link can be resumed if interrupted or lost.

6) Primary use case is live video.

## Neptune Trueflight Specifications

Specifications are stated [with related requirements in brackets] with brief discussions, justifications, action items, and outcomes.

1) subcarrier spacing 60 kHz [Requirement 1, 2, 3, 4]

"wider subcarrier spacings are suitable for low-latency and high-reliability critical applications such as autonomous driving and UAVs, whereas narrower subcarrier spacing is suitable for low data rate or machine-type communications such as narrowband IoT applications. In addition, a wider subcarrier spacing is essential for applications operating at higher frequency bands, such as millimeter wave (mmWave), to alleviate the Doppler spreads for high-mobility scenarios." (6G Massive Radio Access Networks: Key Issues Technologies, and Future Challenges)

"NR V2X with subcarrier spacing of 60 kHz enjoys considerable gains at high velocities (280, 500 kmph) compared to that with subcarrier spacing of 15 kHz." (6G for Vehicle-to-Everything (V2X) Communications: Enabling Technologies, Challenges, and Opportunities)

"NR has a number of novel features. The first is flexible sub-carrier spacing, which can be 2^n multiples of 15 KHz for n an integer in the range of 0–5. Known as 5G NR numerology, different sub-carrier spacing can be used to meet different requirements of autonomous driving, such as high mobility and low latency. ...large numerology helps reduce inter-carrier interference caused by the Doppler effect but makes it more vulnerable to inter-symbol interference due to multipath propagation in a V2X scenario." (Autonomous Vehicles Enabled by the Integration of IoT, Edge Intelligence, 5G, and Blockchain)

Using flexible subcarrier spacing, the choices are 15, 30, 60, 120, 240, and 480 kHz. A rule of thumb based on band conditions, 15 kHZ to 60 kHz is used in carrier channels below 6 GHz, and 60 kHz to 120 kHz is used for higher frequency bands. Larger subcarrier spacings mean lower latency and higher frequencies.

60 kHz subcarrier spacing gives 3dB better SNR than 15 kHz subcarrier spacing at 6 GHz. 

- [x] investigate 60 kHz as a subcarrier spacing for Neptune in order to maximize low latency performance. 60 kHz subcarrier spacing was accepted as the specification. 
 
2) signal bandwidth [Requirement 1, 5, 6]

The 5 GHz licensed amateur radio allocation is the target band for Neptune. The allocation in the US (ITU Region 2) is 5650.0 to 5925.0 MHz and spans 275 MHz. There is a published band plan from ARRL which can be found at https://www.arrl.org/band-plan

The 5 GHz band is internationally allocated to amateur radio and amateur satellite use on a secondary basis. This has implications for Neptune. In ITU regions 1 and 3, the amateur radio band is between 5,650 MHz and 5,850 MHz.

The intermittant and mobile nature of drone communications is an advantage to integrating Neptune into the band plan. If satellite, weak signal, and EME are all avoided, this leaves 5761.0 to 5830.0, which is only 69 MHz. If only the satellite subbands are avoided, this leaves 5670.0 - 5830.0, which is 160 MHz. The proposal is to avoid the satellite subbands. Neptune Transmission would be demonstrated on the 100 MHz located from 5670.0 to 5770.0 MHz. These frequencies are available in all ITU regions. 

Band planning for amateur microwave is inherently local. While ARRL publishes a band plan, it does not enforce the plan. Amateur radio band plans are voluntary agreements by operators. Amateur radio operators are expected to listen before transmitting, transmit legal signals, transmit with the lowest power necessary to complete the transmission, and cooperate with other licensed users of the band. When bands are shared and amateur radio is a secondary user, amateurs are not allowed to cause harmful interference to the primary users, and must accept interference from primary users. Since we must accept interference, there is a possibility that Neptune signals will be interfered with. This is one of the reasons for Requiremnt 5: Data link can be resumed if interrupted or lost.

100 MHz is the bandwidth recommendation for frequency range 1 (FR1) segment of 5G. The amateur allocation at 5 GHz falls within FR1. Therefore, justifications for signal bandwidths for 5G should be taken into consideration. 

- [ ] gather feedback from amateur radio band planners about the proposed signal placement.
- [ ] enumerate the expected throughput for 100 MHz.
